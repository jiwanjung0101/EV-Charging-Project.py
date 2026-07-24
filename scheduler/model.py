"""
Node assignment and the LP-based EV charging / V2G scheduler.

carbon is a nodal dict {node: {period: g CO2/kWh}}; each function looks up the
schedule of the relevant node. Price and carbon are both normalised by their
maximum (not min-max), so every period stays proportional to its real value.
"""

from __future__ import annotations
import math
import pulp as lp
import pandas as pd


PENALTY: float = 1e6   # cost per unit of carbon-cap slack


def active_slots(ev, time_list: list) -> list:
    return [t for t in time_list if ev.arrival <= t <= ev.departure]

def is_active(ev, t: int) -> bool:
    return ev.arrival <= t <= ev.departure


def _effective_carbon_cap(
    beta: float,
    carbon_cap_fraction: float | None,
) -> float | None:
    """
    Scale the carbon cap by the carbon-preference weight beta:
    beta = 0 disables the cap, beta >= 1 applies it in full, and values in
    between interpolate linearly between uncapped (1.0) and carbon_cap_fraction.
    """
    if carbon_cap_fraction is None:
        return None
    if beta <= 0.0:
        return None
    if beta >= 1.0:
        return carbon_cap_fraction
    return 1.0 - beta * (1.0 - carbon_cap_fraction)


def _all_carbon_values(carbon: dict[str, dict[int, float]]) -> list[float]:
    """Every g CO2/kWh value across all nodes and periods."""
    return [v for nc in carbon.values() for v in nc.values()]


def assign_ev_nodes(
    evs,
    nodal_df:         pd.DataFrame,
    carbon:           dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    time_list:        list[int],
    alpha:            float = 1.0,
    beta:             float = 1.0,
    max_evs_per_node: int | None = None,   # kept for API compat, ignored
    grid_cap_kw:      float = 85.0,
    interval_hours:   float = 1.0,
    node_positions:   dict[str, tuple[float, float]] | None = None,
    max_distance:     float | None = 4.0,
) -> tuple[dict, dict]:
    """
    Assign each EV to a charging node using a capacity-aware greedy heuristic,
    scored by the ICS (alpha*price + beta*carbon) over the EV's active window.

    Primary pass: rank all reachable (EV, node) pairs by score and assign
    greedily, accepting a pair only if the node has enough peak-power headroom
    and energy throughput to fully charge the EV. Fallback pass: any EV that
    passes no node is placed at the reachable node with the most headroom.
    Nodes beyond max_distance are excluded; if none is reachable the distance
    cap is relaxed for that EV.
    """
    nodes  = nodal_df.columns.tolist()
    ev_map = {ev.name: ev for ev in evs}

    def _within_distance(ev, node: str) -> bool:
        if max_distance is None or node_positions is None:
            return True
        nx, ny = node_positions.get(node, (0.0, 0.0))
        return math.sqrt((ev.grid_x - nx) ** 2 + (ev.grid_y - ny) ** 2) <= max_distance

    def _reachable_nodes(ev) -> list[str]:
        within = [n for n in nodes if _within_distance(ev, n)]
        if within:
            return within
        print(
            f"  ⚠ EV {ev.name}: no node within max_distance={max_distance} — "
            f"distance constraint relaxed for this EV."
        )
        return nodes

    # Normalise price and carbon by their maxima across all nodes/periods.
    all_prices_flat = nodal_df.values.flatten()
    p_max           = float(all_prices_flat.max())
    all_c_vals      = _all_carbon_values(carbon)
    c_max           = max(all_c_vals)

    # committed[(node, t)] = total max_charging_power of EVs already at that node
    committed: dict[tuple[str, int], float] = {}

    assignment_log: dict = {}
    node_counts:    dict[str, int] = {node: 0 for node in nodes}
    assigned:       set[str]       = set()

    # Score all eligible (EV, node) pairs
    candidates: list[tuple[float, str, str]] = []

    for ev in evs:
        slots = active_slots(ev, time_list)
        if not slots:
            assignment_log[ev.name] = {"scores": {}}
            continue

        reachable    = _reachable_nodes(ev)
        node_scores: dict[str, float] = {}

        for node in reachable:
            # Use this node's own carbon schedule for scoring
            node_carbon = carbon.get(node, {})
            score = 0.0
            for t in slots:
                if t in nodal_df.index:
                    p_norm = nodal_df.loc[t, node] / (p_max + 1e-12)
                    c_norm = node_carbon.get(t, 0.0) / (c_max + 1e-12)
                    score += alpha * p_norm + beta * c_norm
            node_scores[node] = round(score, 4)
            candidates.append((score, ev.name, node))

        assignment_log[ev.name] = {"scores": node_scores}

    candidates.sort(key=lambda x: x[0])

    # Primary greedy assignment
    for score, ev_name, node in candidates:
        if ev_name in assigned:
            continue

        ev    = ev_map[ev_name]
        slots = active_slots(ev, time_list)

        # Check 1 — peak power
        power_ok = all(
            committed.get((node, t), 0.0) + ev.max_charging_power <= grid_cap_kw
            for t in slots
        )
        if not power_ok:
            continue

        # Check 2 — energy throughput
        energy_needed = (ev.desired_energy - ev.arrival_energy) / 0.95
        deliverable   = sum(
            (grid_cap_kw - committed.get((node, t), 0.0)) * interval_hours
            for t in slots
        )
        if deliverable < energy_needed - 1e-6:
            continue

        _do_assign(ev, node, slots, committed, assignment_log, node_counts, assigned)

    # Fallback pass: place remaining EVs at the reachable node with most headroom
    for ev in evs:
        if ev.name in assigned:
            continue

        slots = active_slots(ev, time_list)
        if not slots:
            fallback = nodes[0]
            assignment_log[ev.name]["assigned"] = fallback
            ev.node_id = fallback
            node_counts[fallback] += 1
            continue

        reachable     = _reachable_nodes(ev)
        best_node     = None
        best_headroom = -1.0

        for node in reachable:
            headroom = sum(
                max(0.0, grid_cap_kw - committed.get((node, t), 0.0)) * interval_hours
                for t in slots
            )
            if headroom > best_headroom:
                best_headroom = headroom
                best_node     = node

        _do_assign(ev, best_node, slots, committed, assignment_log, node_counts, assigned)
        print(
            f"  ⚠ EV {ev.name} could not pass capacity checks — "
            f"fallback to {best_node} (headroom={best_headroom:.1f} kWh)"
        )

    nodal_prices: dict = {
        (node, int(t)): float(nodal_df.loc[t, node])
        for node in nodes
        for t    in nodal_df.index
    }

    return nodal_prices, assignment_log


def _do_assign(ev, node, slots, committed, assignment_log, node_counts, assigned):
    """Record an assignment and add the EV's power to the node's committed load."""
    ev.node_id = node
    assignment_log[ev.name]["assigned"] = node
    node_counts[node] += 1
    assigned.add(ev.name)
    for t in slots:
        committed[(node, t)] = committed.get((node, t), 0.0) + ev.max_charging_power


def run_scheduler(
    prices:              dict[int, float],
    carbon:              dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    time_slots,
    evs,
    interval_hours:      float = 1.0,
    alpha:               float = 0.5,
    beta:                float = 0.5,
    carbon_cap_fraction: float | None = 0.85,
    v2g_enabled:         bool  = True,
    eta_c:               float = 0.95,
    eta_d:               float = 0.95,
    deg_cost:            float = 0.02,
    nodal_prices:        dict  | None = None,
    grid_cap_kw:         float = 85.0,
):
    """
    Solve the EV charging / V2G LP and return (c, d, energy_vars, status).

    The objective is the ICS-weighted net charging cost plus a degradation
    penalty. Only the carbon cap is soft (via slack_carbon at cost PENALTY);
    SOC-departure and per-node grid-cap constraints are hard. The cap is
    beta-scaled by _effective_carbon_cap().
    """
    model     = lp.LpProblem("EV_Scheduler", lp.LpMinimize)
    time_list = list(time_slots)

    effective_carbon_cap = _effective_carbon_cap(beta, carbon_cap_fraction)
    if carbon_cap_fraction is not None and effective_carbon_cap != carbon_cap_fraction:
        if effective_carbon_cap is None:
            print(
                f"  ℹ Carbon cap disabled (beta={beta:.2f}; "
                f"configured cap={carbon_cap_fraction} ignored)."
            )
        else:
            print(
                f"  ℹ Carbon cap relaxed by beta: "
                f"configured={carbon_cap_fraction:.3f} → "
                f"effective={effective_carbon_cap:.3f} (beta={beta:.2f})"
            )

    # Primary decision variables
    c = lp.LpVariable.dicts(
        "charge",
        ((ev.name, t) for ev in evs for t in time_list),
        lowBound=0, cat="Continuous",
    )
    d = lp.LpVariable.dicts(
        "discharge",
        ((ev.name, t) for ev in evs for t in time_list),
        lowBound=0, cat="Continuous",
    )

    node_ev_map: dict[str, list] = {}
    for ev in evs:
        node_ev_map.setdefault(ev.node_id, []).append(ev)

    slack_carbon = lp.LpVariable("slack_carbon", lowBound=0)

    # Normalise price by p_max (not min-max). Min-max would give the cheapest
    # period a zero coefficient, so with V2G the LP would over-charge there for
    # "free"; dividing by p_max keeps every period proportional to real $/kWh.
    def eff_price(ev, t):
        if nodal_prices is not None:
            return nodal_prices.get((ev.node_id, t), prices.get(t, 0.0))
        return prices.get(t, 0.0)

    all_eff = [eff_price(ev, t) for ev in evs for t in time_list]
    p_max   = max(all_eff)

    def price_norm(ev, t):
        return eff_price(ev, t) / (p_max + 1e-12)

    # Carbon normalised by c_max, for the same reason (see price_norm above).
    all_c = _all_carbon_values(carbon)
    c_max = max(all_c)

    def carbon_norm(ev, t):
        node_val = carbon.get(ev.node_id, {}).get(t, 0.0)
        return node_val / c_max if c_max > 1e-12 else 0.0

    deg_norm = deg_cost / (p_max + 1e-12)

    # Cost/carbon tiebreaker. When alpha=0 (or beta=0) one term drops out and
    # its total becomes a free variable, so the solver's tie-break would make
    # the reported value depend on the CBC build. A tiny secondary weight makes
    # the LP prefer the cheapest/greenest among equally-optimal solutions
    # without changing the primary optimum for alpha>0 or beta>0.
    TIEBREAK_EPS = 1e-4

    # Objective: ICS-weighted net cost + degradation penalty + cap slack.
    model += (
        lp.lpSum(
            (
                (
                    (alpha + TIEBREAK_EPS) * price_norm(ev, t)
                    + (beta + TIEBREAK_EPS) * carbon_norm(ev, t)
                )
                * (c[(ev.name, t)] - d[(ev.name, t)])
                + deg_norm * d[(ev.name, t)]
            ) * interval_hours
            for ev in evs
            for t  in time_list
        )
        + PENALTY * slack_carbon
    )

    # Per-node grid-cap constraint (hard)
    for node_name, node_evs in node_ev_map.items():
        for t in time_list:
            model += (
                lp.lpSum(c[(ev.name, t)] - d[(ev.name, t)] for ev in node_evs)
                <= grid_cap_kw,
                f"NodeCap_{node_name}_{t}",
            )

    # Global carbon cap (soft, beta-scaled)
    # Baseline = sum over each EV of: its node's carbon[t] × max_charging_power
    if effective_carbon_cap is not None:
        baseline = sum(
            carbon.get(ev.node_id, {}).get(t, 0.0) * ev.max_charging_power * interval_hours
            for ev in evs
            for t  in active_slots(ev, time_list)
        )
        model += (
            lp.lpSum(
                (c[(ev.name, t)] - d[(ev.name, t)])
                * carbon.get(ev.node_id, {}).get(t, 0.0)
                * interval_hours
                for ev in evs
                for t  in time_list
            ) <= effective_carbon_cap * baseline + slack_carbon,
            "CarbonCap",
        )

    # Per-EV battery dynamics
    energy_vars: dict = {}
    for ev in evs:
        slots = active_slots(ev, time_list)

        if not slots:
            for t in time_list:
                model += c[(ev.name, t)] == 0, f"NoCharge_{ev.name}_{t}"
                model += d[(ev.name, t)] == 0, f"NoDischarge_{ev.name}_{t}"
            continue

        energy = lp.LpVariable.dicts(
            f"E_{ev.name}", slots,
            lowBound=0, upBound=ev.battery_capacity,
        )
        energy_vars[ev.name] = energy

        model += energy[slots[0]] == ev.arrival_energy, f"InitSOC_{ev.name}"

        for prev_t, t in zip(slots[:-1], slots[1:]):
            model += (
                energy[t] == energy[prev_t]
                + (c[(ev.name, t)] * eta_c
                   - d[(ev.name, t)] / eta_d) * interval_hours,
                f"SOC_{ev.name}_{t}",
            )

        # Departure SOC — hard constraint
        model += (
            energy[slots[-1]] >= ev.desired_energy,
            f"DepSOC_{ev.name}",
        )

        for t in time_list:
            if is_active(ev, t):
                model += c[(ev.name, t)] <= ev.max_charging_power,    f"MaxC_{ev.name}_{t}"
                if v2g_enabled:
                    model += d[(ev.name, t)] <= ev.max_discharging_power, f"MaxD_{ev.name}_{t}"
                else:
                    model += d[(ev.name, t)] == 0,                        f"NoV2G_{ev.name}_{t}"
            else:
                model += c[(ev.name, t)] == 0, f"AbsC_{ev.name}_{t}"
                model += d[(ev.name, t)] == 0, f"AbsD_{ev.name}_{t}"

    model.solve(lp.PULP_CBC_CMD(msg=0))
    status = lp.LpStatus[model.status]

    _report_carbon_violation(slack_carbon, effective_carbon_cap)

    return c, d, energy_vars, status


def _report_carbon_violation(slack_carbon, carbon_cap_fraction):
    """Warn if the LP had to use carbon-cap slack to stay feasible."""
    carbon_viol = slack_carbon.varValue or 0.0

    if carbon_viol <= 1e-4:
        print("  ✓ Carbon cap satisfied — no slack used.")
        return

    print("\n  ⚠  Soft-constraint violation (LP used slack to stay feasible):")
    print(f"    Carbon cap exceeded by {carbon_viol:.2f} g CO₂")
    print()


def run_baseline(
    evs,
    time_list:      list[int],
    nodal_df:       pd.DataFrame,
    carbon:         dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    charging_nodes,
    prices:         dict[int, float],
    interval_hours: float = 1.0,
    eta_c:          float = 0.95,
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Uncoordinated baseline: assign each EV to its closest node and charge at
    max power until desired SOC is reached. No LP, no V2G. carbon is unused
    here (emissions are computed later by compute_metrics).
    """
    import math as _math

    assignment_log: dict = {}
    for ev in evs:
        best_node, best_dist = None, float("inf")
        for cn in charging_nodes:
            dist = _math.sqrt((ev.grid_x - cn.grid_x) ** 2 +
                              (ev.grid_y - cn.grid_y) ** 2)
            if dist < best_dist:
                best_dist, best_node = dist, cn.name
        ev.node_id = best_node
        assignment_log[ev.name] = {"assigned": best_node}

    c_base:      dict = {}
    d_base:      dict = {}
    energy_base: dict = {}

    for ev in evs:
        soc   = ev.arrival_energy
        e_track: dict = {}

        for t in time_list:
            d_base[(ev.name, t)] = 0.0
            if not is_active(ev, t):
                c_base[(ev.name, t)] = 0.0
                continue

            if soc < ev.desired_energy:
                gap_kwh   = (ev.desired_energy - soc) / eta_c
                charge_kw = min(ev.max_charging_power, gap_kwh / interval_hours)
                soc      += charge_kw * eta_c * interval_hours
            else:
                charge_kw = 0.0

            c_base[(ev.name, t)] = charge_kw
            e_track[t] = min(soc, ev.battery_capacity)

        energy_base[ev.name] = e_track

    nodal_prices_b: dict = {
        (node, int(t)): float(nodal_df.loc[t, node])
        for node in nodal_df.columns
        for t    in nodal_df.index
    }

    return c_base, d_base, energy_base, nodal_prices_b, assignment_log


def compute_metrics(
    prices:        dict[int, float],
    carbon:        dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    time_slots,
    c, d, evs,
    interval_hours: float = 1.0,
    eta_c:          float = 0.95,
    eta_d:          float = 0.95,
    deg_cost:       float = 0.02,
    nodal_prices:   dict  | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Compute (total_cost, total_emissions, total_energy, avg_intensity, total_deg)
    from a schedule, using each EV's assigned-node price and carbon intensity.
    Accepts both LP (LpVariable) and baseline (plain float) result dicts.
    """
    total_cost = total_emissions = total_energy = total_deg = 0.0

    for ev in evs:
        for t in active_slots(ev, list(time_slots)):
            raw_c = c.get((ev.name, t), 0.0)
            raw_d = d.get((ev.name, t), 0.0)
            cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
            dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
            cv = cv or 0.0
            dv = dv or 0.0

            net_grid = (cv - dv) * interval_hours

            if nodal_prices is not None:
                price_t = nodal_prices.get((ev.node_id, t), prices.get(t, 0.0))
            else:
                price_t = prices.get(t, 0.0)

            # Use this EV's node's carbon schedule for emissions
            carbon_t = carbon.get(ev.node_id, {}).get(t, 0.0)

            total_cost      += net_grid * price_t
            total_emissions += net_grid * carbon_t
            total_energy    += net_grid
            total_deg       += dv * interval_hours * deg_cost

    avg_intensity = total_emissions / total_energy if total_energy > 1e-9 else 0.0
    return total_cost, total_emissions, total_energy, avg_intensity, total_deg