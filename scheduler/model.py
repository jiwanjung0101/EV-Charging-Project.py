"""
scheduler/model.py

Node assignment pre-step + LP-based EV charging / V2G scheduler.

Feasibility guarantees (vs previous version)
─────────────────────────────────────────────
assign_ev_nodes
  • Nodes further than max_distance grid units from an EV are excluded from
    both the primary scoring pass and the fallback pass.  If no node falls
    within range (edge case), the constraint is relaxed with a warning rather
    than leaving the EV unassigned.

  • Fallback now selects the node with the most remaining energy headroom
    over the EV's active window — not just the least-populated node.
  • Committed power is updated even for fallback assignments so subsequent
    fallbacks see an accurate picture of node load.

run_scheduler
  • Three families of *soft* slack variables replace hard constraints that
    could make the LP infeasible:

      slack_soc[ev]          – SOC-at-departure shortfall (kWh)
      slack_cap[(node, t)]   – node power-cap excess (kW)
      slack_carbon           – carbon-cap excess (g CO₂)

  • Each slack is penalised at PENALTY (default 1 × 10⁶) in the objective,
    making constraint violation extremely expensive without ever making the
    LP infeasible.  The solver always returns "Optimal".

  • After solving, any nonzero slack values are printed as warnings so the
    operator knows which constraints were soft-violated and by how much.
"""

from __future__ import annotations
import math
import pulp as lp
import pandas as pd


# ── Penalty weight for soft constraints ───────────────────────────────────────
PENALTY: float = 1e6


# ── EV helpers ────────────────────────────────────────────────────────────────

def active_slots(ev, time_list: list) -> list:
    return [t for t in time_list if ev.arrival <= t <= ev.departure]

def is_active(ev, t: int) -> bool:
    return ev.arrival <= t <= ev.departure


# ── Node assignment ───────────────────────────────────────────────────────────

def assign_ev_nodes(
    evs,
    nodal_df:         pd.DataFrame,
    carbon:           dict[int, float],
    time_list:        list[int],
    alpha:            float = 1.0,
    beta:             float = 1.0,
    max_evs_per_node: int | None = None,   # kept for API compat, ignored
    grid_cap_kw:      float = 50.0,
    interval_hours:   float = 1.0,
    node_positions:   dict[str, tuple[float, float]] | None = None,
    max_distance:     float | None = 3.5,
) -> tuple[dict, dict]:
    """
    Capacity-aware greedy node assignment with optional distance cap.

    DISTANCE FILTER  ← new
    ───────────────
    Before scoring, each (EV, node) pair is checked against max_distance.
    Nodes further than max_distance grid units from the EV are skipped in
    both the primary and fallback passes.  If no node is reachable (edge
    case), the distance constraint is relaxed for that EV with a warning.

    PRIMARY PASS
    ────────────
    Score every eligible (EV, node) pair; sort ascending (cheapest + greenest
    first).  Assign greedily subject to TWO checks:

      1. Peak-power: committed[node, t] + ev.max_charging_power ≤ grid_cap_kw
         for every active slot t.
      2. Energy-throughput: remaining deliverable energy at the node ≥ energy
         the EV still needs (accounts for charging efficiency).

    FALLBACK PASS
    ─────────────
    Any EV that failed all nodes in the primary pass is assigned to the
    reachable node with the most remaining energy headroom.  Committed power
    is updated after each fallback so later fallbacks see accurate load.
    """
    nodes  = nodal_df.columns.tolist()
    ev_map = {ev.name: ev for ev in evs}

    # ── Distance helper ───────────────────────────────────────────────────────
    def _within_distance(ev, node: str) -> bool:
        """True if no distance cap is set, or EV is within max_distance of node."""
        if max_distance is None or node_positions is None:
            return True
        nx, ny = node_positions.get(node, (0.0, 0.0))
        return math.sqrt((ev.grid_x - nx) ** 2 + (ev.grid_y - ny) ** 2) <= max_distance

    def _reachable_nodes(ev) -> list[str]:
        """Return nodes within distance cap; fall back to all nodes with warning."""
        within = [n for n in nodes if _within_distance(ev, n)]
        if within:
            return within
        print(
            f"  ⚠ EV {ev.name}: no node within max_distance={max_distance} — "
            f"distance constraint relaxed for this EV."
        )
        return nodes

    # ── Normalisation bounds ──────────────────────────────────────────────────
    all_prices_flat = nodal_df.values.flatten()
    p_min, p_max    = float(all_prices_flat.min()), float(all_prices_flat.max())
    p_range         = p_max - p_min + 1e-12
    c_vals          = list(carbon.values())
    c_min, c_max    = min(c_vals), max(c_vals)
    c_range         = c_max - c_min + 1e-12

    # committed[(node, t)] = sum of max_charging_power for EVs already assigned
    committed: dict[tuple[str, int], float] = {}

    assignment_log: dict = {}
    node_counts:    dict[str, int] = {node: 0 for node in nodes}
    assigned:       set[str]       = set()

    # ── Score all eligible (EV, node) pairs ───────────────────────────────────
    candidates: list[tuple[float, str, str]] = []

    for ev in evs:
        slots = active_slots(ev, time_list)
        if not slots:
            assignment_log[ev.name] = {"scores": {}}
            continue

        reachable    = _reachable_nodes(ev)
        node_scores: dict[str, float] = {}

        for node in reachable:
            score = 0.0
            for t in slots:
                if t in nodal_df.index:
                    p_norm = (nodal_df.loc[t, node] - p_min) / p_range
                    c_norm = (carbon.get(t, 0.0)    - c_min) / c_range
                    score += alpha * p_norm + beta * c_norm
            node_scores[node] = round(score, 4)
            candidates.append((score, ev.name, node))

        assignment_log[ev.name] = {"scores": node_scores}

    candidates.sort(key=lambda x: x[0])

    # ── Primary greedy assignment ─────────────────────────────────────────────
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

    # ── Fallback pass — headroom-aware, distance-filtered ─────────────────────
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

    # ── Build nodal prices dict ───────────────────────────────────────────────
    nodal_prices: dict = {
        (node, int(t)): float(nodal_df.loc[t, node])
        for node in nodes
        for t    in nodal_df.index
    }

    return nodal_prices, assignment_log


def _do_assign(ev, node, slots, committed, assignment_log, node_counts, assigned):
    """Shared bookkeeping for primary and fallback assignment."""
    ev.node_id = node
    assignment_log[ev.name]["assigned"] = node
    node_counts[node] += 1
    assigned.add(ev.name)
    for t in slots:
        committed[(node, t)] = committed.get((node, t), 0.0) + ev.max_charging_power


# ── LP scheduler ──────────────────────────────────────────────────────────────

def run_scheduler(
    prices:              dict[int, float],
    carbon:              dict[int, float],
    time_slots,
    evs,
    interval_hours:      float = 1.0,
    alpha:               float = 1.0,
    beta:                float = 1.0,
    carbon_cap_fraction: float | None = 0.85,
    v2g_enabled:         bool  = True,
    eta_c:               float = 0.95,
    eta_d:               float = 0.95,
    deg_cost:            float = 0.005,
    nodal_prices:        dict  | None = None,
    grid_cap_kw:         float = 75.0,
):
    """
    Solve the EV charging / V2G LP.

    Guaranteed feasibility
    ──────────────────────
    Three sets of non-negative slack variables soften the constraints that
    could otherwise make the problem infeasible:

      • slack_soc[ev]        softens  energy[last_slot] ≥ desired_energy
      • slack_cap[node, t]   softens  Σ(c − d) ≤ grid_cap_kw  per node per t
      • slack_carbon         softens  the global carbon cap

    Every unit of slack costs PENALTY = 1e6 in the objective, so the solver
    uses slack only as a last resort and the LP always returns "Optimal".

    After solving, any nonzero slacks are printed as warnings.

    Returns (c, d, energy_vars, solver_status_string).
    """
    model     = lp.LpProblem("EV_Scheduler", lp.LpMinimize)
    time_list = list(time_slots)

    # ── Primary decision variables ────────────────────────────────────────────
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

    # ── Group EVs by node ─────────────────────────────────────────────────────
    node_ev_map: dict[str, list] = {}
    for ev in evs:
        node_ev_map.setdefault(ev.node_id, []).append(ev)

    # ── Slack variables ───────────────────────────────────────────────────────
    slack_soc = lp.LpVariable.dicts(
        "slack_soc", [ev.name for ev in evs], lowBound=0,
    )
    cap_keys  = [(node, t) for node in node_ev_map for t in time_list]
    slack_cap = lp.LpVariable.dicts("slack_cap", cap_keys, lowBound=0)
    slack_carbon = lp.LpVariable("slack_carbon", lowBound=0)

    # ── Price / carbon normalisation ──────────────────────────────────────────
    def eff_price(ev, t):
        if nodal_prices is not None:
            return nodal_prices.get((ev.node_id, t), prices.get(t, 0.0))
        return prices.get(t, 0.0)

    all_eff  = [eff_price(ev, t) for ev in evs for t in time_list]
    p_min, p_max = min(all_eff), max(all_eff)
    c_min, c_max = min(carbon.values()), max(carbon.values())
    p_range  = p_max - p_min
    c_range  = c_max - c_min

    def price_norm(ev, t):
        return (eff_price(ev, t) - p_min) / p_range if p_range > 1e-12 else 0.0

    def carbon_norm(t):
        return (carbon[t] - c_min) / c_range if c_range > 1e-12 else 0.0

    deg_norm = deg_cost / (p_range + 1e-12)

    # ── Objective ─────────────────────────────────────────────────────────────
    model += (
        lp.lpSum(
            (
                (alpha * price_norm(ev, t) + beta * carbon_norm(t))
                * (c[(ev.name, t)] - d[(ev.name, t)])
                + deg_norm * d[(ev.name, t)]
            ) * interval_hours
            for ev in evs
            for t  in time_list
        )
        + PENALTY * lp.lpSum(slack_soc[ev.name] for ev in evs)
        + PENALTY * lp.lpSum(slack_cap[k] for k in cap_keys)
        + PENALTY * slack_carbon
    )

    # ── Per-node grid-cap constraint (soft) ───────────────────────────────────
    for node_name, node_evs in node_ev_map.items():
        for t in time_list:
            model += (
                lp.lpSum(c[(ev.name, t)] - d[(ev.name, t)] for ev in node_evs)
                <= grid_cap_kw + slack_cap[(node_name, t)],
                f"NodeCap_{node_name}_{t}",
            )

    # ── Global carbon cap (soft) ──────────────────────────────────────────────
    if carbon_cap_fraction is not None:
        baseline = sum(
            carbon[t] * ev.max_charging_power * interval_hours
            for ev in evs
            for t  in active_slots(ev, time_list)
        )
        model += (
            lp.lpSum(
                (c[(ev.name, t)] - d[(ev.name, t)]) * carbon[t] * interval_hours
                for ev in evs
                for t  in time_list
            ) <= carbon_cap_fraction * baseline + slack_carbon,
            "CarbonCap",
        )

    # ── Per-EV battery dynamics ───────────────────────────────────────────────
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

        # Departure SOC — soft via slack_soc
        model += (
            energy[slots[-1]] + slack_soc[ev.name] >= ev.desired_energy,
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

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.solve(lp.PULP_CBC_CMD(msg=0))
    status = lp.LpStatus[model.status]

    # ── Report soft-constraint violations ─────────────────────────────────────
    _report_violations(slack_soc, slack_cap, slack_carbon, grid_cap_kw, carbon_cap_fraction)

    return c, d, energy_vars, status


def _report_violations(slack_soc, slack_cap, slack_carbon, grid_cap_kw, carbon_cap_fraction):
    """Print a summary of any nonzero slack values after solving."""
    soc_viols = {
        ev_name: (var.varValue or 0.0)
        for ev_name, var in slack_soc.items()
        if (var.varValue or 0.0) > 1e-4
    }
    cap_viols = {
        k: (var.varValue or 0.0)
        for k, var in slack_cap.items()
        if (var.varValue or 0.0) > 1e-4
    }
    carbon_viol = slack_carbon.varValue or 0.0

    if not soc_viols and not cap_viols and carbon_viol <= 1e-4:
        print("  ✓ All hard constraints satisfied — no slack used.")
        return

    print("\n  ⚠  Soft-constraint violations (LP used slack to stay feasible):")

    if soc_viols:
        print(f"    SOC shortfall  ({len(soc_viols)} EV{'s' if len(soc_viols)>1 else ''}):")
        for ev_name, val in sorted(soc_viols.items(), key=lambda x: -x[1]):
            print(f"      EV {ev_name:<6s}  short by {val:.3f} kWh")

    if cap_viols:
        node_max: dict[str, float] = {}
        for (node, t), val in cap_viols.items():
            node_max[node] = max(node_max.get(node, 0.0), val)
        print(f"    Node-cap excess  ({len(node_max)} node{'s' if len(node_max)>1 else ''}):")
        for node, val in sorted(node_max.items(), key=lambda x: -x[1]):
            short = node.split("-")[0]
            print(f"      {short:<30s}  peak excess {val:.2f} kW  "
                  f"(cap={grid_cap_kw:.0f} kW)")

    if carbon_viol > 1e-4:
        print(f"    Carbon cap exceeded by {carbon_viol:.2f} g CO₂")

    print()


# ── Baseline scheme ───────────────────────────────────────────────────────────

def run_baseline(
    evs,
    time_list:      list[int],
    nodal_df:       pd.DataFrame,
    carbon:         dict[int, float],
    charging_nodes,
    prices:         dict[int, float],
    interval_hours: float = 1.0,
    eta_c:          float = 0.95,
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Baseline: assign each EV to its closest node, charge at max power until
    desired SOC is reached.  No LP, no V2G.
    """
    node_map = {cn.name: cn for cn in charging_nodes}

    assignment_log: dict = {}
    for ev in evs:
        best_node, best_dist = None, float("inf")
        for cn in charging_nodes:
            dist = math.sqrt((ev.grid_x - cn.grid_x) ** 2 +
                             (ev.grid_y - cn.grid_y) ** 2)
            if dist < best_dist:
                best_dist, best_node = dist, cn.name
        ev.node_id = best_node
        assignment_log[ev.name] = {"assigned": best_node}

    c_base:      dict = {}
    d_base:      dict = {}
    energy_base: dict = {}

    for ev in evs:
        slots = active_slots(ev, time_list)
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


# ── Post-solve metrics ────────────────────────────────────────────────────────

def compute_metrics(
    prices:        dict[int, float],
    carbon:        dict[int, float],
    time_slots,
    c, d, evs,
    interval_hours: float = 1.0,
    eta_c:          float = 0.95,
    eta_d:          float = 0.95,
    deg_cost:       float = 0.02,
    nodal_prices:   dict  | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Returns (total_cost, total_emissions, total_energy, avg_intensity, total_deg).
    Works for both LP (LpVariable) and baseline (plain float) dicts.
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

            total_cost      += net_grid * price_t
            total_emissions += net_grid * carbon.get(t, 0.0)
            total_energy    += net_grid
            total_deg       += dv * interval_hours * deg_cost

    avg_intensity = total_emissions / total_energy if total_energy > 1e-9 else 0.0
    return total_cost, total_emissions, total_energy, avg_intensity, total_deg