"""
scheduler/model.py

Node assignment pre-step + LP-based EV charging / V2G scheduler.

Nodal carbon intensity
──────────────────────
carbon is now dict[str, dict[int, float]]  (node_name → {period → g CO₂/kWh})
instead of the previous flat dict[int, float].

Every function that formerly received a global carbon dict now receives the
nodal version and looks up the correct node's schedule wherever needed:

  assign_ev_nodes : scoring uses carbon[node][t] for each (EV, node) pair
  run_scheduler   : objective, carbon cap, and baseline all use carbon[ev.node_id][t]
  compute_metrics : emissions use carbon[ev.node_id][t]

Normalisation in both assign_ev_nodes and run_scheduler flattens all node
schedules together so α/β weights are comparable across nodes.

Feasibility guarantees (unchanged from previous version)
─────────────────────────────────────────────────────────
assign_ev_nodes
  • Nodes further than max_distance grid units from an EV are excluded from
    both the primary scoring pass and the fallback pass.

  • Fallback selects the node with the most remaining energy headroom.
  • Committed power is updated even for fallback assignments.

run_scheduler
  • One soft slack variable (slack_carbon) softens the carbon-cap constraint.
  • SOC departure and node-cap constraints are hard.
  • PENALTY = 1e6 per unit of slack keeps the solver always returning Optimal.
  • Nonzero slack is reported as a warning after solving.
  • carbon_cap_fraction is beta-scaled via _effective_carbon_cap().
"""

from __future__ import annotations
import math
import pulp as lp
import pandas as pd


# Penalty weight for soft constraints
PENALTY: float = 1e6


# EV helpers
def active_slots(ev, time_list: list) -> list:
    return [t for t in time_list if ev.arrival <= t <= ev.departure]

def is_active(ev, t: int) -> bool:
    return ev.arrival <= t <= ev.departure


# Beta-scaled carbon cap
def _effective_carbon_cap(
    beta: float,
    carbon_cap_fraction: float | None,
) -> float | None:
    """
    Derive the effective carbon-cap fraction from the operator's carbon
    preference weight (beta) and the configured cap.

    Behaviour
    ─────────
    beta = 0          → returns None  (cap fully disabled; no carbon concern)
    0 < beta < 1      → interpolates between 1.0 (uncapped) and the supplied
                        carbon_cap_fraction, so the cap relaxes as beta falls
    beta ≥ 1          → returns carbon_cap_fraction unchanged (full cap)
    cap is None       → returns None  (caller disabled cap unconditionally)

    Interpolation formula
    ─────────────────────
    effective = 1.0 - beta * (1.0 - carbon_cap_fraction)

    Examples with carbon_cap_fraction = 0.85
      beta = 0.0  →  None   (disabled)
      beta = 0.5  →  0.925  (halfway between uncapped and full cap)
      beta = 1.0  →  0.850  (full cap)
      beta = 2.0  →  0.850  (clamped; cap never tightens past supplied value)
    """
    if carbon_cap_fraction is None:
        return None
    if beta <= 0.0:
        return None
    if beta >= 1.0:
        return carbon_cap_fraction

    # Linear interpolation: beta=0 → 1.0 (no cap), beta=1 → carbon_cap_fraction
    return 1.0 - beta * (1.0 - carbon_cap_fraction)


# ── Internal helper: flatten nodal carbon to a list of all values ─────────────

def _all_carbon_values(carbon: dict[str, dict[int, float]]) -> list[float]:
    """Return every g CO₂/kWh value across all nodes and periods."""
    return [v for nc in carbon.values() for v in nc.values()]


# Node assignment
def assign_ev_nodes(
    evs,
    nodal_df:         pd.DataFrame,
    carbon:           dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
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

    carbon is now a nodal dict: {node_name: {period: g_CO2/kWh}}.
    The scoring loop uses carbon[node][t] so each (EV, node) pair is evaluated
    against that specific node's carbon schedule.

    Normalisation spans all nodes and periods together so α/β are comparable.

    DISTANCE FILTER
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

    # Distance helper
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

    # Normalisation bounds — prices
    all_prices_flat = nodal_df.values.flatten()
    p_min, p_max    = float(all_prices_flat.min()), float(all_prices_flat.max())
    p_range         = p_max - p_min + 1e-12

    # Normalisation bounds — carbon (flattened across ALL nodes and periods)
    # Zero-based: c_max only.  See run_scheduler carbon_norm comment for why
    # min-max normalisation causes carbon-only to behave incorrectly.
    all_c_vals = _all_carbon_values(carbon)
    c_max      = max(all_c_vals)

    # committed[(node, t)] = sum of max_charging_power for EVs already assigned
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
                    p_norm = (nodal_df.loc[t, node] - p_min) / p_range
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

    # Fallback pass — headroom-aware, distance-filtered
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

    # Build nodal prices dict
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


# LP scheduler
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
    grid_cap_kw:         float = 75.0,
):
    """
    Solve the EV charging / V2G LP.

    carbon is now dict[str, dict[int, float]]  (node_name → {period → g CO₂/kWh}).
    Each EV's contribution to the objective, the carbon cap baseline, and the
    cap constraint all use carbon[ev.node_id][t] so the LP correctly reflects
    each node's own emission schedule.

    Soft vs hard constraints
    ────────────────────────
    Only the carbon cap is softened with a slack variable (slack_carbon).
    SOC departure and per-node grid-cap constraints are hard.

    slack_carbon costs PENALTY = 1e6 per unit; always returns Optimal.
    Any nonzero slack is reported as a warning after solving.

    Beta-scaled carbon cap
    ──────────────────────
    beta = 0  → cap disabled
    beta < 1  → cap relaxes proportionally
    beta ≥ 1  → cap at carbon_cap_fraction

    Returns (c, d, energy_vars, solver_status_string).
    """
    model     = lp.LpProblem("EV_Scheduler", lp.LpMinimize)
    time_list = list(time_slots)

    # Derive effective carbon cap from beta
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

    # Group EVs by node
    node_ev_map: dict[str, list] = {}
    for ev in evs:
        node_ev_map.setdefault(ev.node_id, []).append(ev)

    # Slack variable (carbon cap only)
    slack_carbon = lp.LpVariable("slack_carbon", lowBound=0)

    # Price normalisation
    def eff_price(ev, t):
        if nodal_prices is not None:
            return nodal_prices.get((ev.node_id, t), prices.get(t, 0.0))
        return prices.get(t, 0.0)

    all_eff  = [eff_price(ev, t) for ev in evs for t in time_list]
    p_min, p_max = min(all_eff), max(all_eff)
    p_range  = p_max - p_min

    def price_norm(ev, t):
        return (eff_price(ev, t) - p_min) / p_range if p_range > 1e-12 else 0.0

    # Carbon normalisation — flatten across all nodes and periods.
    # Zero-based: divide by c_max, NOT (carbon - c_min) / c_range.
    #
    # Why this matters:
    #   (carbon - c_min) / c_range  makes the minimum-carbon period appear
    #   "free" in the LP objective (coefficient = 0).  With V2G enabled, the
    #   LP exploits this by over-charging at that period then discharging
    #   elsewhere, but those "free" kWh carry real emissions (c_min × kWh)
    #   that the objective never sees.  Algebraically:
    #     Objective_old  ≡  (Actual_Emissions − c_min × Net_Energy) / c_range
    #   so carbon-only inadvertently rewards charging MORE, leading to higher
    #   actual emissions than the balanced scheme.
    #
    #   carbon_t / c_max keeps every period proportional to real g CO₂/kWh:
    #     Objective_new  ≡  Actual_Emissions / c_max
    #   β=1 now genuinely minimises actual emissions; β=0 ignores carbon.
    all_c = _all_carbon_values(carbon)
    c_max = max(all_c)

    def carbon_norm(ev, t):
        """
        Node-specific carbon intensity, normalised from zero.
        Proportional to actual g CO₂/kWh so β=1 minimises actual emissions.
        """
        node_val = carbon.get(ev.node_id, {}).get(t, 0.0)
        return node_val / c_max if c_max > 1e-12 else 0.0

    deg_norm = deg_cost / (p_range + 1e-12)

    # Objective — each EV's carbon term uses its own node's schedule
    model += (
        lp.lpSum(
            (
                (alpha * price_norm(ev, t) + beta * carbon_norm(ev, t))
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

    # Solve
    model.solve(lp.PULP_CBC_CMD(msg=0))
    status = lp.LpStatus[model.status]

    # Report soft-constraint violations
    _report_carbon_violation(slack_carbon, effective_carbon_cap)

    return c, d, energy_vars, status


def _report_carbon_violation(slack_carbon, carbon_cap_fraction):
    """Print a summary of any nonzero slack_carbon value after solving."""
    carbon_viol = slack_carbon.varValue or 0.0

    if carbon_viol <= 1e-4:
        print("  ✓ Carbon cap satisfied — no slack used.")
        return

    print("\n  ⚠  Soft-constraint violation (LP used slack to stay feasible):")
    print(f"    Carbon cap exceeded by {carbon_viol:.2f} g CO₂")
    print()


# Baseline scheme

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
    Baseline: assign each EV to its closest node, charge at max power until
    desired SOC is reached.  No LP, no V2G.

    carbon is accepted for API consistency but not used in the baseline
    charging logic; it is used by compute_metrics() called by the caller.
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


# Post-solve metrics

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
    Returns (total_cost, total_emissions, total_energy, avg_intensity, total_deg).
    Works for both LP (LpVariable) and baseline (plain float) dicts.

    Emissions for each EV at each period use that EV's assigned node's own
    carbon intensity: carbon[ev.node_id][t].
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