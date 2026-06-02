"""
main.py — EV Charging Scheduler entry point.

What this does
──────────────
1. Load data (CAISO nodal prices, carbon intensity, EV fleet).

2. Print grid positions for nodes and EVs.

3. Run the balanced scheme (α = β = 0.5) as the primary schedule and print
   the detailed per-EV report that was previously in the old main.py:
     • node assignments + distances
     • full metric comparison against the unmanaged baseline

4. Run all four evaluation schemes (Unmanaged, Cost-only, Carbon-only,
   Balanced) and the Pareto sweep — previously in main_paper.py.

5. Print the four-scheme summary table.

6. Generate all figures and the performance table, saved to `out_dir`.

Bug fix vs old code
───────────────────
The old main.py ran the LP with alpha=1.0, beta=1.0 (the function defaults)
while claiming to show the "balanced" scheme, so results did not match
main_paper.py's balanced run (α=β=0.5).  Both now use α=β=0.5 consistently.

Parameters
──────────
Edit the CONFIG block below.  All parameters flow to every scheme run.
"""

from __future__ import annotations
import math

from scheduler.data_loader import (
    load_nodal_prices,
    load_carbon_intensity,
    load_evs,
    CHARGING_NODES,
    get_node_positions,
)
from scheduler.model import (
    assign_ev_nodes,
    run_scheduler,
    run_baseline,
    compute_metrics,
)
from scheduler.results import run_all_schemes
from scheduler.plot import plot_all


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit here to change global parameters
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = dict(
    n_nodes             = 4,      # number of CAISO nodes used
    periods             = 24,     # scheduling horizon (hours)
    carbon_cap_fraction = 0.85,   # soft carbon cap as fraction of unmanaged baseline
    v2g_enabled         = True,
    eta_c               = 0.95,
    eta_d               = 0.95,
    deg_cost            = 0.02,   # $/kWh discharged (battery degradation)
    grid_cap_kw         = 85.0,   # per-node power cap (kW)
    interval_hours      = 1.0,
    max_distance        = 4.0,    # max EV-to-node Euclidean distance (grid units)
    out_dir             = "paper_figures",
)

# ═══════════════════════════════════════════════════════════════════════════


def _dist(ev, node_name: str, node_positions: dict) -> float:
    nx, ny = node_positions.get(node_name, (0, 0))
    return math.sqrt((ev.grid_x - nx) ** 2 + (ev.grid_y - ny) ** 2)


def main() -> None:

    # ── 1. Load data ──────────────────────────────────────────────────────────
    nodal_df, nodes, time_list = load_nodal_prices(
        n_nodes=CONFIG["n_nodes"],
        periods=CONFIG["periods"],
    )
    carbon, _ = load_carbon_intensity(periods=CONFIG["periods"])
    evs       = load_evs()

    prices: dict[int, float] = nodal_df.mean(axis=1).to_dict()
    node_positions            = get_node_positions()

    print(f"Loaded {len(evs)} EVs, {len(nodes)} nodes, {len(time_list)} periods.")
    print(f"Distance cap: {CONFIG['max_distance']} grid units")

    # ── 2. Grid positions ─────────────────────────────────────────────────────
    print("\nCharging node grid positions:")
    for cn in CHARGING_NODES:
        if cn.name in nodes:
            print(f"  {cn.name:<40s}  grid=({cn.grid_x}, {cn.grid_y})")

    print("\nEV grid positions:")
    for ev in evs:
        print(f"  EV {ev.name:<4s}  grid=({ev.grid_x}, {ev.grid_y})")

    # ── 3. Balanced scheme: node assignment (α = β = 0.5) ────────────────────
    alpha_bal = 0.5
    beta_bal  = 0.5

    nodal_prices, assignment_log = assign_ev_nodes(
        evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
        alpha=alpha_bal, beta=beta_bal,
        grid_cap_kw=CONFIG["grid_cap_kw"],
        interval_hours=CONFIG["interval_hours"],
        node_positions=node_positions,
        max_distance=CONFIG["max_distance"],
    )

    print("\nNode assignments (balanced, α=β=0.5):")
    for ev_name, info in assignment_log.items():
        ev       = next(e for e in evs if e.name == ev_name)
        node     = info["assigned"]
        node_pos = node_positions.get(node, ("?", "?"))
        print(f"  EV {ev_name:<4s} @ ({ev.grid_x},{ev.grid_y})"
              f"  →  {node}  @ {node_pos}")

    # ── Distances (balanced) ──────────────────────────────────────────────────
    print("\n── EV → assigned node distances (balanced) ──────────────────────────")
    print(f"  {'EV':<6} {'EV pos':<10} {'Assigned node':<40} {'Node pos':<10} {'Dist':>6}")
    print(f"  {'-'*6} {'-'*10} {'-'*40} {'-'*10} {'-'*6}")
    dists_opt = []
    for ev in evs:
        node_name = assignment_log[ev.name]["assigned"]
        d         = _dist(ev, node_name, node_positions)
        dists_opt.append(d)
        nx, ny    = node_positions.get(node_name, ("?", "?"))
        over_cap  = (
            f"  ⚠ >{CONFIG['max_distance']}"
            if d > CONFIG["max_distance"]
            else ""
        )
        print(f"  EV {ev.name:<3s}  ({ev.grid_x},{ev.grid_y}){'':>5}  "
              f"{node_name:<40s}  ({nx},{ny}){'':>3}  {d:>6.2f}{over_cap}")
    avg_dist_opt = sum(dists_opt) / len(dists_opt)
    print(f"  {'─'*76}")
    print(f"  Fleet average distance (balanced): {avg_dist_opt:.4f} grid units\n")

    # ── 4. Unmanaged baseline ─────────────────────────────────────────────────
    # Capture balanced assignments before baseline overwrites node_id
    opt_node_ids = {ev.name: ev.node_id for ev in evs}

    c_base, d_base, energy_base, nodal_prices_b, assignment_log_base = run_baseline(
        evs=evs, time_list=time_list, nodal_df=nodal_df, carbon=carbon,
        charging_nodes=CHARGING_NODES, prices=prices,
        interval_hours=CONFIG["interval_hours"], eta_c=CONFIG["eta_c"],
    )
    base_cost, base_emissions, base_energy, base_avg_intensity, base_deg = compute_metrics(
        prices=prices, carbon=carbon, time_slots=time_list,
        c=c_base, d=d_base, evs=evs,
        interval_hours=CONFIG["interval_hours"],
        eta_c=CONFIG["eta_c"], eta_d=CONFIG["eta_d"],
        deg_cost=CONFIG["deg_cost"], nodal_prices=nodal_prices_b,
    )
    dists_base   = [_dist(ev, assignment_log_base[ev.name]["assigned"], node_positions) for ev in evs]
    avg_dist_base = sum(dists_base) / len(dists_base)

    # Restore balanced assignments for LP
    for ev in evs:
        ev.node_id = opt_node_ids[ev.name]

    # ── 5. Balanced LP ────────────────────────────────────────────────────────
    c, d, energy_vars, status = run_scheduler(
        prices=prices, carbon=carbon, time_slots=time_list, evs=evs,
        interval_hours=CONFIG["interval_hours"],
        alpha=alpha_bal, beta=beta_bal,
        carbon_cap_fraction=CONFIG["carbon_cap_fraction"],
        v2g_enabled=CONFIG["v2g_enabled"],
        eta_c=CONFIG["eta_c"], eta_d=CONFIG["eta_d"],
        deg_cost=CONFIG["deg_cost"],
        nodal_prices=nodal_prices,
        grid_cap_kw=CONFIG["grid_cap_kw"],
    )
    print(f"Solver status: {status}")

    total_cost, total_emissions, total_energy, avg_intensity, total_deg = compute_metrics(
        prices=prices, carbon=carbon, time_slots=time_list,
        c=c, d=d, evs=evs,
        interval_hours=CONFIG["interval_hours"],
        eta_c=CONFIG["eta_c"], eta_d=CONFIG["eta_d"],
        deg_cost=CONFIG["deg_cost"], nodal_prices=nodal_prices,
    )

    # ── 6. Balanced vs baseline comparison ───────────────────────────────────
    def pct(opt, base):
        return f"{(opt - base) / abs(base) * 100:+.1f}%" if abs(base) > 1e-9 else "  n/a"

    print("\n══════════════════════════════════════════════════════════════")
    print("  Balanced (α=β=0.5)  vs  Unmanaged baseline")
    print("══════════════════════════════════════════════════════════════")
    print(f"  {'Metric':<28} {'Balanced':>12} {'Baseline':>12} {'Δ':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*8}")
    print(f"  {'Total cost ($)':<28} {total_cost:>12.2f} {base_cost:>12.2f} {pct(total_cost,base_cost):>8}")
    print(f"  {'Total emissions (g CO₂)':<28} {total_emissions:>12.2f} {base_emissions:>12.2f} {pct(total_emissions,base_emissions):>8}")
    print(f"  {'Total net energy (kWh)':<28} {total_energy:>12.2f} {base_energy:>12.2f} {pct(total_energy,base_energy):>8}")
    print(f"  {'Avg carbon intensity':<28} {avg_intensity:>12.2f} {base_avg_intensity:>12.2f} {pct(avg_intensity,base_avg_intensity):>8}")
    print(f"  {'Total degradation ($)':<28} {total_deg:>12.2f} {base_deg:>12.2f} {pct(total_deg,base_deg):>8}")
    print(f"  {'Avg EV–node distance':<28} {avg_dist_opt:>12.4f} {avg_dist_base:>12.4f} {pct(avg_dist_opt,avg_dist_base):>8}")
    print("══════════════════════════════════════════════════════════════\n")

    print("Baseline node assignments (closest node):")
    for ev in evs:
        bn    = assignment_log_base[ev.name]["assigned"]
        nx,ny = node_positions.get(bn, ("?","?"))
        d_    = _dist(ev, bn, node_positions)
        print(f"  EV {ev.name:<4s} @ ({ev.grid_x},{ev.grid_y})"
              f"  →  {bn:<40s}  ({nx},{ny})  dist={d_:.2f}")
    print(f"  Fleet average distance (baseline): {avg_dist_base:.4f} grid units\n")

    # ── 7. All four schemes + Pareto sweep ────────────────────────────────────
    print("=" * 62)
    print("  Running all four schemes + Pareto sweep …")
    print("=" * 62)

    # Reload evs so run_all_schemes gets a clean copy untouched by the
    # detailed run above (run_all_schemes copies internally, but starting
    # from clean data is safer)
    evs_fresh = load_evs()

    results, pareto_points = run_all_schemes(
        evs_orig            = evs_fresh,
        nodal_df            = nodal_df,
        carbon              = carbon,
        time_list           = time_list,
        prices              = prices,
        carbon_cap_fraction = CONFIG["carbon_cap_fraction"],
        v2g_enabled         = CONFIG["v2g_enabled"],
        eta_c               = CONFIG["eta_c"],
        eta_d               = CONFIG["eta_d"],
        deg_cost            = CONFIG["deg_cost"],
        grid_cap_kw         = CONFIG["grid_cap_kw"],
        interval_hours      = CONFIG["interval_hours"],
        max_distance        = CONFIG["max_distance"],
    )

    # ── 8. Four-scheme summary ────────────────────────────────────────────────
    unmanaged = results[0]
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  Summary — all four schemes vs unmanaged baseline")
    print("══════════════════════════════════════════════════════════════════════")
    print(f"  {'Scheme':<34} {'Cost($)':>9} {'':>7} {'Emiss(g)':>11} {'':>7} {'Deg($)':>8}")
    print(f"  {'─'*34} {'─'*9} {'─'*7} {'─'*11} {'─'*7} {'─'*8}")

    def pct2(v, b):
        return f"({(v - b) / abs(b) * 100:+.0f}%)" if abs(b) > 1e-9 else ""

    for sr in results:
        print(
            f"  {sr.label:<34} "
            f"{sr.total_cost:>9.2f} {pct2(sr.total_cost, unmanaged.total_cost):>7} "
            f"{sr.total_emissions:>11.2f} {pct2(sr.total_emissions, unmanaged.total_emissions):>7} "
            f"{sr.total_deg:>8.2f} {pct2(sr.total_deg, unmanaged.total_deg):>7}"
        )
    print()

    # ── 9. Figures + performance table ───────────────────────────────────────
    plot_all(
        results        = results,
        pareto_points  = pareto_points,
        evs            = evs_fresh,
        nodal_df       = nodal_df,
        carbon         = carbon,
        time_list      = time_list,
        charging_nodes = CHARGING_NODES,
        grid_cap_kw    = CONFIG["grid_cap_kw"],
        out_dir        = CONFIG["out_dir"],
    )


if __name__ == "__main__":
    main()