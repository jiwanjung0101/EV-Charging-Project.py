"""
main.py — EV Charging Scheduler entry point.
"""

from __future__ import annotations
import math

from scheduler.data_loader import (
    load_nodal_prices, load_carbon_intensity, load_evs,
    CHARGING_NODES, get_node_positions,
)
from scheduler.model import (
    assign_ev_nodes, run_scheduler, run_baseline, compute_metrics,
)
from scheduler.plot import plot_all


def main(
    n_nodes:             int   = 5,
    periods:             int   = 24,
    alpha:               float = 1.0,
    beta:                float = 1.0,
    carbon_cap_fraction: float = 0.85,
    v2g_enabled:         bool  = True,
    eta_c:               float = 0.95,
    eta_d:               float = 0.95,
    deg_cost:            float = 0.02,
    grid_cap_kw:         float = 50.0,   # ← updated from 50
    interval_hours:      float = 1.0,
) -> None:

    # ── 1. Load data ──────────────────────────────────────────────────────────
    nodal_df, nodes, time_list = load_nodal_prices(n_nodes=n_nodes, periods=periods)
    carbon,   _                = load_carbon_intensity(periods=periods)
    evs                        = load_evs()

    prices: dict[int, float] = nodal_df.mean(axis=1).to_dict()
    print(f"Loaded {len(evs)} EVs, {len(nodes)} nodes, {len(time_list)} periods.")

    # ── 1a. Grid positions ────────────────────────────────────────────────────
    print("\nCharging node grid positions:")
    node_positions = get_node_positions()
    for cn in CHARGING_NODES:
        if cn.name in nodes:
            print(f"  {cn.name:<40s}  grid=({cn.grid_x}, {cn.grid_y})")

    print("\nEV grid positions:")
    for ev in evs:
        print(f"  EV {ev.name:<4s}  grid=({ev.grid_x}, {ev.grid_y})")

    # ── 2. Optimised node assignment ──────────────────────────────────────────
    nodal_prices, assignment_log = assign_ev_nodes(
        evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
        alpha=alpha, beta=beta,
        grid_cap_kw=grid_cap_kw,
        interval_hours=interval_hours,
    )

    print("\nNode assignments (optimised):")
    for ev_name, info in assignment_log.items():
        ev       = next(e for e in evs if e.name == ev_name)
        node     = info["assigned"]
        node_pos = node_positions.get(node, ("?", "?"))
        print(f"  EV {ev_name:<4s} @ ({ev.grid_x},{ev.grid_y})  →  {node}  @ {node_pos}")

    # ── 2b. Distances ─────────────────────────────────────────────────────────
    def dist(ev, node_name):
        nx, ny = node_positions.get(node_name, (0, 0))
        return math.sqrt((ev.grid_x - nx)**2 + (ev.grid_y - ny)**2)

    print("\n── EV → assigned node distances (optimised) ─────────────────────")
    print(f"  {'EV':<6} {'EV pos':<10} {'Assigned node':<40} {'Node pos':<10} {'Dist':>6}")
    print(f"  {'-'*6} {'-'*10} {'-'*40} {'-'*10} {'-'*6}")
    dists = []
    for ev in evs:
        node_name = assignment_log[ev.name]["assigned"]
        d         = dist(ev, node_name)
        dists.append(d)
        nx, ny    = node_positions.get(node_name, ("?", "?"))
        print(f"  EV {ev.name:<3s}  ({ev.grid_x},{ev.grid_y}){'':>5}  "
              f"{node_name:<40s}  ({nx},{ny}){'':>3}  {d:>6.2f}")
    avg_dist_opt = sum(dists) / len(dists)
    print(f"  {'─'*76}")
    print(f"  Fleet average distance (optimised): {avg_dist_opt:.4f} grid units\n")

    # ── 3. Baseline ───────────────────────────────────────────────────────────
    opt_node_ids = {ev.name: ev.node_id for ev in evs}

    c_base, d_base, energy_base, nodal_prices_b, assignment_log_base = run_baseline(
        evs=evs, time_list=time_list, nodal_df=nodal_df, carbon=carbon,
        charging_nodes=CHARGING_NODES, prices=prices,
        interval_hours=interval_hours, eta_c=eta_c,
    )

    base_cost, base_emissions, base_energy, base_avg_intensity, base_deg = compute_metrics(
        prices=prices, carbon=carbon, time_slots=time_list,
        c=c_base, d=d_base, evs=evs,
        interval_hours=interval_hours, eta_c=eta_c, eta_d=eta_d,
        deg_cost=deg_cost, nodal_prices=nodal_prices_b,
    )

    base_dists = [dist(ev, assignment_log_base[ev.name]["assigned"]) for ev in evs]
    avg_dist_base = sum(base_dists) / len(base_dists)

    # Restore optimised assignments for LP + plots
    for ev in evs:
        ev.node_id = opt_node_ids[ev.name]

    # ── 4. Run LP ─────────────────────────────────────────────────────────────
    c, d, energy_vars, status = run_scheduler(
        prices=prices, carbon=carbon, time_slots=time_list, evs=evs,
        interval_hours=interval_hours, alpha=alpha, beta=beta,
        carbon_cap_fraction=carbon_cap_fraction, v2g_enabled=v2g_enabled,
        eta_c=eta_c, eta_d=eta_d, deg_cost=deg_cost,
        nodal_prices=nodal_prices, grid_cap_kw=grid_cap_kw,
    )

    print(f"Solver status: {status}")

    # ── 5. Metrics + comparison ───────────────────────────────────────────────
    total_cost, total_emissions, total_energy, avg_intensity, total_deg = compute_metrics(
        prices=prices, carbon=carbon, time_slots=time_list,
        c=c, d=d, evs=evs,
        interval_hours=interval_hours, eta_c=eta_c, eta_d=eta_d,
        deg_cost=deg_cost, nodal_prices=nodal_prices,
    )

    def pct(opt, base):
        return f"{(opt-base)/abs(base)*100:+.1f}%" if abs(base) > 1e-9 else "  n/a"

    print("\n══════════════════════════════════════════════════════════════")
    print("  Metric comparison: Optimised  vs  Baseline")
    print("══════════════════════════════════════════════════════════════")
    print(f"  {'Metric':<28} {'Optimised':>12} {'Baseline':>12} {'Δ':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*8}")
    print(f"  {'Total cost ($)':<28} {total_cost:>12.2f} {base_cost:>12.2f} {pct(total_cost,base_cost):>8}")
    print(f"  {'Total emissions (g CO₂)':<28} {total_emissions:>12.2f} {base_emissions:>12.2f} {pct(total_emissions,base_emissions):>8}")
    print(f"  {'Total net energy (kWh)':<28} {total_energy:>12.2f} {base_energy:>12.2f} {pct(total_energy,base_energy):>8}")
    print(f"  {'Avg carbon intensity':<28} {avg_intensity:>12.2f} {base_avg_intensity:>12.2f} {pct(avg_intensity,base_avg_intensity):>8}")
    print(f"  {'Total degradation ($)':<28} {total_deg:>12.2f} {base_deg:>12.2f} {pct(total_deg,base_deg):>8}")
    print(f"  {'Avg EV–node distance':<28} {avg_dist_opt:>12.4f} {avg_dist_base:>12.4f} {pct(avg_dist_opt,avg_dist_base):>8}")
    print("══════════════════════════════════════════════════════════════\n")

    # ── 7. Baseline assignments ───────────────────────────────────────────────
    print("Baseline node assignments (closest node):")
    for ev in evs:
        bn    = assignment_log_base[ev.name]["assigned"]
        nx,ny = node_positions.get(bn, ("?","?"))
        d_    = dist(ev, bn)
        print(f"  EV {ev.name:<4s} @ ({ev.grid_x},{ev.grid_y})"
              f"  →  {bn:<40s}  ({nx},{ny})  dist={d_:.2f}")
    print(f"  Fleet average distance (baseline): {avg_dist_base:.4f} grid units\n")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_all(
        prices=prices, carbon=carbon, time_list=time_list, evs=evs,
        c=c, d=d, energy_vars=energy_vars,
        charging_nodes=CHARGING_NODES, assignment_log=assignment_log,
        nodal_df=nodal_df, nodal_prices=nodal_prices,
        interval_hours=interval_hours, grid_cap_kw=grid_cap_kw,
    )


if __name__ == "__main__":
    main()