"""
Run the four evaluation schemes (Uncoordinated, Cost-only, Carbon-only,
Balanced) and the Pareto sweep, returning structured results for the figures
and the performance table. The sweep varies alpha from 0 to 1 in PARETO_STEPS
steps (beta = 1 - alpha), re-running both stages at each point: EVs are
re-scored and reallocated under the new weighting, then the LP is re-solved.
Each point therefore reflects the full two-stage response to that weighting.
Set freeze_pareto_assignments=True to hold Stage 1 fixed at the balanced
allocation instead, which isolates the dispatch effect.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import pandas as pd

from scheduler.model import (
    assign_ev_nodes,
    run_scheduler,
    run_baseline,
    compute_metrics,
    active_slots,
)
from scheduler.data_loader import CHARGING_NODES, get_node_positions


SCHEME_LABELS: list[str] = [
    "Uncoordinated",
    "Cost-only (α=1, β=0)",
    "Carbon-only (α=0, β=1)",
    "Balanced (α=0.5, β=0.5)",
]

PARETO_STEPS: int = 21   # α ∈ {0.00, 0.05, …, 1.00}

@dataclass
class SchemeResult:
    label:             str
    alpha:             float
    beta:              float
    total_cost:        float
    total_emissions:   float
    total_energy:      float
    avg_intensity:     float
    total_deg:         float
    avg_dist:          float
    # raw schedule dicts — needed for plotting
    c:                 dict = field(repr=False)
    d:                 dict = field(repr=False)
    energy_vars:       dict = field(repr=False)
    nodal_prices:      dict = field(repr=False)
    assignment_log:    dict = field(repr=False)
    diag:              dict = field(repr=False, default_factory=dict)


@dataclass
class ParetoPoint:
    alpha:           float
    beta:            float
    total_cost:      float
    total_emissions: float
    total_deg:       float        = 0.0
    cap_g:           float | None = None
    slack_g:         float        = 0.0
    assignments:     dict         = field(repr=False, default_factory=dict)


# Helpers

def _dist(ev, node_name: str, node_positions: dict) -> float:
    nx, ny = node_positions.get(node_name, (0, 0))
    return math.sqrt((ev.grid_x - nx) ** 2 + (ev.grid_y - ny) ** 2)


def _avg_dist(evs, assignment_log: dict, node_positions: dict) -> float:
    dists = [_dist(ev, assignment_log[ev.name]["assigned"], node_positions)
             for ev in evs]
    return sum(dists) / len(dists) if dists else 0.0


def _copy_evs(evs):
    """Return shallow copies so node_id mutations don't bleed between runs."""
    import copy
    return [copy.copy(ev) for ev in evs]


def _apply_frozen_nodes(evs, nodal_df: pd.DataFrame, frozen_nodes: dict):
    """
    Apply a previously computed {ev_name: node} allocation without re-running
    the ICS scoring. Returns (nodal_prices, assignment_log) in the same shape
    assign_ev_nodes() produces. nodal_prices depends only on nodal_df, not on
    the allocation, so it is rebuilt identically.
    """
    assignment_log: dict = {}
    for ev in evs:
        node = frozen_nodes[ev.name]
        ev.node_id = node
        assignment_log[ev.name] = {"assigned": node, "scores": {}}

    nodal_prices = {
        (node, int(t)): float(nodal_df.loc[t, node])
        for node in nodal_df.columns
        for t    in nodal_df.index
    }
    return nodal_prices, assignment_log


def audit_sweep(pareto: list, frozen_nodes: dict | None) -> None:
    """
    Sanity-check the Pareto sweep: cost must be non-increasing in alpha and
    emissions non-decreasing. Also reports whether the carbon cap binds and
    whether the Stage-1 allocation actually stayed frozen.
    """
    rows = sorted(pareto, key=lambda r: r.alpha)

    print("\n" + "=" * 78)
    print("  Pareto sweep audit")
    print("=" * 78)
    print(f"  {'alpha':>6} {'cost($)':>9} {'deg($)':>8} {'cost+deg':>9} "
          f"{'emis(g)':>11} {'cap(g)':>11} {'slack':>7} {'binding':>8} {'frozen':>7}")
    print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*9} {'-'*11} {'-'*11} "
          f"{'-'*7} {'-'*8} {'-'*7}")

    for r in rows:
        binding   = r.cap_g is not None and abs(r.total_emissions - r.cap_g) < 1.0
        cap_str   = f"{r.cap_g:,.0f}" if r.cap_g is not None else "off"
        frozen_ok = "—" if frozen_nodes is None else str(r.assignments == frozen_nodes)
        print(f"  {r.alpha:>6.2f} {r.total_cost:>9.4f} {r.total_deg:>8.4f} "
              f"{r.total_cost + r.total_deg:>9.4f} {r.total_emissions:>11,.0f} "
              f"{cap_str:>11} {r.slack_g:>7.1f} {str(binding):>8} {frozen_ok:>7}")

    problems = 0
    for a, b in zip(rows, rows[1:]):
        # The LP minimises energy cost PLUS degradation, so cost+deg is the
        # quantity that must be monotone. Energy cost alone can wobble because
        # the LP is free to trade a little of it against battery throughput.
        if b.total_cost + b.total_deg > a.total_cost + a.total_deg + 1e-6:
            print(f"  !! cost+deg rose  α {a.alpha:.2f} → {b.alpha:.2f}: "
                  f"{a.total_cost + a.total_deg:.4f} → {b.total_cost + b.total_deg:.4f}")
            problems += 1
        if b.total_cost > a.total_cost + 1e-6:
            print(f"  ·  note: energy cost alone rose  α {a.alpha:.2f} → {b.alpha:.2f}: "
                  f"{a.total_cost:.4f} → {b.total_cost:.4f}")
        if b.total_emissions < a.total_emissions - 1e-3:
            print(f"  !! emissions fell  α {a.alpha:.2f} → {b.alpha:.2f}: "
                  f"{a.total_emissions:,.0f} → {b.total_emissions:,.0f}")
            problems += 1
    if frozen_nodes is not None:
        drifted = [r.alpha for r in rows if r.assignments != frozen_nodes]
        if drifted:
            print(f"  !! allocation drifted at α = {drifted}")
            problems += len(drifted)

    n_binding = sum(1 for r in rows
                    if r.cap_g is not None and abs(r.total_emissions - r.cap_g) < 1.0)
    print(f"\n  Carbon cap binds at {n_binding}/{len(rows)} sweep points.")
    if any(r.slack_g > 1e-4 for r in rows):
        print("  ⚠ Carbon-cap slack was used — the cap is infeasibly tight somewhere.")
    print("  ✓ Frontier is monotone." if problems == 0
          else f"  ✗ {problems} monotonicity/freeze violation(s) above.")
    print("=" * 78 + "\n")


# Core runner

def _run_optimised_scheme(
    label:               str,
    alpha:               float,
    beta:                float,
    evs_orig,
    nodal_df:            pd.DataFrame,
    carbon:              dict[str, dict[int, float]],   # nodal
    time_list:           list[int],
    prices:              dict[int, float],
    node_positions:      dict,
    # LP params
    carbon_cap_fraction: float  = 0.85,
    v2g_enabled:         bool   = True,
    eta_c:               float  = 0.95,
    eta_d:               float  = 0.95,
    deg_cost:            float  = 0.02,
    grid_cap_kw:         float  = 85.0,
    interval_hours:      float  = 1.0,
    max_distance:        float | None = 4.0,
    carbon_baseline_g:   float | None = None,
    frozen_nodes:        dict  | None = None,
) -> SchemeResult:
    evs = _copy_evs(evs_orig)

    if frozen_nodes is None:
        nodal_prices, assignment_log = assign_ev_nodes(
            evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
            alpha=alpha, beta=beta,
            grid_cap_kw=grid_cap_kw,
            interval_hours=interval_hours,
            node_positions=node_positions,
            max_distance=max_distance,
        )
    else:
        # Stage 1 is held fixed: reuse a previously computed allocation instead
        # of re-scoring with this point's alpha/beta. Without this the feasible
        # set shifts at every sweep point and the frontier is not a genuine
        # one-parameter family.
        nodal_prices, assignment_log = _apply_frozen_nodes(
            evs=evs, nodal_df=nodal_df, frozen_nodes=frozen_nodes,
        )

    c, d, energy_vars, status, diag = run_scheduler(
        prices=prices, carbon=carbon, time_slots=time_list, evs=evs,
        interval_hours=interval_hours, alpha=alpha, beta=beta,
        carbon_cap_fraction=carbon_cap_fraction,
        v2g_enabled=v2g_enabled,
        eta_c=eta_c, eta_d=eta_d, deg_cost=deg_cost,
        nodal_prices=nodal_prices, grid_cap_kw=grid_cap_kw,
        carbon_baseline_g=carbon_baseline_g,
    )
    print(f"  [{label}] solver: {status}")

    total_cost, total_emissions, total_energy, avg_intensity, total_deg = compute_metrics(
        prices=prices, carbon=carbon, time_slots=time_list,
        c=c, d=d, evs=evs,
        interval_hours=interval_hours, eta_c=eta_c, eta_d=eta_d,
        deg_cost=deg_cost, nodal_prices=nodal_prices,
    )
    avg_dist = _avg_dist(evs, assignment_log, node_positions)

    return SchemeResult(
        label=label, alpha=alpha, beta=beta,
        total_cost=total_cost, total_emissions=total_emissions,
        total_energy=total_energy, avg_intensity=avg_intensity,
        total_deg=total_deg, avg_dist=avg_dist,
        c=c, d=d, energy_vars=energy_vars,
        nodal_prices=nodal_prices, assignment_log=assignment_log,
        diag=diag,
    )


# Public API

def run_all_schemes(
    evs_orig,
    nodal_df:            pd.DataFrame,
    carbon:              dict[str, dict[int, float]],   # nodal
    time_list:           list[int],
    prices:              dict[int, float],
    # LP / assignment params forwarded to every run
    carbon_cap_fraction: float       = 0.85,
    v2g_enabled:         bool        = True,
    eta_c:               float       = 0.95,
    eta_d:               float       = 0.95,
    deg_cost:            float       = 0.02,
    grid_cap_kw:         float       = 85.0,
    interval_hours:      float       = 1.0,
    max_distance:        float | None = 4.0,
    freeze_pareto_assignments: bool  = True,
) -> tuple[list[SchemeResult], list[ParetoPoint]]:
    """
    Run all four evaluation schemes and the Pareto sweep.

    The carbon cap is anchored to the uncoordinated baseline's actual emissions,
    computed once here and held fixed at every operating point. By default the
    sweep re-runs Stage 1 at every alpha, so each point reflects both the
    reallocation and the dispatch implied by that weighting; set
    freeze_pareto_assignments=True to reuse the balanced allocation instead.

    Returns
    -------
    results : list[SchemeResult]   length 4 — Uncoordinated, Cost, Carbon, Balanced
    pareto  : list[ParetoPoint]    length PARETO_STEPS
    """
    node_positions = get_node_positions()

    # Scheme 0: Uncoordinated baseline
    print("\n[Uncoordinated] running baseline …")
    evs_base = _copy_evs(evs_orig)
    c_base, d_base, energy_base, nodal_prices_b, alog_base = run_baseline(
        evs=evs_base, time_list=time_list, nodal_df=nodal_df, carbon=carbon,
        charging_nodes=CHARGING_NODES, prices=prices,
        interval_hours=interval_hours, eta_c=eta_c,
    )
    tc, te, tnrg, ai, tdeg = compute_metrics(
        prices=prices, carbon=carbon, time_slots=time_list,
        c=c_base, d=d_base, evs=evs_base,
        interval_hours=interval_hours, eta_c=eta_c, eta_d=eta_d,
        deg_cost=deg_cost, nodal_prices=nodal_prices_b,
    )
    avg_dist_base = _avg_dist(evs_base, alog_base, node_positions)

    unmanaged = SchemeResult(
        label="Uncoordinated", alpha=float("nan"), beta=float("nan"),
        total_cost=tc, total_emissions=te, total_energy=tnrg,
        avg_intensity=ai, total_deg=tdeg, avg_dist=avg_dist_base,
        c=c_base, d=d_base, energy_vars=energy_base,
        nodal_prices=nodal_prices_b, assignment_log=alog_base,
    )

    # The carbon cap is phi(beta) x (uncoordinated baseline emissions). Anchoring
    # the baseline to this measured absolute value — rather than recomputing a
    # max-power proxy inside each LP from whatever allocation is current — means
    # the cap fraction multiplies the same number at every operating point.
    carbon_baseline_g = te
    print(f"\n  Carbon-cap baseline (uncoordinated emissions): {carbon_baseline_g:,.2f} g CO₂")
    if carbon_cap_fraction is not None:
        print(f"  Tightest cap (β=1, φ₀={carbon_cap_fraction}): "
              f"{carbon_cap_fraction * carbon_baseline_g:,.2f} g CO₂")

    shared = dict(
        evs_orig=evs_orig, nodal_df=nodal_df, carbon=carbon,
        time_list=time_list, prices=prices,
        node_positions=node_positions,
        carbon_cap_fraction=carbon_cap_fraction,
        v2g_enabled=v2g_enabled, eta_c=eta_c, eta_d=eta_d,
        deg_cost=deg_cost, grid_cap_kw=grid_cap_kw,
        interval_hours=interval_hours, max_distance=max_distance,
        carbon_baseline_g=carbon_baseline_g,
    )

    # Scheme 1: Cost-only
    print("\n[Cost-only] running …")
    cost_only = _run_optimised_scheme(
        label="Cost-only (α=1, β=0)", alpha=1.0, beta=0.0, **shared,
    )

    # Scheme 2: Carbon-only
    print("\n[Carbon-only] running …")
    carbon_only = _run_optimised_scheme(
        label="Carbon-only (α=0, β=1)", alpha=0.0, beta=1.0, **shared,
    )

    # Scheme 3: Balanced
    print("\n[Balanced] running …")
    balanced = _run_optimised_scheme(
        label="Balanced (α=0.5, β=0.5)", alpha=0.5, beta=0.5, **shared,
    )

    results = [unmanaged, cost_only, carbon_only, balanced]

    # Pareto sweep
    # Stage 1 (node allocation) is frozen at the balanced allocation so that
    # alpha is the only quantity varying along the frontier. Re-scoring the
    # allocation at each alpha changes the feasible set between points, which
    # is what produced the non-monotone cost curve.
    frozen_nodes: dict | None = None
    if freeze_pareto_assignments:
        frozen_nodes = {
            name: info["assigned"]
            for name, info in balanced.assignment_log.items()
            if "assigned" in info
        }
        print(f"\n[Pareto] Stage-1 allocation frozen at α=β=0.5 "
              f"({len(frozen_nodes)} EVs).")
    else:
        print("\n[Pareto] Stage-1 allocation re-scored at every α (unfrozen).")

    print(f"[Pareto] sweeping α over {PARETO_STEPS} points …")
    pareto: list[ParetoPoint] = []

    for i in range(PARETO_STEPS):
        a = round(i / (PARETO_STEPS - 1), 4)
        b = round(1.0 - a, 4)
        sr = _run_optimised_scheme(
            label=f"Pareto α={a:.2f}", alpha=a, beta=b,
            frozen_nodes=frozen_nodes, **shared,
        )
        pareto.append(ParetoPoint(
            alpha=a, beta=b,
            total_cost=sr.total_cost, total_emissions=sr.total_emissions,
            total_deg=sr.total_deg,
            cap_g=sr.diag.get("cap_g"),
            slack_g=sr.diag.get("slack_g", 0.0),
            assignments={n: i_["assigned"]
                         for n, i_ in sr.assignment_log.items() if "assigned" in i_},
        ))
        print(f"  α={a:.2f}  cost={sr.total_cost:.2f}  emissions={sr.total_emissions:.2f}")

    print("\nAll schemes complete.\n")
    audit_sweep(pareto, frozen_nodes)
    return results, pareto