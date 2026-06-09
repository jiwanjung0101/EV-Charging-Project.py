"""
scheduler/results.py

Runs all four evaluation schemes and the Pareto sweep, returning
structured results for paper plots and the performance table.

Schemes
───────
  0  Unmanaged   — greedy max-power charging, closest-node assignment
  1  Cost-only   — α=1, β=0
  2  Carbon-only — α=0, β=1
  3  Balanced    — α=0.5, β=0.5

Pareto sweep
────────────
  α swept from 0→1 in PARETO_STEPS uniform steps (β = 1 − α).
  Each step re-runs nodal assignment + LP and records (total_cost,
  total_emissions).

Nodal carbon
────────────
  carbon is now dict[str, dict[int, float]]  (node → {period → g CO₂/kWh}).
  It is passed unchanged to assign_ev_nodes, run_scheduler, run_baseline,
  and compute_metrics, all of which look up each EV's node's own schedule.

Usage
─────
  from scheduler.results import run_all_schemes, SCHEME_LABELS
  results, pareto = run_all_schemes(...)
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

# ── Constants ─────────────────────────────────────────────────────────────────

SCHEME_LABELS: list[str] = [
    "Unmanaged",
    "Cost-only (α=1, β=0)",
    "Carbon-only (α=0, β=1)",
    "Balanced (α=0.5, β=0.5)",
]

PARETO_STEPS: int = 21   # α ∈ {0.00, 0.05, …, 1.00}


# ── Result containers ─────────────────────────────────────────────────────────

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


@dataclass
class ParetoPoint:
    alpha:           float
    beta:            float
    total_cost:      float
    total_emissions: float


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Core runner ───────────────────────────────────────────────────────────────

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
    grid_cap_kw:         float  = 75.0,
    interval_hours:      float  = 1.0,
    max_distance:        float | None = 4.0,
) -> SchemeResult:
    evs = _copy_evs(evs_orig)

    nodal_prices, assignment_log = assign_ev_nodes(
        evs=evs, nodal_df=nodal_df, carbon=carbon, time_list=time_list,
        alpha=alpha, beta=beta,
        grid_cap_kw=grid_cap_kw,
        interval_hours=interval_hours,
        node_positions=node_positions,
        max_distance=max_distance,
    )

    c, d, energy_vars, status = run_scheduler(
        prices=prices, carbon=carbon, time_slots=time_list, evs=evs,
        interval_hours=interval_hours, alpha=alpha, beta=beta,
        carbon_cap_fraction=carbon_cap_fraction,
        v2g_enabled=v2g_enabled,
        eta_c=eta_c, eta_d=eta_d, deg_cost=deg_cost,
        nodal_prices=nodal_prices, grid_cap_kw=grid_cap_kw,
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
    )


# ── Public API ────────────────────────────────────────────────────────────────

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
    grid_cap_kw:         float       = 75.0,
    interval_hours:      float       = 1.0,
    max_distance:        float | None = 4.0,
) -> tuple[list[SchemeResult], list[ParetoPoint]]:
    """
    Run all four evaluation schemes and the Pareto sweep.

    Returns
    -------
    results : list[SchemeResult]   length 4 — Unmanaged, Cost, Carbon, Balanced
    pareto  : list[ParetoPoint]    length PARETO_STEPS
    """
    node_positions = get_node_positions()

    shared = dict(
        evs_orig=evs_orig, nodal_df=nodal_df, carbon=carbon,
        time_list=time_list, prices=prices,
        node_positions=node_positions,
        carbon_cap_fraction=carbon_cap_fraction,
        v2g_enabled=v2g_enabled, eta_c=eta_c, eta_d=eta_d,
        deg_cost=deg_cost, grid_cap_kw=grid_cap_kw,
        interval_hours=interval_hours, max_distance=max_distance,
    )

    # ── Scheme 0: Unmanaged baseline ──────────────────────────────────────────
    print("\n[Unmanaged] running baseline …")
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
        label="Unmanaged", alpha=float("nan"), beta=float("nan"),
        total_cost=tc, total_emissions=te, total_energy=tnrg,
        avg_intensity=ai, total_deg=tdeg, avg_dist=avg_dist_base,
        c=c_base, d=d_base, energy_vars=energy_base,
        nodal_prices=nodal_prices_b, assignment_log=alog_base,
    )

    # ── Scheme 1: Cost-only ───────────────────────────────────────────────────
    print("\n[Cost-only] running …")
    cost_only = _run_optimised_scheme(
        label="Cost-only (α=1, β=0)", alpha=1.0, beta=0.0, **shared,
    )

    # ── Scheme 2: Carbon-only ─────────────────────────────────────────────────
    print("\n[Carbon-only] running …")
    carbon_only = _run_optimised_scheme(
        label="Carbon-only (α=0, β=1)", alpha=0.0, beta=1.0, **shared,
    )

    # ── Scheme 3: Balanced ────────────────────────────────────────────────────
    print("\n[Balanced] running …")
    balanced = _run_optimised_scheme(
        label="Balanced (α=0.5, β=0.5)", alpha=0.5, beta=0.5, **shared,
    )

    results = [unmanaged, cost_only, carbon_only, balanced]

    # ── Pareto sweep ──────────────────────────────────────────────────────────
    print(f"\n[Pareto] sweeping α over {PARETO_STEPS} points …")
    pareto: list[ParetoPoint] = []

    for i in range(PARETO_STEPS):
        a = round(i / (PARETO_STEPS - 1), 4)
        b = round(1.0 - a, 4)
        sr = _run_optimised_scheme(
            label=f"Pareto α={a:.2f}", alpha=a, beta=b, **shared,
        )
        pareto.append(ParetoPoint(
            alpha=a, beta=b,
            total_cost=sr.total_cost, total_emissions=sr.total_emissions,
        ))
        print(f"  α={a:.2f}  cost={sr.total_cost:.2f}  emissions={sr.total_emissions:.2f}")

    print("\nAll schemes complete.\n")
    return results, pareto