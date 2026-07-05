"""
scheduler/plots.py

All visualisation for the EV scheduler.

Figures
───────
  plot_grid_positions  — pre-assignment: just nodes + EVs, no assignment lines
  plot_grid            — post-assignment: nodes, EVs, and assignment lines
  fig_nodal_carbon         — per-node carbon intensity over time
  fig_nodal_prices         — per-node LMP prices over time
  fig_cost_price_overlay   — per-node load vs LMP price  (cost-only scheme)
  fig_carbon_overlay       — fleet load vs per-node carbon intensity  (carbon-only)
  fig_balanced_node_profiles — per-node net power  (balanced scheme)
  fig_pareto               — Pareto frontier cost vs emissions
  table_performance        — printed LaTeX table + saved CSV/tex

  plot_all  — master call: generates all plots + table

Nodal carbon
────────────
  carbon is now dict[str, dict[int, float]]  (node_name → {period → g CO₂/kWh}).
  fig_carbon_overlay draws one coloured intensity line per node on the right
  y-axis so per-node schedule differences are visible alongside the fleet load.
"""

from __future__ import annotations

import csv
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  saved -> {path}")


# ── Restore EV node IDs from an assignment log ────────────────────────────────

def _restore_node_ids(evs, assignment_log: dict) -> None:
    """Mutate the shared evs list so node_id matches a specific scheme's log."""
    for ev in evs:
        if ev.name in assignment_log:
            ev.node_id = assignment_log[ev.name]["assigned"]


# ── Shared: extract aggregate fleet net power per period ─────────────────────

def _fleet_net_power(evs, c: dict, d: dict, time_list: list[int]) -> np.ndarray:
    totals = np.zeros(len(time_list))
    for i, t in enumerate(time_list):
        for ev in evs:
            if not (ev.arrival <= t <= ev.departure):
                continue
            raw_c = c.get((ev.name, t), 0.0)
            raw_d = d.get((ev.name, t), 0.0)
            cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
            dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
            totals[i] += (cv or 0.0) - (dv or 0.0)
    return totals


# ── Shared node colours (consistent across all figures) ──────────────────────

NODE_COLORS = [
    "#2CA25F",  # node 0
    "#E6550D",  # node 1
    "#2171B5",  # node 2
    "#C94040",  # node 3
    "#7B55A8",  # node 4 (if present)
]


# ═════════════════════════════════════════════════════════════════════════════
# Fig — Nodal carbon intensities over time
# ═════════════════════════════════════════════════════════════════════════════

def fig_nodal_carbon(
    carbon:    dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    time_list: list[int],
    save_path: str | None = "plots/fig_nodal_carbon.pdf",
) -> plt.Figure:
    """
    One line per node showing its carbon intensity schedule (g CO₂/kWh).
    No title — intended for direct inclusion in a research paper.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.0))

    for idx, (node_name, schedule) in enumerate(sorted(carbon.items())):
        color  = NODE_COLORS[idx % len(NODE_COLORS)]
        values = np.array([schedule.get(t, 0.0) for t in time_list])
        short  = node_name.split("-")[0]
        ax.plot(
            time_list, values,
            color=color, lw=2.0, marker="o", markersize=3.5,
            label=short,
        )

    ax.set_xlabel("Period (hour)", fontsize=11)
    ax.set_ylabel("Carbon Intensity (g CO$_2$/kWh)", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig — Nodal LMP prices over time
# ═════════════════════════════════════════════════════════════════════════════

def fig_nodal_prices(
    nodal_df,
    time_list: list[int],
    save_path: str | None = "plots/fig_nodal_prices.pdf",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))

    for idx, node_name in enumerate(nodal_df.columns):
        color  = NODE_COLORS[idx % len(NODE_COLORS)]
        values = np.array([
            nodal_df.loc[t, node_name] if t in nodal_df.index else 0.0
            for t in time_list
        ])
        short  = node_name.split("-")[0]
        ax.plot(
            time_list, values,
            color=color, lw=2.0, marker="o", markersize=3.5,
            label=short,
        )

    ax.set_xlabel("Period (hour)", fontsize=11)
    ax.set_ylabel("LMP (\\$/kWh)", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 0 — Pre-assignment grid: just node + EV locations, no lines
# ═════════════════════════════════════════════════════════════════════════════

def plot_grid_positions(
    evs,
    charging_nodes,
    save_path: str | None = "plots/grid_positions.pdf",
) -> plt.Figure:
    """
    Plain grid map showing where EVs and charging nodes sit.
    No assignment lines — use this before assign_ev_nodes() is called,
    or as a standalone position reference.
    """
    GRID = 10

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0.5, GRID + 0.5)
    ax.set_ylim(0.5, GRID + 0.5)
    ax.set_xticks(range(1, GRID + 1))
    ax.set_yticks(range(1, GRID + 1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_aspect("equal")

    # EVs — neutral grey
    for ev in evs:
        ax.scatter(
            ev.grid_x, ev.grid_y,
            s=110, color="#AAAAAA", alpha=0.85,
            edgecolors="black", linewidths=0.5, zorder=2,
        )
        ax.annotate(
            ev.name, (ev.grid_x, ev.grid_y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, zorder=3,
        )

    # Charging nodes — coloured stars
    for idx, node in enumerate(charging_nodes):
        color = NODE_COLORS[idx % len(NODE_COLORS)]
        ax.scatter(
            node.grid_x, node.grid_y,
            s=220, marker="*", color=color,
            edgecolors="black", linewidths=0.8, zorder=5,
        )
        short = node.name.split("-")[0]
        ax.annotate(
            short, (node.grid_x, node.grid_y),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color=color, zorder=6,
        )

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig — Post-assignment grid map with coloured assignment lines
# ═════════════════════════════════════════════════════════════════════════════

def plot_grid(
    evs,
    charging_nodes,
    assignment_log: dict,
    save_path: str | None = "plots/grid_map.pdf",
) -> plt.Figure:
    """
    Grid map showing EV→node assignment lines.
    EVs are coloured by their assigned node; nodes are coloured stars.
    """
    GRID = 10

    node_map   = {cn.name: cn for cn in charging_nodes}
    node_color = {
        cn.name: NODE_COLORS[i % len(NODE_COLORS)]
        for i, cn in enumerate(charging_nodes)
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0.5, GRID + 0.5)
    ax.set_ylim(0.5, GRID + 0.5)
    ax.set_xticks(range(1, GRID + 1))
    ax.set_yticks(range(1, GRID + 1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_aspect("equal")

    # Assignment lines
    for ev in evs:
        assigned = assignment_log[ev.name]["assigned"]
        cn = node_map.get(assigned)
        if cn is None:
            continue
        ax.plot(
            [ev.grid_x, cn.grid_x], [ev.grid_y, cn.grid_y],
            ls="--", lw=1.0, color=node_color[assigned],
            alpha=0.55, zorder=1,
        )

    # EVs — coloured by assigned node
    for ev in evs:
        assigned = assignment_log[ev.name]["assigned"]
        color = node_color.get(assigned, "#AAAAAA")
        ax.scatter(
            ev.grid_x, ev.grid_y,
            s=110, color=color, alpha=0.85,
            edgecolors="black", linewidths=0.5, zorder=3,
        )
        ax.annotate(
            ev.name, (ev.grid_x, ev.grid_y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, zorder=4,
        )

    # Charging nodes — coloured stars
    for idx, cn in enumerate(charging_nodes):
        color = NODE_COLORS[idx % len(NODE_COLORS)]
        ax.scatter(
            cn.grid_x, cn.grid_y,
            s=220, marker="*", color=color,
            edgecolors="black", linewidths=0.8, zorder=5,
        )
        short = cn.name.split("-")[0]
        ax.annotate(
            short, (cn.grid_x, cn.grid_y),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color=color, zorder=6,
        )

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 — Per-node load vs LMP price  (cost-only scheme, α=1)
# ═════════════════════════════════════════════════════════════════════════════

def fig_cost_price_overlay(
    cost_result,
    nodal_df,
    time_list: list[int],
    evs,
    save_path: str | None = "plots/fig_cost_price_overlay.pdf",
) -> plt.Figure:
    """
    One subplot per charging node.
      left  y-axis (line)  — aggregate net power of EVs at that node (kW)
      right y-axis (line)  — that node's own LMP price ($/kWh)
    """
    _restore_node_ids(evs, cost_result.assignment_log)

    node_evs: dict[str, list] = defaultdict(list)
    for ev in evs:
        node_evs[ev.node_id].append(ev)

    nodes_sorted   = sorted(node_evs.keys())
    n_nodes        = len(nodes_sorted)
    color_charge   = "#2171B5"
    color_price    = "#E6550D"

    fig, axes = plt.subplots(n_nodes, 1, figsize=(9.5, 3.6 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    for ax1, node_name in zip(axes, nodes_sorted):
        evs_here  = node_evs[node_name]
        node_load = np.zeros(len(time_list))

        for ev in evs_here:
            for i, t in enumerate(time_list):
                if not (ev.arrival <= t <= ev.departure):
                    continue
                raw_c = cost_result.c.get((ev.name, t), 0.0)
                raw_d = cost_result.d.get((ev.name, t), 0.0)
                cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
                dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
                node_load[i] += (cv or 0.0) - (dv or 0.0)

        if node_name in nodal_df.columns:
            node_price = np.array([
                nodal_df.loc[t, node_name] if t in nodal_df.index else 0.0
                for t in time_list
            ])
        else:
            node_price = np.array([
                nodal_df.loc[t].mean() if t in nodal_df.index else 0.0
                for t in time_list
            ])

        ax1.plot(time_list, node_load, color=color_charge, lw=2.0,
                 marker="o", markersize=3.5, label="Net load (kW)", zorder=3)
        ax1.axhline(0, color="black", lw=0.7)
        ax1.set_ylabel("Net Power (kW)", color=color_charge, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=color_charge)
        ax1.set_xlim(time_list[0] - 0.5, time_list[-1] + 0.5)
        ax1.grid(True, axis="y", alpha=0.3, zorder=0)

        ax2 = ax1.twinx()
        ax2.plot(time_list, node_price, color=color_price, lw=2.0,
                 marker="o", markersize=3.5, label="Node LMP (\\$/kWh)", zorder=3)
        ax2.set_ylabel("LMP (\\$/kWh)", color=color_price, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=color_price)

        short       = node_name.split("-")[0]
        n_ev_label  = f"{len(evs_here)} EV{'s' if len(evs_here) != 1 else ''}"
        ax1.set_title(f"Node: {short}  ({n_ev_label})", loc="left", fontsize=9)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Period (hour)", fontsize=10)
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Fleet load vs per-node carbon intensity  (carbon-only scheme, β=1)
# ═════════════════════════════════════════════════════════════════════════════

def fig_carbon_overlay(
    carbon_result,
    carbon: dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    time_list: list[int],
    evs,
    save_path: str | None = "plots/fig_carbon_overlay.pdf",
) -> plt.Figure:
    """
    One subplot per charging node (mirrors fig_cost_price_overlay structure).
      left  y-axis (line) — aggregate net power of EVs at that node (kW)
      right y-axis (line) — that node's own carbon intensity (g CO₂/kWh)
    """
    _restore_node_ids(evs, carbon_result.assignment_log)

    node_evs: dict[str, list] = defaultdict(list)
    for ev in evs:
        node_evs[ev.node_id].append(ev)

    nodes_sorted   = sorted(node_evs.keys())
    n_nodes        = len(nodes_sorted)
    color_charge   = "#2171B5"
    color_carbon    = "#2CA25F"

    fig, axes = plt.subplots(n_nodes, 1, figsize=(9.5, 3.6 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    for ax1, node_name in zip(axes, nodes_sorted):
        evs_here  = node_evs[node_name]
        node_load = np.zeros(len(time_list))

        for ev in evs_here:
            for i, t in enumerate(time_list):
                if not (ev.arrival <= t <= ev.departure):
                    continue
                raw_c = carbon_result.c.get((ev.name, t), 0.0)
                raw_d = carbon_result.d.get((ev.name, t), 0.0)
                cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
                dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
                node_load[i] += (cv or 0.0) - (dv or 0.0)

        node_carbon = carbon.get(node_name, {})
        ci_vals = np.array([node_carbon.get(t, 0.0) for t in time_list])

        ax1.plot(time_list, node_load, color=color_charge, lw=2.0,
                 marker="o", markersize=3.5, label="Net load (kW)", zorder=3)
        ax1.axhline(0, color="black", lw=0.7)
        ax1.set_ylabel("Net Power (kW)", color=color_charge, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=color_charge)
        ax1.set_xlim(time_list[0] - 0.5, time_list[-1] + 0.5)
        ax1.grid(True, axis="y", alpha=0.3, zorder=0)

        ax2 = ax1.twinx()
        ax2.plot(time_list, ci_vals, color=color_carbon, lw=2.0,
                 marker="o", markersize=3.5,
                 label="Carbon intensity (g CO$_2$/kWh)", zorder=3)
        ax2.set_ylabel("Carbon Intensity (g CO$_2$/kWh)", color=color_carbon, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=color_carbon)

        short      = node_name.split("-")[0]
        n_ev_label = f"{len(evs_here)} EV{'s' if len(evs_here) != 1 else ''}"
        ax1.set_title(f"Node: {short}  ({n_ev_label})", loc="left", fontsize=9)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Period (hour)", fontsize=10)
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 — Per-node net power profiles  (balanced scheme, α=β=0.5)
# ═════════════════════════════════════════════════════════════════════════════

def fig_balanced_node_profiles(
    balanced_result,
    time_list:   list[int],
    evs,
    grid_cap_kw: float = 75.0,
    save_path:   str | None = "plots/fig_balanced_node_profiles.pdf",
) -> plt.Figure:
    """
    One subplot per node showing per-EV thin lines, node aggregate, and cap.
    """
    _restore_node_ids(evs, balanced_result.assignment_log)

    node_evs: dict[str, list] = defaultdict(list)
    for ev in evs:
        node_evs[ev.node_id].append(ev)

    nodes_sorted = sorted(node_evs.keys())
    n_nodes      = len(nodes_sorted)

    fig, axes = plt.subplots(n_nodes, 1, figsize=(9.5, 3.5 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    EV_COLORS = plt.get_cmap("tab20")

    for ax, node_name in zip(axes, nodes_sorted):
        evs_here = node_evs[node_name]
        node_agg = np.zeros(len(time_list))

        for color_idx, ev in enumerate(evs_here):
            net = []
            for t in time_list:
                if ev.arrival <= t <= ev.departure:
                    raw_c = balanced_result.c.get((ev.name, t), 0.0)
                    raw_d = balanced_result.d.get((ev.name, t), 0.0)
                    cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
                    dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
                    net.append((cv or 0.0) - (dv or 0.0))
                else:
                    net.append(0.0)
            net = np.array(net)
            node_agg += net

            ax.plot(
                time_list, net, lw=0.9, alpha=0.55,
                color=EV_COLORS(color_idx / max(len(evs_here) - 1, 1)),
                marker="o", markersize=2, label=f"EV {ev.name}",
            )

        ax.plot(time_list, node_agg, lw=2.2, ls="--", color="black",
                marker="o", markersize=3, label="Node total", zorder=5)
        ax.axhline(grid_cap_kw, color="#C94040", lw=1.6, ls="--",
                   label=f"Node cap ({grid_cap_kw:.0f} kW)", zorder=6)
        ax.axhline(0, color="black", lw=0.7)

        short = node_name.split("-")[0]
        ax.set_title(
            f"Node: {short}  ({len(evs_here)} EV{'s' if len(evs_here) != 1 else ''})",
            loc="left", fontsize=9,
        )
        ax.set_ylabel("Net Power (kW)", fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.legend(ncol=5, fontsize=7, framealpha=0.85)

    axes[-1].set_xlabel("Period (hour)", fontsize=10)
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — Pareto frontier: cost vs emissions
# ═════════════════════════════════════════════════════════════════════════════

def fig_pareto(
    pareto_points:  list,   # list[ParetoPoint]
    scheme_results: list,   # list[SchemeResult] — for annotating key points
    save_path:      str | None = "plots/fig_pareto.pdf",
) -> plt.Figure:
    """
    Scatter + line of (total_cost, total_emissions) as α varies 0 → 1.
    The three managed operating points are annotated with distinct markers.
    """
    costs  = [p.total_cost      for p in pareto_points]
    emits  = [p.total_emissions for p in pareto_points]
    alphas = [p.alpha           for p in pareto_points]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    sc = ax.scatter(costs, emits, c=alphas, cmap="RdYlGn_r",
                    s=55, zorder=4, edgecolors="white", linewidths=0.5)
    ax.plot(costs, emits, color="grey", lw=1.0, ls="--", zorder=3, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("$\\alpha$ (cost weight)", fontsize=9)

    MARKERS = {
        1: ("Cost-only\n($\\alpha=1$)",           "^", "#1F78B4"),
        2: ("Carbon-only\n($\\beta=1$)",           "v", "#33A02C"),
        3: ("Balanced\n($\\alpha=\\beta=0.5$)",    "D", "#E31A1C"),
    }
    for idx, (label, marker, color) in MARKERS.items():
        sr = scheme_results[idx]
        ax.scatter(
            sr.total_cost, sr.total_emissions,
            marker=marker, s=140, color=color, zorder=6,
            edgecolors="black", linewidths=0.7, label=label,
        )

    ax.set_xlabel("Total Electricity Cost (\\$)", fontsize=11)
    ax.set_ylabel("Total Carbon Emissions (g CO$_2$)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        _save(fig, save_path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Table — Performance comparison across all 4 schemes
# ═════════════════════════════════════════════════════════════════════════════

def table_performance(
    results:  list,   # list[SchemeResult], length 4
    save_dir: str = "plots",
) -> str:
    """
    Prints a formatted performance table and saves:
      performance_table.csv  — raw numbers
      performance_table.tex  — ready-to-paste LaTeX table body
    Returns the LaTeX string.
    """
    unmanaged = results[0]

    def pct(val, base):
        if abs(base) < 1e-9:
            return "—"
        return f"{(val - base) / abs(base) * 100:+.1f}\\%"

    rows = []
    for sr in results:
        rows.append({
            "Scheme":          sr.label,
            "Cost ($)":        f"{sr.total_cost:.2f}",
            "Emissions (g)":   f"{sr.total_emissions:.2f}",
            "Degradation ($)": f"{sr.total_deg:.2f}",
            "Δ Cost":          pct(sr.total_cost,      unmanaged.total_cost),
            "Δ Emissions":     pct(sr.total_emissions, unmanaged.total_emissions),
            "Δ Deg":           pct(sr.total_deg,        unmanaged.total_deg),
        })

    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "performance_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  saved -> {csv_path}")

    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Performance Comparison Across Scheduling Schemes}",
        r"\label{tab:performance}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Scheme & Cost (\$) & Emiss.\ (g CO$_2$) & Deg.\ (\$)"
        r" & $\Delta$Cost & $\Delta$Emiss. & $\Delta$Deg. \\",
        r"\midrule",
    ]
    for r in rows:
        latex_lines.append(
            f"{r['Scheme']} & {r['Cost ($)']} & {r['Emissions (g)']} "
            f"& {r['Degradation ($)']} & {r['Δ Cost']} & {r['Δ Emissions']} "
            f"& {r['Δ Deg']} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex_str = "\n".join(latex_lines)

    tex_path = os.path.join(save_dir, "performance_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"  saved -> {tex_path}")

    print("\n── Performance Table ───────────────────────────────────────────────────")
    header = (f"  {'Scheme':<34} {'Cost ($)':>10} {'Emiss.(g)':>12} "
              f"{'Deg($)':>10} {'ΔCost':>8} {'ΔEmiss':>9} {'ΔDeg':>8}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['Scheme']:<34} {r['Cost ($)']:>10} {r['Emissions (g)']:>12} "
            f"{r['Degradation ($)']:>10} {r['Δ Cost']:>8} {r['Δ Emissions']:>9} "
            f"{r['Δ Deg']:>8}"
        )
    print()

    return latex_str


# ═════════════════════════════════════════════════════════════════════════════
# Master call
# ═════════════════════════════════════════════════════════════════════════════

def plot_all(
    results:        list,   # list[SchemeResult], length 4
    pareto_points:  list,   # list[ParetoPoint]
    evs,
    nodal_df,
    carbon:         dict[str, dict[int, float]],   # nodal: {node → {t → g/kWh}}
    time_list:      list[int],
    charging_nodes,
    grid_cap_kw:    float = 75.0,
    out_dir:        str   = "plots",
) -> None:
    """
    Generate and save all figures + performance table to `out_dir`.

      grid_positions.pdf             — pre-assignment node + EV locations
      fig_grid_map.pdf               — post-assignment grid with assignment lines
      fig_nodal_carbon.pdf           — per-node carbon intensity schedules
      fig_nodal_prices.pdf           — per-node LMP price schedules
      fig_cost_price_overlay.pdf     — load vs LMP, cost-only scheme
      fig_carbon_overlay.pdf         — load vs per-node carbon intensity,
                                       carbon-only scheme
      fig_balanced_node_profiles.pdf — per-node profiles, balanced scheme
      fig_pareto.pdf                 — Pareto frontier
      performance_table.csv / .tex   — performance comparison table
    """
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)

    cost_result     = results[1]
    carbon_result   = results[2]
    balanced_result = results[3]

    print(f"\n{'='*60}")
    print("  Generating figures …")
    print(f"{'='*60}\n")

    plot_grid_positions(
        evs=evs, charging_nodes=charging_nodes,
        save_path=p("grid_positions.pdf"),
    )
    plt.close("all")

    plot_grid(
        evs=evs, charging_nodes=charging_nodes,
        assignment_log=balanced_result.assignment_log,
        save_path=p("fig_grid_map.pdf"),
    )
    plt.close("all")

    fig_nodal_carbon(
        carbon=carbon, time_list=time_list,
        save_path=p("fig_nodal_carbon.pdf"),
    )
    plt.close("all")

    fig_nodal_prices(
        nodal_df=nodal_df, time_list=time_list,
        save_path=p("fig_nodal_prices.pdf"),
    )
    plt.close("all")

    fig_cost_price_overlay(
        cost_result=cost_result, nodal_df=nodal_df,
        time_list=time_list, evs=evs,
        save_path=p("fig_cost_price_overlay.pdf"),
    )
    plt.close("all")

    fig_carbon_overlay(
        carbon_result=carbon_result, carbon=carbon,
        time_list=time_list, evs=evs,
        save_path=p("fig_carbon_overlay.pdf"),
    )
    plt.close("all")

    fig_balanced_node_profiles(
        balanced_result=balanced_result, time_list=time_list,
        evs=evs, grid_cap_kw=grid_cap_kw,
        save_path=p("fig_balanced_node_profiles.pdf"),
    )
    plt.close("all")

    fig_pareto(
        pareto_points=pareto_points, scheme_results=results,
        save_path=p("fig_pareto.pdf"),
    )
    plt.close("all")

    table_performance(results=results, save_dir=out_dir)

    print(f"\nAll outputs saved to '{out_dir}/'")