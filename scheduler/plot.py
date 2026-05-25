"""
scheduler/plot.py

All visualisation for the EV scheduler.

Changes vs previous version
────────────────────────────
• plot_ev_energy   : combined single figure showing all 30 EVs as individual
                     lines on one set of axes (replaces n-subplot layout).
• plot_fleet_power : now split by node — one subplot per node showing
                     per-EV net power and node aggregate.
• plot_grid        : unchanged.
• All other plots  : unchanged.

Call `plot_all(...)` from main.py to generate everything in one shot.
"""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

from scheduler.model import active_slots


def _savefig(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")


# ── 1. Prices ─────────────────────────────────────────────────────────────────

def plot_prices(
    prices:    dict[int, float],
    time_list: list[int],
    nodal_df=None,
    save_path: str | None = "prices.png",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Nodal LMP Prices ($/kWh)")

    if nodal_df is not None:
        for node in nodal_df.columns:
            ax.plot(time_list, [nodal_df.loc[t, node] for t in time_list],
                    marker="o", markersize=3, label=node)

    ax.plot(time_list, [prices[t] for t in time_list],
            lw=2, ls="--", color="black", label="Mean (fallback)")

    ax.set_xlabel("Period (hour)")
    ax.set_ylabel("Price ($/kWh)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# ── 2. Carbon intensity ───────────────────────────────────────────────────────

def plot_carbon(
    carbon:    dict[int, float],
    time_list: list[int],
    save_path: str | None = "carbon.png",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_title("Carbon Intensity (g CO2/kWh)")

    ax.plot(time_list, [carbon[t] for t in time_list],
            marker="o", markersize=3, color="tab:orange")

    ax.set_xlabel("Period (hour)")
    ax.set_ylabel("g CO2/kWh")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# ── 3. Per-EV power ───────────────────────────────────────────────────────────

def plot_ev_power(
    evs,
    c, d,
    time_list: list[int],
    save_path: str | None = "ev_power.png",
) -> plt.Figure:
    n = len(evs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Per-EV Charge / Discharge Power (kW)")

    for ax, ev in zip(axes, evs):
        slot_set = set(active_slots(ev, time_list))

        charge_vals, discharge_vals = [], []
        for t in time_list:
            if t in slot_set:
                raw_c = c.get((ev.name, t), 0.0)
                raw_d = d.get((ev.name, t), 0.0)
                cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
                dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
                charge_vals.append(cv or 0.0)
                discharge_vals.append(dv or 0.0)
            else:
                charge_vals.append(0.0)
                discharge_vals.append(0.0)

        net_vals = [cv - dv for cv, dv in zip(charge_vals, discharge_vals)]

        ax.plot(time_list, charge_vals,                   marker="o", markersize=3,
                label="Charge",    color="tab:blue")
        ax.plot(time_list, [-v for v in discharge_vals],  marker="o", markersize=3,
                label="Discharge", color="tab:red")
        ax.plot(time_list, net_vals,                      marker="o", markersize=3,
                label="Net",       color="tab:green", ls="--")
        ax.axhline(0, color="black", lw=0.8)

        ax.set_ylabel("kW")
        ax.set_title(
            f"{ev.name}  (arr={ev.arrival}, dep={ev.departure}, node={ev.node_id})",
            loc="left", fontsize=9,
        )
        ax.grid(True)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Period (hour)")
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# ── 4. Aggregate fleet power — split by node ──────────────────────────────────

def plot_fleet_power(
    evs,
    c, d,
    time_list:   list[int],
    grid_cap_kw: float = 50.0,
    save_path:   str | None = "fleet_power.png",
) -> plt.Figure:
    """
    One subplot per node.  Each subplot shows:
      • A thin line per EV assigned to that node (net power)
      • A thick dashed line for the node aggregate
      • A red dotted line at grid_cap_kw
    """
    # Group EVs by node
    node_evs: dict[str, list] = defaultdict(list)
    for ev in evs:
        node_evs[ev.node_id].append(ev)

    nodes_sorted = sorted(node_evs.keys())
    n_nodes = len(nodes_sorted)

    fig, axes = plt.subplots(
        n_nodes, 1,
        figsize=(11, 3.5 * n_nodes),
        sharex=True,
    )
    if n_nodes == 1:
        axes = [axes]

    fig.suptitle("Aggregate Fleet Net Power by Node (kW)", fontsize=12, y=1.01)

    for ax, node_name in zip(axes, nodes_sorted):
        evs_here = node_evs[node_name]
        node_total = np.zeros(len(time_list))

        for ev in evs_here:
            slot_set = set(active_slots(ev, time_list))
            net = []
            for t in time_list:
                if t in slot_set:
                    raw_c = c.get((ev.name, t), 0.0)
                    raw_d = d.get((ev.name, t), 0.0)
                    cv = raw_c.varValue if hasattr(raw_c, "varValue") else raw_c
                    dv = raw_d.varValue if hasattr(raw_d, "varValue") else raw_d
                    net.append((cv or 0.0) - (dv or 0.0))
                else:
                    net.append(0.0)
            net = np.array(net)
            node_total += net
            ax.plot(time_list, net, lw=0.9, alpha=0.55,
                    marker="o", markersize=2, label=ev.name)

        ax.plot(time_list, node_total, lw=2.2, ls="--", color="black",
                marker="o", markersize=3, label="Node total")
        ax.axhline(grid_cap_kw, color="tab:red", lw=1.4, ls=":",
                   label=f"Node cap ({grid_cap_kw:.0f} kW)")
        ax.axhline(0, color="black", lw=0.8)

        # Short node label for title
        short = node_name.split("-")[0]
        ax.set_title(
            f"Node: {short}  ({len(evs_here)} EV{'s' if len(evs_here) != 1 else ''})",
            loc="left", fontsize=9,
        )
        ax.set_ylabel("Net power (kW)")
        ax.grid(True)
        ax.legend(ncol=4, fontsize=7)

    axes[-1].set_xlabel("Period (hour)")
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# ── 5. Per-EV energy / SoC — all EVs on one combined chart ───────────────────

def plot_ev_energy(
    evs,
    energy_vars: dict,
    time_list:   list[int],
    save_path:   str | None = "ev_energy.png",
) -> plt.Figure:
    """
    All EVs on a single axes — each EV is one line coloured by a continuous
    palette so 30 lines are distinguishable.  Desired-energy and capacity
    references are drawn as single horizontal bands using the fleet median
    (to avoid 30 duplicate legend entries).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Per-EV Battery Energy — all EVs (kWh)", fontsize=12)

    cmap   = plt.get_cmap("tab20")
    n_evs  = len(evs)
    colors = [cmap(i / max(n_evs - 1, 1)) for i in range(n_evs)]

    cap_vals     = [ev.battery_capacity for ev in evs]
    desired_vals = [ev.desired_energy   for ev in evs]

    # Shaded bands for fleet capacity range and desired-energy range
    ax.axhspan(min(desired_vals), max(desired_vals),
               alpha=0.08, color="tab:orange", label="Desired energy range")
    ax.axhspan(min(cap_vals), max(cap_vals),
               alpha=0.05, color="tab:gray",   label="Capacity range")

    for i, ev in enumerate(evs):
        if ev.name not in energy_vars:
            continue

        energy = energy_vars[ev.name]

        # Support both LP (LpVariable) and baseline (plain float) energy dicts
        if isinstance(energy, dict):
            slots  = sorted(energy.keys())
            e_vals = []
            for t in slots:
                val = energy[t]
                e_vals.append(val.varValue if hasattr(val, "varValue") else val)
        else:
            continue

        ax.plot(slots, e_vals,
                color=colors[i], lw=1.3, alpha=0.75,
                marker="o", markersize=2.5,
                label=ev.name)

    ax.set_xlabel("Period (hour)")
    ax.set_ylabel("Energy (kWh)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.4)

    # Legend: keep EV lines + band entries; place outside plot if many EVs
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,
              ncol=5, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.tight_layout()

    if save_path:
        _savefig(fig, save_path)
    return fig


# ── 6. Grid map ───────────────────────────────────────────────────────────────

_NODE_PALETTE = [
    "#1D9E75", "#E07B39", "#3A7EBF", "#C94040", "#7B55A8",
    "#B5860D", "#2AABB8", "#D45E9A", "#5E8C3A", "#7B6E5A",
]


def plot_grid(
    evs,
    charging_nodes,
    assignment_log: dict,
    save_path: str | None = "grid_map.png",
) -> plt.Figure:
    GRID = 10

    node_map   = {cn.name: cn for cn in charging_nodes}
    node_names = [cn.name for cn in charging_nodes]
    node_color = {name: _NODE_PALETTE[i % len(_NODE_PALETTE)]
                  for i, name in enumerate(node_names)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("EV Charging Grid — Node Assignments", fontsize=13, pad=12)

    for i in range(1, GRID + 2):
        ax.axhline(i - 0.5, color="lightgrey", lw=0.6, zorder=0)
        ax.axvline(i - 0.5, color="lightgrey", lw=0.6, zorder=0)

    ax.set_xlim(0.5, GRID + 0.5)
    ax.set_ylim(0.5, GRID + 0.5)
    ax.set_xticks(range(1, GRID + 1))
    ax.set_yticks(range(1, GRID + 1))
    ax.set_xlabel("Grid X", fontsize=10)
    ax.set_ylabel("Grid Y", fontsize=10)
    ax.set_aspect("equal")

    for ev in evs:
        assigned = assignment_log[ev.name]["assigned"]
        cn       = node_map.get(assigned)
        if cn is None:
            continue
        color = node_color[assigned]
        dist  = math.sqrt((ev.grid_x - cn.grid_x) ** 2 +
                          (ev.grid_y - cn.grid_y) ** 2)
        mx = (ev.grid_x + cn.grid_x) / 2
        my = (ev.grid_y + cn.grid_y) / 2

        ax.plot(
            [ev.grid_x, cn.grid_x], [ev.grid_y, cn.grid_y],
            ls="--", lw=0.9, color=color, alpha=0.55, zorder=1,
        )
        ax.text(
            mx, my, f"{dist:.1f}",
            fontsize=6.5, color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            zorder=2,
        )

    for ev in evs:
        assigned = assignment_log[ev.name]["assigned"]
        color    = node_color.get(assigned, "grey")
        ax.scatter(ev.grid_x, ev.grid_y, s=120, color=color,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(ev.grid_x, ev.grid_y, ev.name,
                fontsize=5.5, ha="center", va="center",
                color="white", fontweight="bold", zorder=4)

    for cn in charging_nodes:
        color = node_color[cn.name]
        short = cn.name.split("-")[0].split("_")[-1]
        ax.scatter(cn.grid_x, cn.grid_y, s=380, color=color, marker="*",
                   edgecolors="white", linewidths=0.9, zorder=5)
        ax.text(cn.grid_x, cn.grid_y + 0.42, short,
                fontsize=7, ha="center", va="bottom",
                color=color, fontweight="bold", zorder=6)

    legend_handles = [
        mpatches.Patch(color=node_color[cn.name],
                       label=f"{cn.name.split('-')[0]}  ({cn.grid_x},{cn.grid_y})")
        for cn in charging_nodes
    ]
    legend_handles += [
        plt.scatter([], [], s=120, color="grey",
                    edgecolors="white", linewidths=0.8, label="EV (coloured by node)"),
        plt.scatter([], [], s=380, color="grey", marker="*",
                    edgecolors="white", linewidths=0.9, label="Charging node ★"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5,
              loc="upper right", framealpha=0.9, title="Nodes", title_fontsize=8)

    fig.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    return fig


# ── Master call ───────────────────────────────────────────────────────────────

def plot_all(
    prices:          dict[int, float],
    carbon:          dict[int, float],
    time_list:       list[int],
    evs,
    c, d,
    energy_vars:     dict,
    charging_nodes,
    assignment_log:  dict,
    nodal_df=None,
    nodal_prices:    dict | None = None,
    interval_hours:  float = 1.0,
    grid_cap_kw:     float = 50.0,
    out_dir:         str   = ".",
) -> None:
    """Generate and save all plots to `out_dir`."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)

    print("\nGenerating plots...")
    plot_prices(     prices, time_list, nodal_df=nodal_df,  save_path=p("prices.png"))
    plot_carbon(     carbon, time_list,                     save_path=p("carbon.png"))
    plot_ev_power(   evs, c, d, time_list,                  save_path=p("ev_power.png"))
    plot_fleet_power(evs, c, d, time_list,
                     grid_cap_kw=grid_cap_kw,               save_path=p("fleet_power.png"))
    plot_ev_energy(  evs, energy_vars, time_list,           save_path=p("ev_energy.png"))
    plot_grid(       evs, charging_nodes, assignment_log,   save_path=p("grid_map.png"))
    print("Done.\n")
    plt.show()