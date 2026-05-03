import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"
LARGE  = 20
MEDIUM = 18
SMALL  = 14


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
#  Input data plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_prices(time_list, price_list, save_path="plots/prices.png"):
    """Line plot of electricity prices over the 48 time slots."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_list, price_list, color="black", linewidth=2)
    ax.set_xlabel("Time", fontsize=LARGE)
    ax.set_ylabel("Price ($ per kWh)", fontsize=LARGE)
    ax.tick_params(labelsize=MEDIUM)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_carbon_intensity(time_list, carbon_list, save_path="plots/carbon_intensity.png"):
    """Line plot of carbon intensity over the 48 time slots."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_list, carbon_list, color="green", linewidth=2)
    ax.set_xlabel("Time", fontsize=LARGE)
    ax.set_ylabel("Carbon Intensity (gCO2/kWh)", fontsize=LARGE)
    ax.tick_params(labelsize=MEDIUM)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Schedule result plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_power(time_list, total_charge, total_discharge,
               save_path="plots/power.png"):
    """Total fleet charging and discharging power over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_list, total_charge,    label="Total Charging (kW)",    linewidth=2)
    ax.plot(time_list, total_discharge, label="Total Discharging (kW)", linewidth=2)
    ax.set_xlabel("Time", fontsize=LARGE)
    ax.set_ylabel("Power (kW)", fontsize=LARGE)
    ax.tick_params(labelsize=MEDIUM)
    ax.legend(fontsize=SMALL)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_power_price(time_list, total_charge, total_discharge, price_list,
                     save_path="plots/power_price.png"):
    """Charging/discharging power on the left axis, price overlaid on the right."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(time_list, total_charge,    label="Total Charging (kW)",    linewidth=2)
    ax1.plot(time_list, total_discharge, label="Total Discharging (kW)", linewidth=2)
    ax1.set_xlabel("Time", fontsize=LARGE)
    ax1.set_ylabel("Power (kW)", fontsize=LARGE)
    ax1.tick_params(labelsize=MEDIUM)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(time_list, price_list, color="black", linewidth=2,
             linestyle="--", label="Price ($/kWh)")
    ax2.set_ylabel("Price ($/kWh)", fontsize=LARGE)
    ax2.tick_params(labelsize=MEDIUM)

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, fontsize=SMALL, loc="upper left")

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_energy(time_list, ev_energy_dict, ev_name,
                save_path="plots/energy.png"):
    """
    State-of-charge profile for a single EV.

    Parameters
    ----------
    ev_energy_dict : dict  {t -> LpVariable}  from energy_vars[ev_name]
    """
    energy_full = [
        ev_energy_dict[t].value() if t in ev_energy_dict else None
        for t in time_list
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_list, energy_full, color="orange", linewidth=2,
            label=f"Energy Level of {ev_name} (kWh)")
    ax.set_xlabel("Time", fontsize=LARGE)
    ax.set_ylabel("Energy (kWh)", fontsize=LARGE)
    ax.tick_params(labelsize=MEDIUM)
    ax.legend(fontsize=MEDIUM)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_schedule(prices, carbon, time_slots, c, d, evs,
                  alpha, beta, interval_hours=0.5,
                  save_path="plots/schedule.png"):
    """
    Stacked bar chart of per-EV net power with price and carbon overlaid.

    Positive bars = charging, negative bars = discharging (V2G).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors   = plt.cm.tab10.colors
    t_arr    = np.array(time_slots)
    bot_pos  = np.zeros(len(time_slots))
    bot_neg  = np.zeros(len(time_slots))

    for k, ev in enumerate(evs):
        net = np.array([
            (c[(ev.name, t)].varValue or 0) - (d[(ev.name, t)].varValue or 0)
            for t in time_slots
        ])
        pos = np.where(net > 0, net, 0)
        neg = np.where(net < 0, net, 0)

        ax1.bar(t_arr, pos, bottom=bot_pos, width=0.8,
                color=colors[k % 10], label=ev.name, alpha=0.85)
        ax1.bar(t_arr, neg, bottom=bot_neg, width=0.8,
                color=colors[k % 10], alpha=0.85)
        bot_pos += pos
        bot_neg += neg

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Net Power (kW)", fontsize=LARGE)
    ax1.set_title(f"Charging Schedule  (α={alpha}, β={beta})",
                  fontsize=LARGE, fontweight="bold")
    ax1.legend(fontsize=SMALL)
    ax1.tick_params(labelsize=MEDIUM)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    price_vals  = [prices[t] for t in time_slots]
    carbon_vals = [carbon[t] for t in time_slots]

    ax2_r = ax2.twinx()
    ax2.plot(time_slots, price_vals,   color="#e67e22", linewidth=2,
             label="Price ($/kWh)")
    ax2_r.plot(time_slots, carbon_vals, color="#27ae60", linewidth=2,
               linestyle="--", label="Carbon (gCO₂/kWh)")
    ax2.set_xlabel("Time Slot", fontsize=LARGE)
    ax2.set_ylabel("Price ($/kWh)",          fontsize=MEDIUM, color="#e67e22")
    ax2_r.set_ylabel("Carbon (gCO₂/kWh)",   fontsize=MEDIUM, color="#27ae60")
    ax2.tick_params(labelsize=MEDIUM)
    ax2_r.tick_params(labelsize=MEDIUM)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    lines  = ax2.get_lines() + ax2_r.get_lines()
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=SMALL, loc="upper right")

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Alpha–Beta analysis plots
# ─────────────────────────────────────────────────────────────────────────────

_GREEN_RED = LinearSegmentedColormap.from_list("gr", ["#2ecc71", "#e74c3c"])


def plot_pareto(df, save_path="plots/pareto_frontier.png"):
    """
    Scatter of total cost vs total emissions for every (α, β) run.
    Point colour reflects how price-weighted the objective was.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ratio = df["alpha"] / (df["alpha"] + df["beta"] + 1e-9)
    sc = ax.scatter(
        df["total_cost"], df["total_emissions"],
        c=ratio, cmap=_GREEN_RED, s=80,
        edgecolors="k", linewidths=0.4, zorder=3
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"α{row['alpha']:.2f} β{row['beta']:.2f}",
            (row["total_cost"], row["total_emissions"]),
            fontsize=7, xytext=(4, 3), textcoords="offset points",
            color="#333333"
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("α / (α + β)  →  price weight", fontsize=SMALL)
    ax.set_xlabel("Total Cost ($)",         fontsize=LARGE)
    ax.set_ylabel("Total Emissions (gCO₂)", fontsize=LARGE)
    ax.set_title("Cost–Emissions Trade-off  (Pareto Frontier)",
                 fontsize=LARGE, fontweight="bold")
    ax.tick_params(labelsize=MEDIUM)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_heatmaps(df, save_path="plots/heatmaps.png"):
    """
    Side-by-side heatmaps of total cost and total emissions
    over the full α–β parameter grid.
    """
    alphas = sorted(df["alpha"].unique())
    betas  = sorted(df["beta"].unique())

    cost_grid  = np.full((len(betas), len(alphas)), np.nan)
    emiss_grid = np.full((len(betas), len(alphas)), np.nan)

    a_idx = {a: i for i, a in enumerate(alphas)}
    b_idx = {b: i for i, b in enumerate(betas)}

    for _, row in df.iterrows():
        i = b_idx[row["beta"]]
        j = a_idx[row["alpha"]]
        cost_grid[i, j]  = row["total_cost"]
        emiss_grid[i, j] = row["total_emissions"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid, title, cmap in zip(
        axes,
        [cost_grid, emiss_grid],
        ["Total Cost ($)", "Total Emissions (gCO₂)"],
        ["YlOrRd", "YlGn"]
    ):
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.2f}" for a in alphas],
                           rotation=45, ha="right", fontsize=SMALL)
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"{b:.2f}" for b in betas], fontsize=SMALL)
        ax.set_xlabel("Alpha  (price weight)",  fontsize=MEDIUM)
        ax.set_ylabel("Beta  (carbon weight)",  fontsize=MEDIUM)
        ax.set_title(title, fontsize=LARGE, fontweight="bold")

        # Annotate each cell
        for i in range(len(betas)):
            for j in range(len(alphas)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if val > np.nanmean(grid) else "black")

    fig.suptitle("Alpha–Beta Grid Search", fontsize=LARGE,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()
def _get_pareto_front(df):
    """
    Return only the non-dominated rows from df.
    A point is non-dominated if no other point has both
    lower (or equal) cost AND lower (or equal) emissions,
    with at least one strictly lower.
    """
    pareto_mask = []
    costs     = df["total_cost"].values
    emissions = df["total_emissions"].values

    for i in range(len(df)):
        dominated = any(
            (costs[j] <= costs[i] and emissions[j] <= emissions[i]) and
            (costs[j] <  costs[i] or  emissions[j] <  emissions[i])
            for j in range(len(df)) if j != i
        )
        pareto_mask.append(not dominated)

    return df[pareto_mask].copy()


def _get_pareto_front(df):
    """
    Return only the non-dominated rows from df.
    A point is non-dominated if no other point has both
    lower (or equal) cost AND lower (or equal) emissions,
    with at least one strictly lower.
    """
    pareto_mask = []
    costs     = df["total_cost"].values
    emissions = df["total_emissions"].values

    for i in range(len(df)):
        dominated = any(
            (costs[j] <= costs[i] and emissions[j] <= emissions[i]) and
            (costs[j] <  costs[i] or  emissions[j] <  emissions[i])
            for j in range(len(df)) if j != i
        )
        pareto_mask.append(not dominated)

    return df[pareto_mask].copy()


def plot_pareto_curve(df, save_path="plots/pareto_curve.png"):
    """
    Plot the true Pareto frontier of cost vs emissions.

    Non-dominated solutions are connected as a step curve and
    labelled with their (α, β) weights. Dominated solutions are
    shown as faint grey dots for context. The knee point (best
    balance between cost and emissions) is marked with a star.

    Parameters
    ----------
    df        : pd.DataFrame with columns:
                  total_cost, total_emissions, alpha, beta
    save_path : str  path to save the figure (None to skip saving)
    """
    from adjustText import adjust_text

    pareto_df    = _get_pareto_front(df).sort_values("total_cost")
    dominated_df = df.drop(pareto_df.index)

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Dominated points (background) ────────────────────────────────────────
    ax.scatter(
        dominated_df["total_cost"],
        dominated_df["total_emissions"],
        color="lightgrey", edgecolors="grey",
        s=55, linewidths=0.5, zorder=2,
        label="Dominated solutions"
    )

    # ── Pareto frontier points ────────────────────────────────────────────────
    ax.scatter(
        pareto_df["total_cost"],
        pareto_df["total_emissions"],
        color="#2980b9", edgecolors="black",
        s=90, linewidths=0.6, zorder=4,
        label="Pareto-optimal solutions"
    )

    # ── Step curve connecting the frontier ───────────────────────────────────
    ax.step(
        pareto_df["total_cost"],
        pareto_df["total_emissions"],
        where="post",
        color="#2980b9", linewidth=1.8,
        linestyle="--", zorder=3
    )

    # ── Labels: (α, β) next to each Pareto point — non-overlapping ───────────
    texts = [
        ax.annotate(
            f"(α={row['alpha']:.2f}, β={row['beta']:.2f})",
            xy=(row["total_cost"], row["total_emissions"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, color="#1a252f"
        )
        for _, row in pareto_df.iterrows()
    ]
    adjust_text(texts, ax=ax)

    # ── Knee point: closest to origin after normalising both axes ─────────────
    c_min, c_max = pareto_df["total_cost"].min(),      pareto_df["total_cost"].max()
    e_min, e_max = pareto_df["total_emissions"].min(), pareto_df["total_emissions"].max()

    costs_norm = (pareto_df["total_cost"]      - c_min) / (c_max - c_min)
    emiss_norm = (pareto_df["total_emissions"] - e_min) / (e_max - e_min)
    knee_idx   = (costs_norm**2 + emiss_norm**2).idxmin()
    knee       = pareto_df.loc[knee_idx]

    ax.scatter(
        knee["total_cost"], knee["total_emissions"],
        marker="*", color="gold", edgecolors="black",
        s=220, zorder=5, label="Knee point"
    )

    # ── Labels & formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Total Cost ($)",                    fontsize=LARGE)
    ax.set_ylabel("Total Emissions (gCO$_2$)",         fontsize=LARGE)
    ax.set_title("Pareto Frontier  —  Cost vs Emissions",
                 fontsize=LARGE, fontweight="bold")
    ax.tick_params(labelsize=MEDIUM)
    ax.legend(fontsize=SMALL)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, save_path)
    plt.show()