import numpy as np
import pandas as pd

from scheduler.data_loader import load_prices, load_carbon_intensity
from scheduler.ev          import load_evs
from scheduler.model       import run_scheduler, compute_metrics
from scheduler.plot        import (
    plot_prices,
    plot_carbon_intensity,
    plot_power,
    plot_power_price,
    plot_energy,
    plot_schedule,
    plot_pareto,
    plot_heatmaps,
    plot_pareto_curve,
)


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    prices,     time_slots = load_prices()
    carbon,     _          = load_carbon_intensity()
    evs                    = load_evs()

    # ── Plot raw inputs ───────────────────────────────────────────────────────
    price_list  = [prices[t] for t in time_slots]
    carbon_list = [carbon[t] for t in time_slots]

    plot_prices(time_slots, price_list)
    plot_carbon_intensity(time_slots, carbon_list)

    # ── Alpha / Beta grid ─────────────────────────────────────────────────────
    # Full grid: every combination of the values below (25 runs).
    # For a Pareto sweep along α + β = 1, uncomment the two lines beneath.
    grid_values  = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_values = grid_values
    beta_values  = grid_values

    # lambdas      = np.linspace(0, 1, 11)
    # alpha_values = list(lambdas)
    # beta_values  = list(1 - lambdas)

    results        = []
    balanced_run   = None   # cache the α=β=0.5 solution for the schedule plot

    print("\n===== Alpha–Beta Grid Search =====\n")

    for alpha in alpha_values:
        for beta in beta_values:
            c, d, energy_vars, status = run_scheduler(
                prices, carbon, time_slots, evs,
                interval_hours=0.5,
                alpha=alpha,
                beta=beta,
                carbon_cap_fraction=0.85,
            )

            if status != "Optimal":
                print(f"  [SKIP] α={alpha:.2f}  β={beta:.2f}  → {status}")
                continue

            cost, emissions, energy, avg_intensity = compute_metrics(
                prices, carbon, time_slots, c, d, evs
            )

            results.append({
                "alpha":              alpha,
                "beta":               beta,
                "total_energy_kWh":   round(energy,        4),
                "total_cost":         round(cost,          4),
                "total_emissions":    round(emissions,     2),
                "avg_emission_per_kwh": round(avg_intensity, 4),
                "status":             status,
            })

            print(
                f"  α={alpha:.2f}  β={beta:.2f} | "
                f"Cost={cost:8.2f} $   "
                f"Emissions={emissions:10.1f} gCO₂   "
                f"Energy={energy:.2f} kWh   [{status}]"
            )

            if abs(alpha - 0.5) < 1e-6 and abs(beta - 0.5) < 1e-6:
                balanced_run = (c, d, energy_vars, alpha, beta)

    # ── Save results CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv("alpha_beta_results.csv", index=False)
    print("\nResults saved → alpha_beta_results.csv")

    # ── Analysis plots ────────────────────────────────────────────────────────
    if df.empty:
        print("No optimal solutions found — skipping plots.")
        return

    print("\nGenerating analysis plots …")
    plot_pareto(df)
    plot_pareto_curve(df)
    plot_heatmaps(df)

    # Schedule plot for the balanced run (or the last solved run as fallback)
    if balanced_run is not None:
        c_s, d_s, ev_s, a_s, b_s = balanced_run
    else:
        c_s, d_s, ev_s, a_s, b_s = c, d, energy_vars, alpha, beta

    plot_schedule(prices, carbon, time_slots, c_s, d_s, evs,
                  alpha=a_s, beta=b_s)

    # Optional: per-EV energy profiles for the balanced run
    for ev in evs:
        plot_energy(time_slots, ev_s[ev.name], ev.name,
                    save_path=f"plots/energy_{ev.name}.png")

    # Optional: aggregate power + price overlay
    total_charge    = [sum((c_s[(ev.name, t)].varValue or 0) for ev in evs)
                       for t in time_slots]
    total_discharge = [sum((d_s[(ev.name, t)].varValue or 0) for ev in evs)
                       for t in time_slots]

    plot_power(time_slots, total_charge, total_discharge)
    plot_power_price(time_slots, total_charge, total_discharge, price_list)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n===== Summary =====")
    print(df.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()