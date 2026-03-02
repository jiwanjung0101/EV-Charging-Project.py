from scheduler.data_loader import load_prices, load_evs
from scheduler.model import run_scheduler
from scheduler.plot import plot_prices, plot_power, plot_energy, plot_power_price
from scheduler.ev_schedule import ev_schedule
from scipy import stats
import numpy as np

def compute_total_cost(prices, time_slots, c, d, evs, interval_hours=0.5):
    total_energy_kWh = 0.0
    total_cost = 0.0
    ev_costs = {}

    for ev in evs:
        name = ev.name
        ev_energy = 0.0
        ev_cost = 0.0

        for t in ev.active_slots(time_slots):
            charge = c[(name, t)].value()
            discharge = d[(name, t)].value()
            net_energy = (charge - discharge) * interval_hours
            cost = net_energy * prices[t]

            ev_energy += net_energy
            ev_cost += cost

        ev_costs[name] = ev_cost
        total_energy_kWh += ev_energy
        total_cost += ev_cost

    return total_energy_kWh, total_cost, ev_costs


def compute_naive_cost_per_ev(prices, time_slots, evs, interval_hours=0.5):
    total_cost = 0.0
    total_energy = 0.0
    ev_costs = {}

    for ev in evs:
        active = ev.active_slots(time_slots)
        required_energy = ev.desired_energy - ev.arrival_energy
        energy_per_slot = required_energy / len(active) if len(active) > 0 else 0.0

        ev_cost = 0.0
        for t in active:
            total_energy += energy_per_slot
            cost = energy_per_slot * prices[t]
            total_cost += cost
            ev_cost += cost
        
        ev_costs[ev.name] = ev_cost

    return total_energy, total_cost, ev_costs


def print_cost_comparison(prices, time_slots, c, d, evs, interval_hours=0.5):
    # Optimized costs
    _, _, optimized_ev_costs = compute_total_cost(prices, time_slots, c, d, evs, interval_hours)

    # Naive costs
    _, _, naive_ev_costs = compute_naive_cost_per_ev(prices, time_slots, evs, interval_hours)

    # Print header
    print(f"{'EV':<10} {'Naive Cost ($)':>15} {'Optimized Cost ($)':>20} {'Savings ($)':>15}")
    print("-" * 60)

    # Print each EV
    for ev in evs:
        name = ev.name
        naive = naive_ev_costs.get(name, 0.0)
        optimized = optimized_ev_costs.get(name, 0.0)
        savings = naive - optimized
        print(f"{name:<10} {naive:>15.2f} {optimized:>20.2f} {savings:>15.2f}")

    # Totals
    total_naive = sum(naive_ev_costs.values())
    total_optimized = sum(optimized_ev_costs.values())
    total_savings = total_naive - total_optimized
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_naive:>15.2f} {total_optimized:>20.2f} {total_savings:>15.2f}")



def cost_t_test(prices, time_slots, c, d, evs, interval_hours=0.5, alpha=0.05):
    # Get costs
    _, _, optimized_ev_costs = compute_total_cost(
        prices, time_slots, c, d, evs, interval_hours
    )
    _, _, naive_ev_costs = compute_naive_cost_per_ev(
        prices, time_slots, evs, interval_hours
    )

    # Align data per EV
    naive = []
    optimized = []

    for ev in evs:
        name = ev.name
        naive.append(naive_ev_costs.get(name, 0.0))
        optimized.append(optimized_ev_costs.get(name, 0.0))

    naive = np.array(naive)
    optimized = np.array(optimized)

    naive = np.round(naive, 2)
    optimized = np.round(optimized, 2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(naive, optimized)

    print("\nPaired t-test: Naive vs Optimized Costs")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < alpha:
        print("Result: Statistically significant cost reduction")
    else:
        print("Result: No statistically significant difference")

    return t_stat, p_value



def main():
    prices, time_slots = load_prices()
    evs = load_evs()

    c, d, energy_vars = run_scheduler(prices, time_slots, evs)

    # Prepare plot data
    price_list = [prices[t] for t in time_slots]

    # Total charge and discharge now use ev.name
    total_charge = [
        sum(c[(ev.name, t)].value() for ev in evs)
        for t in time_slots
    ]

    # Discharge is negative direction
    total_discharge = [
        sum(-d[(ev.name, t)].value() for ev in evs)
        for t in time_slots
    ]

    # Plot
    plot_prices(time_slots, price_list)
    plot_power(time_slots, total_charge, total_discharge)
    plot_power_price(time_slots, total_charge, total_discharge, price_list)

    # Plot energy profile for first EV
    first_ev = evs[0]
    energy_dict = energy_vars[first_ev.name]
    plot_energy(time_slots, energy_dict, first_ev.name)

    
    sixth_ev = evs[5]
    energy_dict = energy_vars[sixth_ev.name]
    plot_energy(time_slots, energy_dict, sixth_ev.name)
    # Save results
    ev_schedule(prices, time_slots, c, d, evs)

    print_cost_comparison(prices, time_slots, c, d, evs)
    cost_t_test(prices, time_slots, c, d, evs)




if __name__ == "__main__":
    main()

    

