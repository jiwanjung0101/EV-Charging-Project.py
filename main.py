from scheduler.data_loader import load_prices, load_evs, load_carbon_intensity
from scheduler.model import run_scheduler
from scheduler.plot import plot_prices, plot_power, plot_energy, plot_power_price, plot_carbon_intensity
from scheduler.ev_schedule import ev_schedule

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

def compute_total_emissions(carbon, time_slots, c, d, evs, interval_hours=0.5):
    total_emissions = 0.0
    ev_emissions = {}

    for ev in evs:
        name = ev.name
        ev_emission = 0.0

        for t in ev.active_slots(time_slots):
            charge = c[(name, t)].value()
            emission = charge * interval_hours * carbon[t]
            ev_emission += emission

        ev_emissions[name] = ev_emission
        total_emissions += ev_emission

    return total_emissions, ev_emissions


def main():
    prices, time_slots = load_prices()
    evs = load_evs()
    carbon, _ = load_carbon_intensity()
    c, d, energy_vars = run_scheduler(
        prices,
        carbon,
        time_slots,
        evs,
        interval_hours=0.5
    )

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
    carbon_list = [carbon[t] for t in time_slots]
    plot_carbon_intensity(time_slots, carbon_list)
    plot_prices(time_slots, price_list)
    plot_power(time_slots, total_charge, total_discharge)
    plot_power_price(time_slots, total_charge, total_discharge, price_list)

    # Plot energy profile for first EV
    first_ev = evs[0]
    energy_dict = energy_vars[first_ev.name]
    plot_energy(time_slots, energy_dict, first_ev.name)

    # Save results
    ev_schedule(prices, carbon, time_slots, c, d, evs)

    total_emissions, ev_emissions = compute_total_emissions(
        carbon, time_slots, c, d, evs
    )

    print("Total emissions (kgCO2):", total_emissions)



if __name__ == "__main__":
    main()

    

