from scheduler.data_loader import load_prices, load_evs
from scheduler.model import run_scheduler
from scheduler.plot import plot_prices, plot_power, plot_energy
from scheduler.ev_schedule import ev_schedule

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

    # Plot energy profile for first EV
    first_ev = evs[0]
    energy_dict = energy_vars[first_ev.name]
    plot_energy(time_slots, energy_dict, first_ev.name)

    # Save results
    ev_schedule(prices, time_slots, c, d, evs)


if __name__ == "__main__":
    main()

    

