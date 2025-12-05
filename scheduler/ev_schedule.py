# Save EV schedule data
import os
import pandas as pd
from scheduler.data_loader import load_prices, load_evs
from scheduler.model import run_scheduler
from scheduler.plot import plot_prices, plot_power, plot_energy

def ev_schedule(prices, time_slots, c, d, evs, interval_hours=0.5, save_path="data/ev_schedule.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_rows = []

    for ev in evs:
        name = ev.name
        for t in ev.active_slots(time_slots):
            charge = c[(name, t)].value()
            discharge = d[(name, t)].value()
            if abs(charge) < 1e-6 and abs(discharge) < 1e-6:
                continue
            net_energy = (charge - discharge) * interval_hours
            cost = net_energy * prices[t]
            data_rows.append({
                "EV": name,
                "Time": t,
                "Charge (kW)": charge,
                "Discharge (kW)": discharge,
                "Net Energy (kWh)": net_energy,
                "Price ($/kWh)": prices[t],
                "Cost ($)": cost,
            })
    df = pd.DataFrame(data_rows)
    df.to_csv(save_path, index=False)
