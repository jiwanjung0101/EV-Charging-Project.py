import os
import pandas as pd

def save_ev_schedule_csv(prices, time_slots, c, d, evs, interval_hours=0.5, save_path="plots/ev_schedule.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data_rows = []

    for ev in evs:
        name = ev["name"]
        for t in time_slots:
            charge = c[(name, t)].value()
            discharge = d[(name, t)].value()
            net_energy = (charge - discharge) * interval_hours
            if charge > 1e-6 or discharge > 1e-6:
                row = {
                    "EV": name,
                    "Time": t,
                    "Charge (kW)": charge,
                    "Discharge (kW)": discharge,
                    "Net Energy (kWh)": net_energy,
                    "Price ($/kWh)": prices[t],
                    "Cost ($)": net_energy * prices[t]
                }
                data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df.to_csv(save_path, index=False)
