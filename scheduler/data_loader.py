#data_loader
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_prices(filename="WEP_Apr-2025.csv"):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df = df.iloc[240:].reset_index(drop=True) # Start from row 242 to skip to Apr 06
    df = df.head(48)

    df["Period"] = range(1, 49)
    df["USEP ($/MWh)"] = df["USEP ($/MWh)"] / 1000 # Convert from $/MWh to $/kWh
    prices = dict(zip(df["Period"], df["USEP ($/MWh)"]))
    
    return prices, list(prices.keys())

def load_evs(filename="ev_info.csv"):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    evs = []

    for _, row in df.iterrows():
        evs.append({
            "name": str(row["EV"]),
            "arrival": int(row["Arrival Time"]),
            "departure": int(row["Departure Time"]),
            "arrival_energy": float(row["Arrival Energy"]),
            "desired_energy": float(row["Desired Energy"]),
            "max_charging_power": 11.0,
            "max_discharging_power": 4.0,
            "battery_capacity": 50.0
        })
    return evs