import pandas as pd
import pulp as lp
import matplotlib.pyplot as plt

'''def load_prices(path="eprice.csv"):
    df = pd.read_csv(path).head(48)
    df["Time"] = range(1, 49)
    prices = dict(zip(df["Time"], df["Price"]))
    return prices, list(prices.keys())'''

def load_prices(path="WEP_Apr-2025.csv"):
    df = pd.read_csv(path)
    df = df.iloc[240:].reset_index(drop=True) # Start from row 242 to skip to Apr 06
    df = df.head(48)

    df["Period"] = range(1, 49)
    df["USEP ($/MWh)"] = df["USEP ($/MWh)"] / 1000 # Convert from $/MWh to $/kWh
    prices = dict(zip(df["Period"], df["USEP ($/MWh)"]))
    
    return prices, list(prices.keys())

def load_evs(path="ev_info.csv"):
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
            "battery_capacity": 55.0
        })
    return evs

def ev_scheduler():
    prices, time_slots = load_prices()
    evs = load_evs()

    interval_hours = 0.5  # 30-minute 
    model = lp.LpProblem("EV_Scheduler", lp.LpMinimize)

    # Decision variables: charging and discharging
    c = lp.LpVariable.dicts(
        "charge",
        ((ev["name"], t) for ev in evs for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )
    d = lp.LpVariable.dicts(
        "discharge",
        ((ev["name"], t) for ev in evs for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )

    # Objective: minimize cost over all EVs and times
    model += lp.lpSum(
        prices[t] * (c[(ev["name"], t)] - (d[(ev["name"],t)]) * interval_hours)  # cost = price * (charging - discharging) * time
        for ev in evs for t in time_slots
    )

    # Constraints
    for ev in evs:
        # Required energy
        required_energy = ev["desired_energy"] - ev["arrival_energy"]
        model += lp.lpSum(
            (c[(ev["name"], t)] - d[(ev["name"], t)]) * interval_hours
            for t in time_slots
            if ev["arrival"] <= t <= ev["departure"]
        )>= required_energy #charged energy - discharged energy >= required energy

        model += lp.lpSum(
            (c[(ev["name"], t)] - d[(ev["name"], t)]) * interval_hours
            for t in time_slots
            if ev["arrival"] <= t <= ev["departure"]
        ) <= ev["battery_capacity"]  # battery capacity constraint

        # Only charge inside availability window
        for t in time_slots:
            if not (ev["arrival"] <= t <= ev["departure"]):
                model += c[(ev["name"], t)] == 0
                model += d[(ev["name"], t)] == 0

                
        # Power limit
        for t in time_slots:
            model += c[(ev["name"], t)] <= ev["max_charging_power"]
            model += d[(ev["name"], t)] <= ev["max_discharging_power"]

    # Solve
    model.solve(lp.PULP_CBC_CMD(msg=0))

    #Preparting for plotting
    price_list = []
    time_list = list(time_slots)
    ev_charge_profiles = {ev["name"]: [] for ev in evs}
    ev_discharge_profiles = {ev["name"]: [] for ev in evs}

    # Fill prices
    for t in time_slots:
        price_list.append(prices[t])

    # Fill charging and discharging data
    for ev in evs:
        for t in time_slots:
            ev_charge_profiles[ev["name"]].append(c[(ev["name"], t)].value())
            ev_discharge_profiles[ev["name"]].append(d[(ev["name"], t)].value())

    total_charge = []
    total_discharge = []

    for i in range(len(time_list)):
        total_charge.append(sum(ev_charge_profiles[ev["name"]][i] for ev in evs))
        total_discharge.append(sum(ev_discharge_profiles[ev["name"]][i] for ev in evs))

    #Results
    '''print("\n=== EV Scheduling Results ===\n")
    total_cost = 0

    for ev in evs:
        name = ev["name"]
        print(f"\nEV: {name}")
        print(f"Arrival Time: {ev['arrival']}")
        print(f"Departure Time: {ev['departure']}")

        ev_energy = 0
        ev_cost = 0

        print("Time | Charge (kW) | Discharge (kW) | Net Energy (kWh)")
        print("--------------------------------------------------------")

        for t in time_slots:
            charge = c[(name, t)].value()
            discharge = d[(name, t)].value()

            if charge > 1e-6 or discharge > 1e-6:  # print only active slots
                net = (charge - discharge) * interval_hours
                ev_energy += net
                ev_cost += prices[t] * (charge - discharge) * interval_hours

                print(f"{t:4d} | {charge:12.3f} | {discharge:14.3f} | {net:14.3f}")

        print(f"\nTotal Energy Delivered: {ev_energy:.3f} kWh")
        print(f"Cost for {name}: ${ev_cost:.4f}")

        total_cost += ev_cost

    print("\n=== Total Cost for All EVs ===")
    print(f"Total Cost: ${total_cost:.4f}")'''
    
    #Plotting Time vs Price
    plt.figure(figsize=(10, 4))
    plt.plot(time_list, price_list, linewidth=2)
    plt.xlabel("Time (30 min intervals)")
    plt.ylabel("Price ($ per kWh)")
    plt.title("Electricity Price Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Plotting Charging and Discharging Profiles
    plt.figure(figsize=(10, 4))

    plt.plot(time_list, total_charge, label="Total Charging (kW)", linewidth=2)
    plt.plot(time_list, total_discharge, label="Total Discharging (kW)", linewidth=2)

    plt.xlabel("Time (30 min intervals)")
    plt.ylabel("Power (kW)")
    plt.title("Total Charging and Discharging Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ev_scheduler()