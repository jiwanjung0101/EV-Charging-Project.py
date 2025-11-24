import pandas as pd
import pulp as lp

def load_prices(path="eprice.csv"):
    df = pd.read_csv(path).head(48)
    df["Time"] = range(1, 49)
    prices = dict(zip(df["Time"], df["Price"]))
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
            "max_power": 7.0   # kW power limit
        })

    return evs

def ev_scheduler():
    prices, time_slots = load_prices()
    evs = load_evs()
    discharging_price = 0.05  # $0.05 per kWh for discharging

    interval_hours = 0.5  # 30-minute interval
    model = lp.LpProblem("SimpleChargingOnly", lp.LpMinimize)

    # Decision variables: c[ev, t]
    c = lp.LpVariable.dicts(
        "c",
        ((ev["name"], t) for ev in evs for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )
    d = lp.LpVariable.dicts(
        "d",
        ((ev["name"], t) for ev in evs for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )

    # Objective: minimize cost over all EVs and times
    model += lp.lpSum(
        prices[t] * (c[(ev["name"], t)] * interval_hours) -  
        discharging_price*(d[(ev["name"],t)] * interval_hours)  # cost = price * energy
        for ev in evs for t in time_slots
    )

    # Constraints
    for ev in evs:
        # Required energy
        required_energy = ev["desired_energy"] - ev["arrival_energy"]
        model += lp.lpSum(
            c[(ev["name"], t)] * interval_hours
            for t in time_slots
            if ev["arrival"] <= t <= ev["departure"]
        ) - lp.lpSum(
            d[(ev["name"], t)] * interval_hours
            for t in time_slots
            if ev["arrival"] <= t <= ev["departure"]
        ) >= required_energy

        # Only charge inside availability window
        for t in time_slots:
            if not (ev["arrival"] <= t <= ev["departure"]):
                model += c[(ev["name"], t)] == 0
                model += d[(ev["name"], t)] == 0

                
        # Power limit
        for t in time_slots:
            model += c[(ev["name"], t)] <= ev["max_power"]
            model += d[(ev["name"], t)] <= ev["max_power"]

    # Solve
    model.solve(lp.PULP_CBC_CMD(msg=0))

    print("Status:", lp.LpStatus[model.status])
    print("Total Cost:", lp.value(model.objective))

    # Print schedule
    for ev in evs:
        print(f"\nEV {ev['name']} schedule:")
        for t in time_slots:
            val = c[(ev["name"], t)].value()
            if val > 1e-4:
                print(f"  Time {t}: charge {val:.2f} kW")


if __name__ == "__main__":
    ev_scheduler()