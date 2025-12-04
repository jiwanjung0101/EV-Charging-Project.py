# EV Charging Scheduler
import pulp as lp

def run_scheduler(prices, time_slots, evs, interval_hours=0.5):
    # Define the optimization model
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
        prices[t] * (c[(ev["name"], t)] - d[(ev["name"],t)]) * interval_hours  # cost = price * (charging - discharging) * time
        for ev in evs for t in time_slots
    )

    # Constraints
    # Gird power constrains
    for t in time_slots:
        model += lp.lpSum(c[(ev["name"], t)] - d[(ev["name"], t)] for ev in evs) <= 50.0

    # EV constraints
    energy_vars = {}
    for ev in evs:
        # energy dynamics
        energy_times = [t for t in time_slots if ev["arrival"] <= t <= ev["departure"]]
        energy = lp.LpVariable.dicts(
            f"Energy_{ev['name']}",
            [t for t in time_slots if t >= ev["arrival"] and t <= ev["departure"]],
            lowBound=0,
            upBound=ev["battery_capacity"],
            cat="Continuous"
        )
        energy_vars[ev["name"]] = energy

        # initial Energy at arrival
        model += energy[ev["arrival"]] == ev["arrival_energy"]

        # energy update
        for i in range(1, len(energy_times)):
            t = energy_times[i]
            prev_t = energy_times[i-1]
            model += energy[t] == energy[prev_t] + (c[(ev["name"], t)] - d[(ev["name"], t)]) * interval_hours
       
        # required energy at departure
        model += energy[ev["departure"]] >= ev["desired_energy"]

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

    return c, d, energy_vars