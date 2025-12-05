# EV Charging Scheduler
import pulp as lp
from scheduler.ev import EV

def run_scheduler(prices, time_slots, evs, interval_hours=0.5):
    # Define the optimization model
    model = lp.LpProblem("EV_Scheduler", lp.LpMinimize)

    # Decision variables: charging and discharging
    c = lp.LpVariable.dicts(
        "charge",
        ((ev.name, t) for ev in evs for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )
    d = lp.LpVariable.dicts(
        "discharge",
        ((ev.name, t) for ev in evs for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )

    # Objective: minimize cost over all EVs and times
    model += lp.lpSum(
        prices[t] * (c[(ev.name, t)] - d[(ev.name,t)]) * interval_hours  # cost = price * (charging - discharging) * time
        for ev in evs for t in time_slots
    )

    # Constraints
    # Gird power constrains
    for t in time_slots:
        model += lp.lpSum(c[(ev.name, t)] - d[(ev.name, t)] for ev in evs) <= 50.0

    # EV constraints
    energy_vars = {}
    for ev in evs:
        #only consider active slots
        active = ev.active_slots(time_slots)
        
        # Energy variable
        energy = lp.LpVariable.dicts(
            f"Energy_{ev.name}",
            active,
            lowBound=0,
            upBound=ev.battery_capacity
        )
        energy_vars[ev.name] = energy

        # Energy = arrival energy
        model += energy[ev.arrival] == ev.arrival_energy

        # Energy charge and discharge
        for prev_t, t in zip(active[:-1], active[1:]):
            model += energy[t] == energy[prev_t] + (c[(ev.name, t)] - d[(ev.name, t)]) * interval_hours
       
        # energy departue >= desired energy
        model += energy[ev.departure] >= ev.desired_energy

        # No power outside arrival-departure with max limits
        for t in time_slots:
            if ev.is_active(t):
                model += c[(ev.name, t)] <= ev.max_charging_power
                model += d[(ev.name, t)] <= ev.max_discharging_power
            else:
                model += c[(ev.name, t)] == 0
                model += d[(ev.name, t)] == 0

    # Solve
    model.solve(lp.PULP_CBC_CMD(msg=0))

    return c, d, energy_vars