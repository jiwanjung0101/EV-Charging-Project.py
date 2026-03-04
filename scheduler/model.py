# EV Charging Scheduler
import pulp as lp
from scheduler.ev import EV

def run_scheduler(prices, carbon, time_slots, evs, interval_hours=0.5, carbon_price=0.05):
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
    #potential V2V variables for future extension
    """v = lp.LpVariable.dicts(
        "v2v",
        ((ev_i.name, ev_j.name, t)
        for ev_i in evs
        for ev_j in evs
        if ev_i != ev_j
        for t in time_slots),
        lowBound=0,
        cat="Continuous"
    )"""

    # Objective: minimize cost over all EVs and times, added carbon cost as a tax
    model += lp.lpSum(
        ((prices[t] + carbon_price * carbon[t]) * c[(ev.name, t)] - prices[t] * d[(ev.name,t)]) * interval_hours  
        for ev in evs for t in time_slots
    )

    #Alternative objective with separate weighting for price and carbon
    '''model += lp.lpSum(
        (
            alpha * prices[t] * (c[(ev.name, t)] - d[(ev.name, t)]) +
            beta  * carbon[t] * c[(ev.name, t)]
        ) * interval_hours
        for ev in evs for t in time_slots
    )'''

    '''model += lp.lpSum(
        carbon[t] * c[(ev.name, t)] * interval_hours
        for ev in evs for t in time_slots
    ) <= carbon_cap'''

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
        
        """
        for prev_t, t in zip(active[:-1], active[1:]):
            model += energy[t] == (
                energy[prev_t]
                + c[(ev.name, t)] * interval_hours
                - d[(ev.name, t)] * interval_hours
                - lp.lpSum(
                    v[(ev.name, other.name, t)]
                    for other in evs if other.name != ev.name
                ) * interval_hours
                + lp.lpSum(
                    v[(other.name, ev.name, t)]
                    for other in evs if other.name != ev.name
                ) * interval_hours
            )"""

    # Solve
    model.solve(lp.PULP_CBC_CMD(msg=0))

    return c, d, energy_vars