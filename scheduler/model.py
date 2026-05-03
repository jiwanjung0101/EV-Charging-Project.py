import pulp as lp


def run_scheduler(prices, carbon, time_slots, evs,
                  interval_hours=0.5, alpha=1.0, beta=1.0,
                  carbon_cap_fraction=0.85):
    
    model = lp.LpProblem("EV_Scheduler", lp.LpMinimize)

    # ── Decision variables ────────────────────────────────────────────────────
    c = lp.LpVariable.dicts(
        "charge",
        ((ev.name, t) for ev in evs for t in time_slots),
        lowBound=0, cat="Continuous"
    )
    d = lp.LpVariable.dicts(
        "discharge",
        ((ev.name, t) for ev in evs for t in time_slots),
        lowBound=0, cat="Continuous"
    )

    # ── Normalise price and carbon to [0, 1] ─────────────────────────────────
    p_min, p_max = min(prices.values()), max(prices.values())
    c_min, c_max = min(carbon.values()), max(carbon.values())

    price_norm  = {t: (prices[t] - p_min) / (p_max - p_min) for t in time_slots}
    carbon_norm = {t: (carbon[t] - c_min) / (c_max - c_min) for t in time_slots}

    # ── Objective: minimise weighted cost + carbon ────────────────────────────
    model += lp.lpSum(
        (alpha * price_norm[t] + beta * carbon_norm[t])
        * (c[(ev.name, t)] - d[(ev.name, t)])
        * interval_hours
        for ev in evs
        for t in time_slots
    )

    # ── Grid power limit ──────────────────────────────────────────────────────
    for t in time_slots:
        model += lp.lpSum(
            c[(ev.name, t)] - d[(ev.name, t)] for ev in evs
        ) <= 50.0

    # ── Global carbon cap ─────────────────────────────────────────────────────
    # Baseline = every EV charges at max power for all its active slots.
    # This gives a true worst-case emissions upper bound.
    if carbon_cap_fraction is not None:
        baseline_emissions = sum(
            carbon[t] * ev.max_charging_power * interval_hours
            for ev in evs
            for t in ev.active_slots(time_slots)
        )
        model += lp.lpSum(
            (c[(ev.name, t)] - d[(ev.name, t)]) * carbon[t] * interval_hours
            for ev in evs
            for t in time_slots
        ) <= carbon_cap_fraction * baseline_emissions

    # ── Per-EV battery dynamics ───────────────────────────────────────────────
    energy_vars = {}
    for ev in evs:
        active = ev.active_slots(time_slots)

        energy = lp.LpVariable.dicts(
            f"Energy_{ev.name}", active,
            lowBound=0, upBound=ev.battery_capacity
        )
        energy_vars[ev.name] = energy

        # Initial state of charge
        model += energy[ev.arrival] == ev.arrival_energy

        # SOC evolution
        for prev_t, t in zip(active[:-1], active[1:]):
            model += energy[t] == (
                energy[prev_t]
                + (c[(ev.name, t)] - d[(ev.name, t)]) * interval_hours
            )

        # Must meet desired energy at departure
        model += energy[ev.departure] >= ev.desired_energy

        # Power limits: zero outside connection window, bounded inside
        for t in time_slots:
            if ev.is_active(t):
                model += c[(ev.name, t)] <= ev.max_charging_power
                model += d[(ev.name, t)] <= ev.max_discharging_power
            else:
                model += c[(ev.name, t)] == 0
                model += d[(ev.name, t)] == 0

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.solve(lp.PULP_CBC_CMD(msg=0))
    return c, d, energy_vars, lp.LpStatus[model.status]


def compute_metrics(prices, carbon, time_slots, c, d, evs, interval_hours=0.5):
    total_cost = total_emissions = total_energy = 0.0

    for ev in evs:
        for t in ev.active_slots(time_slots):
            cv  = c[(ev.name, t)].varValue or 0.0
            dv  = d[(ev.name, t)].varValue or 0.0
            net = (cv - dv) * interval_hours

            total_cost      += net * prices[t]
            total_emissions += net * carbon[t]
            total_energy    += net

    avg_intensity = total_emissions / total_energy if total_energy > 0 else 0.0
    return total_cost, total_emissions, total_energy, avg_intensity