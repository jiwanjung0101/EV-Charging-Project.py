class EV:
    def __init__(self, name, arrival, departure, arrival_energy,
                 desired_energy, battery_capacity,
                 max_charging_power, max_discharging_power):
        self.name = name
        self.arrival = arrival
        self.departure = departure
        self.arrival_energy = arrival_energy
        self.desired_energy = desired_energy
        self.battery_capacity = battery_capacity
        self.max_charging_power = max_charging_power
        self.max_discharging_power = max_discharging_power

    def active_slots(self, time_slots):
        """Return only the time slots where this EV is connected."""
        return [t for t in time_slots if self.arrival <= t <= self.departure]

    def is_active(self, t):
        return self.arrival <= t <= self.departure


def load_evs():
    """
    Return the list of EVs for the simulation.
    Edit this list to match your actual fleet.
    """
    return [
        EV("EV1", arrival=8,  departure=36, arrival_energy=10.0,
           desired_energy=40.0, battery_capacity=60.0,
           max_charging_power=7.4, max_discharging_power=3.7),
        EV("EV2", arrival=4,  departure=40, arrival_energy=5.0,
           desired_energy=30.0, battery_capacity=50.0,
           max_charging_power=7.4, max_discharging_power=3.7),
        EV("EV3", arrival=10, departure=44, arrival_energy=8.0,
           desired_energy=35.0, battery_capacity=55.0,
           max_charging_power=7.4, max_discharging_power=3.7),
    ]