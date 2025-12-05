#EV class
class EV:
    def __init__(self, name, arrival, departure, arrival_energy, desired_energy,
                 max_charging_power=11.0, max_discharging_power=4.0, battery_capacity=50.0):
        
        self.name = name
        self.arrival = arrival
        self.departure = departure
        self.arrival_energy = arrival_energy
        self.desired_energy = desired_energy
        self.max_charging_power = max_charging_power
        self.max_discharging_power = max_discharging_power
        self.battery_capacity = battery_capacity

    def active_slots(self, time_slots):
        return [t for t in time_slots if self.arrival <= t <= self.departure]
    
    def is_active(self, t):
        return self.arrival <= t <= self.departure
