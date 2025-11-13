from datetime import datetime
# EV class definition
class EV:
    def __init__(self, arrival_soc, target_soc, battery_capacity, charging_efficiency):
        #EV attributes
        self.arrival_soc = arrival_soc
        self.target_soc = target_soc
        self.battery_capacity = battery_capacity
        self.charging_efficiency = charging_efficiency

    # Calculate energy needed considering efficiency
    def energy_needed(self):
        raw_energy = max(0, (self.target_soc - self.arrival_soc) * self.battery_capacity)
        adjusted_energy = raw_energy / self.charging_efficiency  # adjust for efficiency losses
        return min(adjusted_energy, self.battery_capacity)

# EV charging scheduler function
def ev_scheduler(ev, arrival_time, departure_time):
    #charging station contraints
    max_power = 50 # kW
    min_power = 0

    # Calculate required power
    time_duration = (departure_time - arrival_time).total_seconds()/3600 # in hours
    power_kw = ev.energy_needed() / time_duration # in kW
    
    # Determine charging status
    status = min_power <= power_kw <= max_power
    
    # Calculate total cost
    interval_hour = 0.25
    intervals = int(time_duration / interval_hour)
    price = 0.20 #per kWh
    total_cost = 0

    # Calculate total cost only if charging is feasible
    if status:
        for i in range(intervals):
            total_cost += power_kw * price * interval_hour
        return {
            "status": True,
            "total_cost": total_cost,
            "power_kw": power_kw,
            "duration_hr": time_duration,
            "intervals": intervals
        }
    else:
        return {
            "status": False,
            "total_cost": None,
            "power_kw": power_kw,
            "duration_hr": time_duration,
            "intervals": intervals
        }

# Test the EV charging scheduler
def main():
    ev = EV(0.2, 0.8, 50, 0.9)
    arrival = datetime(2025, 11, 13, 8, 0)
    departure = datetime(2025, 11, 13, 12, 0)

    result = ev_scheduler(ev, arrival, departure)

    print("=== EV Charging Summary ===")
    print(f"Arrival:  {arrival.strftime('%H:%M')} | Departure: {departure.strftime('%H:%M')}")
    if result["status"]:
        print(f"Power: {result['power_kw']:.2f} kW")
        print(f"Total Cost: ${result['total_cost']:.2f}")
    else:
        print("⚠️ Charging not possible (power constraint).")

if __name__ == "__main__":
    main()