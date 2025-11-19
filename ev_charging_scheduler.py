from datetime import datetime, timedelta
import pulp as lp

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

def power(price_now, base_price, Erem, Tleft, max_power):
    # minimum power required to finish on time
    Pmin_needed = Erem / Tleft

    # sensitivity coeffecient
    k = 20.0

    # power calculation
    P = Pmin_needed + k * (base_price - price_now)

    # physical constraints
    if P < Pmin_needed:
        P = Pmin_needed
    if P > max_power:
        P = max_power

    return P

# EV charging scheduler function
def ev_scheduler(ev, arrival_time, departure_time, price):
    #charging station contraints
    max_power = 50 # kW
    dt = 0.25 # hours
    time_duration = (departure_time - arrival_time).total_seconds()/3600 # in hours
    Erem = ev.energy_needed()
    intervals = int(time_duration / dt)
    base_price = 0.30  # base price in $/kWh
    
    # Initialize variables
    total_cost = 0.0
    power_values = []
    cumulative_energy = 0.0

    # Iterate over each time interval
    for i in range(intervals):
        power_kw = power(price[i], base_price, Erem, time_duration - i * dt, max_power)
        
        #
        energy_this_interval = power_kw * dt
        if cumulative_energy + energy_this_interval > Erem:
            energy_this_interval = Erem - cumulative_energy
            power_kw = energy_this_interval / dt

        total_cost += power_kw * price[i] * dt
        power_values.append(power_kw)
        cumulative_energy += energy_this_interval 

        print(f"Interval {i + 1}: Power = {power_kw:.2f} kW, Cumulative Energy = {cumulative_energy:.2f} kWh")

        if cumulative_energy >= Erem:
            finish_time = arrival_time + timedelta(hours=i * dt)
            break

    return {
        "status": True,
        "total_cost": total_cost,
        "finish_time": finish_time,
        "duration_hr": time_duration,
        "intervals": intervals,
        "power_values": power_values
    }

# Test the EV charging scheduler
def main():
    ev = EV(0.2, 0.8, 50, 0.9)
    arrival = datetime(2025, 11, 13, 8, 0)
    departure = datetime(2025, 11, 13, 12, 0)
    price = [0.25, 0.20, 0.15, 0.10, 0.12, 0.18, 0.22, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14]  # $/kWh for each 15-min interval

    result = ev_scheduler(ev, arrival, departure, price)

    print("=== EV Charging Summary ===")
    print(f"Arrival:  {arrival.strftime('%H:%M')} | Departure: {departure.strftime('%H:%M')}")
    if result["status"]:
        print(f"Total Cost: ${result['total_cost']:.2f}")
        total_energy_charged = sum(result['power_values']) * 0.25  # Multiply by interval duration (0.25 hours)
        desired_energy = ev.energy_needed()
        print(f"Total Energy Charged: {total_energy_charged:.2f} kWh")
        print(f"Desired Energy: {desired_energy:.2f} kWh")
        if abs(total_energy_charged - desired_energy) < 0.01:  # Allow a small margin of error
            print("✅ Charging met the desired energy.")
        else:
            print("⚠️ Charging did not meet the desired energy.")
        print(f"Finish Time: {result['finish_time'].strftime('%H:%M')}")
    else:
        print("⚠️ Charging not possible (power constraint).")

    for i, power_kw in enumerate(result['power_values']):
        print(f"Interval {i + 1}: Power = {power_kw:.2f} kW, Price = ${price[i]:.2f}")

if __name__ == "__main__":
    main()