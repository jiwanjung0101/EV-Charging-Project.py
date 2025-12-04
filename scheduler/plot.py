import matplotlib.pyplot as plt
import os

# Set global font settings
plt.rcParams["font.family"] = "Times New Roman"
large = 20
medium = 18

# Plotting electricity prices over time
def plot_prices(time_list, price_list, save_path="plots/prices.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(time_list, price_list, linewidth=2)
    plt.xlabel("Time (30 min intervals)", fontsize=large)
    plt.ylabel("Price ($ per kWh)", fontsize=large)
    plt.xticks(fontsize=medium)
    plt.yticks(fontsize=medium)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

# Plotting total charging and discharging power over time
def plot_power(time_list, total_charge, total_discharge, save_path="plots/power.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(time_list, total_charge, label="Total Charging (kW)", linewidth=2)
    plt.plot(time_list, total_discharge, label="Total Discharging (kW)", linewidth=2)

    plt.xlabel("Time (30 min intervals)", fontsize=large)
    plt.ylabel("Power (kW)", fontsize=large)
    plt.xticks(fontsize=medium)
    plt.yticks(fontsize=medium)
    plt.legend(fontsize=medium)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

# Plotting energy profile for a single EV over time
def plot_energy(time_list, ev_energy_dict, ev_name, save_path="plots/energy.png"):

    # Convert dict to full list with None for missing times
    ev_energy_full = []
    for t in time_list:
        if t in ev_energy_dict:
            value = ev_energy_dict[t].value()
        else:
            value = None   # EV not connected at this time
        ev_energy_full.append(value)

    plt.figure(figsize=(10, 4))
    plt.plot(time_list, ev_energy_full, label=f"Energy Level of {ev_name} (kWh)", color='orange', linewidth=2)
    plt.xlabel("Time (30 min intervals)", fontsize=large)
    plt.ylabel("Energy (kWh)", fontsize=large)
    plt.xticks(fontsize=medium)
    plt.yticks(fontsize=medium)
    plt.legend(fontsize=medium)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
