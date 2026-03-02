import matplotlib.pyplot as plt
import os

# Set global font settings
plt.rcParams["font.family"] = "Times New Roman"
large = 20
medium = 18

# Plotting electricity prices over time
def plot_prices(time_list, price_list, save_path="plots/prices.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(time_list, price_list, color='black', linewidth=2)
    plt.xlabel("Time", fontsize=large)
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

    plt.xlabel("Time", fontsize=large)
    plt.ylabel("Power (kW)", fontsize=large)
    plt.xticks(fontsize=medium)
    plt.yticks(fontsize=medium)
    plt.legend(fontsize=14)
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

    plt.figure(figsize=(10, 6))
    plt.plot(time_list, ev_energy_full, label=f"Energy Level of {ev_name} (kWh)", color='orange', linewidth=2)
    plt.xlabel("Time", fontsize=large)
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

def plot_power_price(time_list, total_charge, total_discharge, price_list,
                     save_path="plots/power_price.png"):

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: power (kW)
    ax1.plot(time_list, total_charge, label="Total Charging (kW)", linewidth=2)
    ax1.plot(time_list, total_discharge, label="Total Discharging (kW)", linewidth=2)
    ax1.set_xlabel("Time", fontsize=large)
    ax1.set_ylabel("Power (kW)", fontsize=large)
    ax1.tick_params(axis='y', labelsize=medium)
    ax1.tick_params(axis='x', labelsize=medium)
    ax1.grid(True)

    # Right axis: price ($/kWh)
    ax2 = ax1.twinx()
    ax2.plot(time_list, price_list, color="black", linewidth=2, linestyle="--",
             label="Price ($/kWh)")
    ax2.set_ylabel("Price ($/kWh)", fontsize=large)
    ax2.tick_params(axis='y', labelsize=medium)

    # Combined legend
    lines_left = ax1.get_lines()
    lines_right = ax2.get_lines()
    labels = [l.get_label() for l in lines_left + lines_right]
    fig.legend(lines_left + lines_right, labels, fontsize=10, loc="upper left")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()