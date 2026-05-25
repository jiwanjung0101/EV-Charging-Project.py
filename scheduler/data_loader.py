"""
scheduler/data_loader.py

Loads nodal LMP prices (CAISO), carbon intensity, and EV specs.
EV is a plain dataclass — no methods — active_slots/is_active live in model.py.

Grid positions
──────────────
ChargingNode  : hardcoded (x, y) positions for the 5 charging nodes on the
                10×10 grid.  Edit CHARGING_NODES below to move them.
EV.grid_x/y   : loaded from the "X" and "Y" columns in ev_info.csv.
                 These are fixed values written into the CSV — not random.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── Charging node grid positions (hardcoded) ──────────────────────────────────

@dataclass
class ChargingNode:
    """
    Maps a CAISO node name to a fixed (x, y) position on the 10×10 grid.
    Edit CHARGING_NODES to change positions — everything else reads from there.
    """
    name:   str    # must match a column in the nodal price DataFrame
    grid_x: int    # 1–10
    grid_y: int    # 1–10


# ── Edit these positions to move nodes on the grid ────────────────────────────
CHARGING_NODES: list[ChargingNode] = [
    ChargingNode("CLAP_BUNDLD-APND",            grid_x=2, grid_y=1),
    ChargingNode("POD_DUTCH1_7_UNIT 1-APND",    grid_x=5, grid_y=4),
    ChargingNode("POD_SLST13_2_SOLAR1-APND",    grid_x=4, grid_y=3),
    ChargingNode("ALAMIT_2_PL1X3-APND",         grid_x=2, grid_y=9),
    ChargingNode("POD_CRSTWD_6_KUMYAY-APND",    grid_x=4, grid_y=7),
]


def get_node_positions() -> dict[str, tuple[int, int]]:
    """Return {node_name: (grid_x, grid_y)} for all hardcoded charging nodes."""
    return {n.name: (n.grid_x, n.grid_y) for n in CHARGING_NODES}


# ── EV data container ─────────────────────────────────────────────────────────

@dataclass
class EV:
    """
    Plain data container for one electric vehicle.
    No behaviour lives here — see model.active_slots / model.is_active.

    grid_x / grid_y : fixed position on the 10×10 grid, loaded from the
                      X and Y columns in ev_info.csv.
    node_id         : None on load; assign_ev_nodes() fills it in before
                      the LP runs.
    """
    name:                  str
    arrival:               int
    departure:             int
    arrival_energy:        float
    desired_energy:        float
    battery_capacity:      float = 50.0
    max_charging_power:    float = 11.0
    max_discharging_power: float = 5.5
    grid_x:                int   = field(default=0)   # from CSV column "X"
    grid_y:                int   = field(default=0)   # from CSV column "Y"
    node_id:               object = field(default=None)   # set by assign_ev_nodes


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_nodal_prices(
    n_nodes: int = 5,
    periods: int = 24,
    filename: str = "caiso.csv",
) -> tuple[pd.DataFrame, list, list[int]]:
    """
    Load CAISO LMP data and return:
      - DataFrame  indexed by hour (1…periods), columns = node names, values in $/kWh
      - list of selected node names
      - list of integer time periods

    Node order is driven by CHARGING_NODES so grid positions stay in sync.
    """
    csv_path = os.path.join(DATA_DIR, filename)
    df_raw = pd.read_csv(csv_path)

    df_raw = df_raw[df_raw["LMP_TYPE"] == "LMP"].copy()
    df_raw = (
        df_raw[["OPR_HR", "NODE", "MW"]]
        .rename(columns={"OPR_HR": "Hour", "NODE": "Node", "MW": "Price_MWh"})
    )

    # Keep only nodes that have a full record
    full_nodes = (
        df_raw.groupby("Node")["Hour"]
        .nunique()
        .pipe(lambda s: s[s >= periods])
        .index.tolist()
    )

    # Use the node names from CHARGING_NODES (preserves grid-position mapping)
    diverse_nodes = [cn.name for cn in CHARGING_NODES]
    selected = [n for n in diverse_nodes if n in full_nodes][:n_nodes]

    df_raw = df_raw[df_raw["Node"].isin(selected) & (df_raw["Hour"] <= periods)].copy()
    df_raw["Price_kWh"] = df_raw["Price_MWh"] / 1_000.0

    df = (
        df_raw
        .pivot_table(index="Hour", columns="Node", values="Price_kWh")
        .sort_index()[selected]
    )

    return df, selected, list(range(1, periods + 1))


def load_carbon_intensity(
    filename: str = "df_fuel_ckan.csv",
    periods: int = 24,
) -> tuple[dict[int, float], list[int]]:
    """
    Returns carbon intensity dict {period: g CO₂/kWh} and period list.
    Rows are subsampled so that 24 half-hourly rows → 24 hourly periods.
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df = df.iloc[300338 : 300338 + periods * 2 : 2].reset_index(drop=True).head(periods)

    df["Period"]           = range(1, periods + 1)
    df["CARBON_INTENSITY"] = df["CARBON_INTENSITY"] / 1_000.0   # g/MWh → g/kWh

    carbon = dict(zip(df["Period"], df["CARBON_INTENSITY"]))
    return carbon, list(carbon.keys())


def load_evs(filename: str = "ev_info.csv") -> list[EV]:
    """
    Load EV fleet from CSV.  Columns expected:
      EV, Arrival Time, Departure Time, Arrival Energy, Desired Energy,
      X, Y  (fixed grid position — integers 1–10),
      Battery Capacity (opt), Max Charging Power (opt),
      Max Discharging Power (opt), Node ID (opt — overwritten later).

    X and Y must be present in the CSV.  They are written as fixed values
    (not generated at runtime) so positions are stable across runs.
    """
    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path)

    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(
            "ev_info.csv is missing 'X' and/or 'Y' columns.\n"
            "Add fixed grid positions (integers 1–10) for every EV and re-run."
        )

    evs = []
    for _, row in df.iterrows():
        evs.append(EV(
            name                  = str(row["EV"]),
            arrival               = int(row["Arrival Time"]),
            departure             = int(row["Departure Time"]),
            arrival_energy        = float(row["Arrival Energy"]),
            desired_energy        = float(row["Desired Energy"]),
            battery_capacity      = float(row.get("Battery Capacity",      60.0)),
            max_charging_power    = float(row.get("Max Charging Power",     7.4)),
            max_discharging_power = float(row.get("Max Discharging Power",  3.7)),
            grid_x                = int(row["X"]),
            grid_y                = int(row["Y"]),
            node_id               = None,   # assigned by assign_ev_nodes()
        ))
    return evs