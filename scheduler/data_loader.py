"""Loads nodal LMP prices (CAISO), per-node carbon intensity, and the EV fleet."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


@dataclass
class ChargingNode:
    """A CAISO node name and its fixed (x, y) position on the 10x10 grid."""
    name:   str    # must match a column in the nodal price DataFrame
    grid_x: int    # 1-10
    grid_y: int    # 1-10


# Node positions on the grid. Edit here to move nodes.
CHARGING_NODES: list[ChargingNode] = [
    ChargingNode("CLAP_BUNDLD-APND",            grid_x=3, grid_y=3),
    ChargingNode("POD_DUTCH1_7_UNIT 1-APND",    grid_x=8, grid_y=3),
    ChargingNode("POD_SLST13_2_SOLAR1-APND",    grid_x=3, grid_y=8),
    ChargingNode("ALAMIT_2_PL1X3-APND",         grid_x=8, grid_y=8),
]


def get_node_positions() -> dict[str, tuple[int, int]]:
    """Return {node_name: (grid_x, grid_y)} for all charging nodes."""
    return {n.name: (n.grid_x, n.grid_y) for n in CHARGING_NODES}


# Carbon intensity date assigned to each node. Keys must match CHARGING_NODES
# names; values must be dates present in carbon_intensity.csv (M/D/YYYY).
NODE_DATE_MAP: dict[str, str] = {
    "CLAP_BUNDLD-APND":            "2/17/2026",
    "POD_DUTCH1_7_UNIT 1-APND":    "2/18/2026",
    "POD_SLST13_2_SOLAR1-APND":    "2/19/2026",
    "ALAMIT_2_PL1X3-APND":         "2/20/2026",
}


@dataclass
class EV:
    """One electric vehicle. node_id is None on load and set by assign_ev_nodes()."""
    name:                  str
    arrival:               int
    departure:             int
    arrival_energy:        float
    desired_energy:        float
    battery_capacity:      float = 40.0
    max_charging_power:    float = 12.0
    max_discharging_power: float = 4.0
    grid_x:                int   = field(default=0)   # from CSV column "X"
    grid_y:                int   = field(default=0)   # from CSV column "Y"
    node_id:               object = field(default=None)


def load_nodal_prices(
    n_nodes: int = 5,
    periods: int = 24,
    filename: str = "caiso.csv",
) -> tuple[pd.DataFrame, list, list[int]]:
    """
    Load CAISO LMP data. Returns a DataFrame indexed by hour with one column
    per node ($/kWh), the selected node names, and the list of time periods.
    Node order follows CHARGING_NODES so grid positions stay in sync.
    """
    csv_path = os.path.join(DATA_DIR, filename)
    df_raw = pd.read_csv(csv_path)

    df_raw = df_raw[df_raw["LMP_TYPE"] == "LMP"].copy()
    df_raw = (
        df_raw[["OPR_HR", "NODE", "MW"]]
        .rename(columns={"OPR_HR": "Hour", "NODE": "Node", "MW": "Price_MWh"})
    )

    # Keep only nodes with a full 24-hour record
    full_nodes = (
        df_raw.groupby("Node")["Hour"]
        .nunique()
        .pipe(lambda s: s[s >= periods])
        .index.tolist()
    )

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


def load_nodal_carbon(
    filename:      str                    = "carbon_intensity.csv",
    periods:       int                    = 24,
    node_date_map: dict[str, str] | None  = None,
) -> dict[str, dict[int, float]]:
    """
    Load a per-node carbon intensity schedule. Each node gets the 24-hour
    AVG_EM_RATE profile for its date in node_date_map (defaults to NODE_DATE_MAP),
    converted from MTCO2e/MWh to g CO2/kWh. Returns {node: {period: g/kWh}}.
    """
    if node_date_map is None:
        node_date_map = NODE_DATE_MAP

    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path)

    available_dates = set(df["TRADE_DT"].unique())

    nodal_carbon: dict[str, dict[int, float]] = {}
    for node_name, date_str in node_date_map.items():
        if date_str not in available_dates:
            raise ValueError(
                f"Date '{date_str}' mapped to node '{node_name}' was not found "
                f"in {filename}. Available dates: {sorted(available_dates)}"
            )
        day_df = (
            df[df["TRADE_DT"] == date_str]
            .sort_values("TRADE_HR")
            .head(periods)
            .reset_index(drop=True)
        )
        day_df = day_df.assign(Period=range(1, len(day_df) + 1))
        nodal_carbon[node_name] = dict(
            zip(day_df["Period"], day_df["AVG_EM_RATE"] * 1_000.0)
        )

    return nodal_carbon


def load_evs(filename: str = "ev_infoV5.csv") -> list[EV]:
    """
    Load the EV fleet from CSV. Requires columns: EV, Arrival Time,
    Departure Time, Arrival Energy, Desired Energy, X, Y. Battery Capacity,
    Max Charging Power, and Max Discharging Power are optional.
    """
    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path)

    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(
            "ev_info.csv is missing 'X' and/or 'Y' columns. "
            "Add fixed grid positions (integers 1-10) for every EV and re-run."
        )

    evs = []
    for _, row in df.iterrows():
        evs.append(EV(
            name                  = str(row["EV"]),
            arrival               = int(row["Arrival Time"]),
            departure             = int(row["Departure Time"]),
            arrival_energy        = float(row["Arrival Energy"]),
            desired_energy        = float(row["Desired Energy"]),
            battery_capacity      = float(row.get("Battery Capacity",      40.0)),
            max_charging_power    = float(row.get("Max Charging Power",     12.0)),
            max_discharging_power = float(row.get("Max Discharging Power",  4.0)),
            grid_x                = int(row["X"]),
            grid_y                = int(row["Y"]),
            node_id               = None,
        ))
    return evs
