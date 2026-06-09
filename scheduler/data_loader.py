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

Nodal carbon intensity
──────────────────────
NODE_DATE_MAP assigns each charging node its own 24-hour carbon intensity
schedule drawn from a different date in carbon_intensity.csv.

▶  TO CHANGE WHICH DATE A NODE USES — edit NODE_DATE_MAP below.
   Keys must match CHARGING_NODES names; values must be dates present in
   carbon_intensity.csv (format: M/D/YYYY, e.g. "2/18/2026").
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
    ChargingNode("CLAP_BUNDLD-APND",            grid_x=3, grid_y=3),
    ChargingNode("POD_DUTCH1_7_UNIT 1-APND",    grid_x=8, grid_y=3),
    ChargingNode("POD_SLST13_2_SOLAR1-APND",    grid_x=3, grid_y=8),
    ChargingNode("ALAMIT_2_PL1X3-APND",         grid_x=8, grid_y=8),
]


def get_node_positions() -> dict[str, tuple[int, int]]:
    """Return {node_name: (grid_x, grid_y)} for all hardcoded charging nodes."""
    return {n.name: (n.grid_x, n.grid_y) for n in CHARGING_NODES}


# ═══════════════════════════════════════════════════════════════════════════════
# ▶  NODE DATE MAP — edit here to change each node's carbon intensity schedule
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Keys   : must exactly match the node names in CHARGING_NODES above.
#   Values : date strings in M/D/YYYY format that exist in carbon_intensity.csv.
#            Available dates: 2/1/2026 through 2/28/2026.
#
#   Example: to give node 1 data from 2/1/2026 and node 2 data from 2/5/2026:
#     "CLAP_BUNDLD-APND":         "2/1/2026",
#     "POD_DUTCH1_7_UNIT 1-APND": "2/5/2026",
#
NODE_DATE_MAP: dict[str, str] = {
    "CLAP_BUNDLD-APND":            "2/17/2026",
    "POD_DUTCH1_7_UNIT 1-APND":    "2/18/2026",
    "POD_SLST13_2_SOLAR1-APND":    "2/19/2026",
    "ALAMIT_2_PL1X3-APND":         "2/20/2026",
}
# ═══════════════════════════════════════════════════════════════════════════════


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
    battery_capacity:      float = 40.0
    max_charging_power:    float = 12.0
    max_discharging_power: float = 4.0
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


def load_nodal_carbon(
    filename:      str                    = "carbon_intensity.csv",
    periods:       int                    = 24,
    node_date_map: dict[str, str] | None  = None,
) -> dict[str, dict[int, float]]:
    """
    Load a per-node carbon intensity schedule from the multi-day CSV.

    Each node in node_date_map gets the 24-hour AVG_EM_RATE profile for its
    assigned date.  Values are converted from MTCO2e/MWh → g CO₂/kWh (×1 000).

    Parameters
    ----------
    filename      : CSV file in DATA_DIR.  Must contain columns TRADE_DT,
                    TRADE_HR, and AVG_EM_RATE.
    periods       : number of hourly periods to load (default 24).
    node_date_map : override for NODE_DATE_MAP; pass None to use the module-
                    level constant (the normal case).

    Returns
    -------
    dict[str, dict[int, float]]
        {node_name: {period_1..24: g_CO2_per_kWh}}

    ▶  To change which date each node uses, edit NODE_DATE_MAP above.
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
                f"in {filename}.\n"
                f"Available dates: {sorted(available_dates)}"
            )
        day_df = (
            df[df["TRADE_DT"] == date_str]
            .sort_values("TRADE_HR")
            .head(periods)
            .reset_index(drop=True)
        )
        day_df = day_df.assign(Period=range(1, len(day_df) + 1))
        # MTCO2e/MWh → g CO₂/kWh  (1 MTCO2e = 1 000 000 g; 1 MWh = 1 000 kWh)
        nodal_carbon[node_name] = dict(
            zip(day_df["Period"], day_df["AVG_EM_RATE"] * 1_000.0)
        )

    return nodal_carbon


def load_carbon_intensity(
    filename: str = "carbon_intensity.csv",
    periods:  int = 24,
) -> tuple[dict[int, float], list[int]]:
    """
    Legacy flat carbon intensity loader.

    Returns the average g CO₂/kWh across all nodes in NODE_DATE_MAP for
    each period — useful for quick diagnostics but not used by the main
    pipeline, which calls load_nodal_carbon() instead.
    """
    nodal = load_nodal_carbon(filename=filename, periods=periods)
    # Average across all nodes per period
    n_nodes = len(nodal)
    carbon: dict[int, float] = {}
    for period in range(1, periods + 1):
        carbon[period] = sum(
            nc.get(period, 0.0) for nc in nodal.values()
        ) / max(n_nodes, 1)
    return carbon, list(carbon.keys())


def load_evs(filename: str = "ev_infoV5.csv") -> list[EV]:
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
            battery_capacity      = float(row.get("Battery Capacity",      40.0)),
            max_charging_power    = float(row.get("Max Charging Power",     12.0)),
            max_discharging_power = float(row.get("Max Discharging Power",  4.0)),
            grid_x                = int(row["X"]),
            grid_y                = int(row["Y"]),
            node_id               = None,   # assigned by assign_ev_nodes()
        ))
    return evs