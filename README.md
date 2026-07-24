# A Locational Price and Carbon Intensity-Aware Charger Allocation and Charge–Discharge Scheduling Strategy for Eco-smart Electric Vehicles

Code and data for the paper *"A Locational Price and Carbon Intensity-Aware
Charger Allocation and Charge–Discharge Scheduling Strategy for Eco-smart
Electric Vehicles"* (Jung, Dash, and Srinivasan, 2026).

A multi-objective EV charging scheduler that trades off **electricity cost**
against **carbon emissions** using real CAISO nodal price data and per-node grid
carbon intensity. The two signals are combined into a single Integrated Charging
Signal (ICS), weighted by α (cost) and β (carbon).

The pipeline has two stages. First, a capacity-aware greedy procedure assigns
each EV to a charging node subject to a distance cap, per-node power limits, and
energy-throughput feasibility. Then a linear program schedules charging — and
optional vehicle-to-grid (V2G) discharging — over a 24-hour horizon, with a
linear battery-degradation cost and a soft, β-scaled carbon cap. The model is
evaluated under four schemes (Uncoordinated baseline, Cost-only, Carbon-only,
Balanced) plus a Pareto sweep, and all figures and tables are generated
automatically.

On a synthetic 30-EV fleet across 4 CAISO nodes, the balanced scheme (α = β =
0.5) cuts electricity cost by 39.6% and carbon emissions by 21.6% relative to the
uncoordinated baseline.

## Requirements

- Python 3
- Dependencies (installed in the bundled `venv/`): `pandas`, `numpy`,
  `matplotlib`, and an LP solver (`pulp`).

## Setup

```bash
# from the project root
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib pulp
```

## Usage

```bash
python3 main.py
```

This will:

1. Load data — CAISO nodal prices, per-node carbon intensity, and the EV fleet.
2. Print grid positions for charging nodes and EVs.
3. Run the **Balanced** scheme (α = β = 0.5) as the primary schedule and print a
   detailed per-EV report (node assignments, distances, metrics vs. baseline).
4. Run all four evaluation schemes and the Pareto sweep.
5. Print the four-scheme summary table.
6. Generate all figures and the performance table into `plots/`.

## Configuration

Edit the `CONFIG` block near the top of `main.py` to change global parameters:

| Parameter | Meaning |
|---|---|
| `n_nodes` | Number of CAISO nodes used |
| `periods` | Scheduling horizon (hours) |
| `carbon_cap_fraction` | Soft carbon cap as a fraction of the uncoordinated baseline |
| `v2g_enabled` | Enable vehicle-to-grid discharging |
| `eta_c` / `eta_d` | Charge / discharge efficiency |
| `deg_cost` | Battery degradation cost ($/kWh discharged) |
| `grid_cap_kw` | Per-node power cap (kW) |
| `max_distance` | Max EV-to-node distance (grid units) |
| `out_dir` | Output directory for figures/tables |

**To change which carbon-intensity date each node uses**, edit the
`NODE_DATE_MAP` dict near the top of `scheduler/data_loader.py`.

The values used for the paper are: 30 EVs, 4 nodes, 24-hour horizon, node power
cap 85 kW, max assignment distance 4.0 grid units, charge/discharge efficiency
0.95, max charge 12 kW, max discharge 4 kW, degradation cost \$0.02/kWh, and a
carbon-cap fraction of 0.85.

## Data

- **LMP prices** — CAISO day-ahead market via the Open Access Same-Time
  Information System (OASIS), trade date Feb 18, 2026, for four nodes:
  `CLAP_BUNDLD`, `POD_DUTCH1_7_UNIT 1`, `POD_SLST13_2_SOLAR1`, `ALAMIT_2_PL1X3`.
- **Carbon intensity** — CAISO Average Emissions Rate report (Feb 2026); each
  node is assigned one day's profile (Feb 17–20) via `NODE_DATE_MAP`.

## Project layout

```
.
├── main.py                  # Entry point (config + orchestration)
├── data/
│   ├── caiso.csv            # CAISO nodal price data
│   ├── carbon_intensity.csv # Per-node carbon intensity profiles
│   └── ev_infoV5.csv        # EV fleet specs
├── scheduler/
│   ├── data_loader.py       # Loads prices, carbon, EVs; node/date mapping
│   ├── model.py             # Node assignment, LP scheduler, baseline, metrics
│   ├── results.py           # Runs all schemes + Pareto sweep
│   └── plot.py              # Figure and performance-table generation
└── plots/                   # Generated PDFs + performance_table.{csv,tex}
```

## Outputs (`plots/`)

- `fig_pareto.pdf` — cost vs. carbon Pareto frontier
- `fig_carbon_overlay.pdf`, `fig_nodal_carbon.pdf` — carbon intensity over time / by node
- `fig_cost_price_overlay.pdf`, `fig_nodal_prices.pdf` — nodal electricity prices
- `fig_grid_map.pdf`, `grid_positions.pdf` — network topology / node & EV positions
- `fig_balanced_node_profiles.pdf` — charging profiles under the balanced scheme
- `performance_table.csv` / `performance_table.tex` — summary metrics table
