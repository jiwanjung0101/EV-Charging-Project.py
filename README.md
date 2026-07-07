# EV Charging Project

An EV charging scheduler that optimizes fleet charging schedules to trade off
**electricity cost** against **carbon emissions**, using real CAISO nodal price
data and per-node grid carbon intensity.

Each EV is assigned to a charging node (subject to a distance cap and per-node
power limits), then a linear program schedules charging — and optional
vehicle-to-grid (V2G) discharging — over a 24-hour horizon. The model is
evaluated under four schemes (Unmanaged baseline, Cost-only, Carbon-only,
Balanced) plus a Pareto sweep, and all figures and summary tables are generated
automatically.

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
| `carbon_cap_fraction` | Soft carbon cap as a fraction of the unmanaged baseline |
| `v2g_enabled` | Enable vehicle-to-grid discharging |
| `eta_c` / `eta_d` | Charge / discharge efficiency |
| `deg_cost` | Battery degradation cost ($/kWh discharged) |
| `grid_cap_kw` | Per-node power cap (kW) |
| `max_distance` | Max EV-to-node distance (grid units) |
| `out_dir` | Output directory for figures/tables |

**To change which carbon-intensity date each node uses**, edit the
`NODE_DATE_MAP` dict near the top of `scheduler/data_loader.py`.

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
