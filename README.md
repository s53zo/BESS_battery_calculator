# BESS Battery Calculator

Simulate how a battery energy storage system (BESS) can soak up PV exports and sell the surplus on Slovenia's day-ahead electricity market.

## Requirements
- Python 3.8 or newer
- `pandas` and `matplotlib` (install with `pip install pandas matplotlib`)

## Data
- PV production samples for 2023–2025 are already included in the repository (`pv_measurements_2023.csv`, `pv_measurements_2024.csv`, `pv_measurements_2025.csv`).
- Day-ahead price data for 2025 is provided in `energy-charts_Electricity_production_and_spot_prices_in_Slovenia_in_2025.csv`.

If you use different files, pass their paths via the `--pv-files` and `--price-file` arguments.

## Usage
```bash
python3 bess_calculator.py [options]
```

### Common options
- `--capacity`: BESS size in kWh (default `50`).
- `--pv-files`: Space-separated PV CSV paths (defaults cover 2023–2025 data shipped with this repo).
- `--price-file`: Day-ahead price CSV path (default is the 2025 file listed above).
- `--max-sell`: Maximum kWh that can be sold per 15-minute interval (default `4.5`).
- `--battery-cost-per-kwh`: Used by the capacity sweep to compute ROI (default `150`).
- `--optimize-capacity`: Enable a sweep over capacities defined by `--capacity-min`/`--capacity-max`/`--capacity-step`.
- `--one-day DATE`: Print per-interval state for the given date and generate a detailed plot.
- `--average`: Average historical PV production per interval before running the simulation.

## Output
- Console summary of the simulated sales, lost energy, and planned selling windows.
- Optional per-day breakdowns when `--one-day` is provided.
- Charts are written to the `bess_charts/` folder:
  - `bess_monthly_best.png`, `bess_daily_energy.png`, and a handful of timelines/windows plots.

## Examples
- Run the default simulation:
  ```bash
  python3 bess_calculator.py
  ```
- Sweep capacity for ROI (uses the default files):
  ```bash
  python3 bess_calculator.py --optimize-capacity --optimize-objective roi
  ```
- Inspect a single day using averaged PV production:
  ```bash
  python3 bess_calculator.py --average --one-day 2025-07-01
  ```

Feel free to add your own PV or price files and rerun the script with `--pv-files`/`--price-file` to explore different scenarios.
