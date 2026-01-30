# Simulation Data

This folder contains simulation output data for the Cell Colony Lattice Model.

## Data Source

The simulation data is hosted on ERDA (Electronic Research Data Archive):

**Share Link:** https://sid.erda.dk/sharelink/l8bcd32JVM

Specifically, this project uses data from:
- `simulation_data/no_pushing/` - Simulations without cell pushing mechanics

## Downloading Data

To download the data, run the following command from the project root:

```bash
python3 scripts/download_data.py
```

This will download RAR archives containing the simulation results.

To also extract the archives:

```bash
python3 scripts/download_data.py --extract
```

Note: Extraction requires `unrar` to be installed on your system, or the `rarfile` Python package.

## Data Format

Each simulation run is stored in a folder with the naming convention:
```
2drop_CRE{r_outer}{r_inner}_rp{pushing_radius}_icc{initial_cell_count}_g{gamma}_ka{k_A}_A{antibiotic_conc}_{run_number}
```

Each folder contains:
- `*_cell_grid.npy` - Cell positions and states
- `*_run_info.npy` - Simulation parameters
- `*_r_at_t.npy` - Resistant cell count over time
- `*_s_at_t.npy` - Sensitive cell count over time
- `*_antibiotic_grid.npy` - Antibiotic concentration field
- `*_div_mask.npy` - Division mask data

## Available Datasets

| File | Description |
|------|-------------|
| `2drop_..._A0_0.rar` | No antibiotic (control) |
| `2drop_..._A3_0.rar` | Antibiotic concentration 3 |
| `2drop_..._A4_0.rar` | Antibiotic concentration 4 |
| `2drop_..._A5_0.rar` | Antibiotic concentration 5 |
| `2drop_..._A6_0.rar` | Antibiotic concentration 6 |
| `2drop_..._A7_1.rar` | Antibiotic concentration 7 |
