# Cell Colony Lattice Model

Simulation code accompanying the paper:

> **Pushing for survival: Spatial intermixing and indirect resistance enable collective growth**
> 
> M. Cordero, B.H. Thomsen, A. Talliou, A.K. Ehrmann, S.L. Svenningsen, N. Mitarai, L. Jauffred

---

## Overview

This repository contains the lattice-based simulation model used to study how spatial structure and indirect antibiotic resistance shape the outcome of competition between resistant (R) and sensitive (S) bacterial cells growing as colonies on a surface.

The model places cells on a 2D lattice (1 µm grid spacing) and simulates:
- **Cell growth and division** with mechanical pushing of neighbors (up to a configurable pushing radius)
- **Antibiotic diffusion** solved via finite differences with no-flux boundary conditions
- **Antibiotic degradation** by resistant cells, which creates a local protection zone for sensitive neighbours
- **Parallelised computation** using [Ray](https://www.ray.io/) to distribute the cell-division work across CPU cores

Two main experimental geometries are supported:
- **Two-drop collision** (`2drop`): a resistant and a sensitive colony grow toward each other and collide
- **Coffee-ring / mixed drop** (`1drop_co`): a mixed population arranged in an annular ring

---

## Repository structure

```
CellColonyLatticeModel/
├── scripts/
│   ├── LM_func22.py                  # Core simulation library (functions for grid setup,
│   │                                 #   cell division, diffusion, Ray remote workers)
│   ├── LM_run_file.py                # Main simulation runner — configure parameters here
│   │                                 #   and run to produce .npy output files
│   ├── plot.py                       # Original plotting script: loads .npy output and
│   │                                 #   displays colony images with a scale bar
│   │
│   ├── run_RSR_with_timepoints.py    # Fig. 4 simulation — RSR stripe geometry,
│   │                                 #   saves spatial snapshots at specified timepoints
│   ├── RSR_initial_conditions.py     # Helper: creates the R-S-R vertical stripe
│   │                                 #   initial condition used by the above script
│   ├── create_panel_figure_pil2.py   # Fig. 4 figure — assembles the multi-panel image
│   │                                 #   from RSR timeseries snapshots using PIL
│   │
│   ├── wellmixed_competition.py      # Well-mixed ODE model (R vs S with antibiotics);
│   │                                 #   used for comparison, not included in the article
│   │
│   ├── analyze_RSR_closure.py        # Analysis: RSR stripe closure width vs time
│   ├── visualize_timepoints.py       # Visualisation helper for RSR timepoint snapshots
│   └── download_data.py              # Downloads simulation output data from ERDA
│
├── src/
│   └── plotting.py                   # Enhanced CLI plotting tool for exploring results
│                                     #   (supports magnification, scale bar, TIFF export)
│
├── GrowthRate_Experiment/
│   ├── growth_rate_analysis.py       # Fits growth curves to experimental OD data
│   ├── growth_rate_fit.py            # Parameter estimation for growth rate model
│   └── *.csv                         # Raw growth assay data
│
├── data/                             # Simulation output (not tracked in git)
├── results/                          # Generated figures (not tracked in git)
└── requirements.txt
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/namikomit/CellColonyLatticeModel.git
   cd CellColonyLatticeModel
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) To extract downloaded data archives, install `unrar`:
   ```bash
   # Ubuntu/Debian
   sudo apt install unrar
   ```

---

## Running simulations

> **Note:** Simulations are computationally intensive and were run on a compute cluster using 25 CPU cores. Expect multi-hour runtimes even on a workstation. All scripts with cluster-specific file paths are noted below — update the `working_dir` / `save_dir` variables before running locally.
>
### Growth rate fit (Fig 1a)
GrowthRate_Experiment/

### Main simulation (Figs. 1–3)

Edit parameters and paths at the top of the script, then run:

```bash
python scripts/LM_run_file.py
```

Key parameters to set:
- `working_dir` — path to the directory containing `LM_run_file.py` and `LM_func22.py`
- `exp_setup` — `"2drop"` for collision experiment, `"1drop_co"` for coffee-ring
- `a_conc` — list of initial antibiotic concentrations to sweep over
- `pushing_radius`, `k_A`, `g_nonnorm` — physical model parameters

Output `.npy` files are written to `data/<run_name>/`.

### Fig. 4 simulation (RSR stripe)

```bash
python scripts/run_RSR_with_timepoints.py
```

Update `working_dir` and `save_dir` inside the script to point to your local paths. The script iterates over a list of sensitive-stripe widths and saves spatial snapshots at specified timepoints.

### Fig. 4 panel figure

```bash
python scripts/create_panel_figure_pil2.py
```

Update `data_dir` and `output_dir` at the top of the script.

### Well-mixed ODE model

```bash
python scripts/wellmixed_competition.py
```

Runs entirely locally; no path configuration required. Saves three figures (`wellmixed_single.png`, `wellmixed_scan.png`, `wellmixed_phase.png`) to the current directory.

---

## Downloading simulation data

Pre-computed simulation output is hosted on ERDA and can be downloaded with:

```bash
python scripts/download_data.py          # download only
python scripts/download_data.py --extract  # download and extract
```

---

## Visualising results

### Original plot script

Edit the `main_path` and `run_name` variables at the top of `scripts/plot.py`, then run:

```bash
python scripts/plot.py
```

### Enhanced CLI tool

```bash
# List available runs in data/
python src/plotting.py --list

# Full colony view
python src/plotting.py --run <run_name>

# Magnified interface view saved as TIFF (default)
python src/plotting.py --run <run_name> --magnify --save output

# Custom width and scale bar
python src/plotting.py --run <run_name> --magnify --width 500 --scalebar 100
```

---

## Data format

Each simulation produces a set of `.npy` files in `data/<run_name>/`:

| File | Contents |
|---|---|
| `*_cell_grid.npy` | Sparse cell positions: rows (row, col, layer, cell_type) |
| `*_run_info.npy` | 2D array of [parameter\_name, value] pairs |
| `*_r_at_t.npy` | Resistant cell count at each timestep |
| `*_s_at_t.npy` | Sensitive cell count at each timestep |
| `*_antibiotic_grid.npy` | Final antibiotic concentration field |

Cell type encoding: values 1–10 = resistant (division cycle stages), 11–20 = sensitive.

---

## Authors

- Mireia Cordero
- Brage H. Thomsen
- Artemis Talliou
- Anja K. Ehrmann
- Sine L. Svenningsen
- Namiko Mitarai
- Liselotte Jauffred

The main part of simulation code was developed by Brage H. Thomsen. Namiko Mitarai developed some part of analalysis and plotting, with using Claude Code.

---

## License

This code is made available for research and reproducibility purposes in conjunction with the above publication.
