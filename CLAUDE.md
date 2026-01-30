# CLAUDE.md - Project Context for AI Assistants

## Project Overview

This is a **Cell Colony Lattice Model** simulation project that models bacterial colony growth and collision dynamics on a 2D lattice. The simulation tracks resistant (R) and sensitive (S) cell populations with antibiotic diffusion.

## Key Concepts

- **Two-drop experiment (2drop)**: Two cell colonies (resistant and sensitive) grow and eventually collide
- **Coffee ring effect (1drop_co)**: Mixed population in an annular initial configuration
- **CRE**: Coffee Ring Effect - cells distributed in an annular pattern with inner/outer radius
- **Pushing radius (rp)**: Maximum distance cells can push neighbors during division

## Project Structure

```
src/
  lattice_model.py  - Core simulation library (grid creation, diffusion, cell growth, Ray workers)
  plotting.py       - Visualization with CLI interface

scripts/
  run_simulation.py - Main simulation runner (uses Ray for parallelization)
  download_data.py  - Downloads simulation data from ERDA
```

## Key Parameters

- `ddim_r`, `ddim_c`: Diffusion grid dimensions
- `cdx`: Cell grid spacing (μm)
- `ddx`: Diffusion grid spacing (μm)
- `D`: Diffusion coefficient
- `gamma`: Antibiotic removal rate by resistant cells
- `k_A`: Antibiotic sensitivity constant
- `lambda_max_r/s`: Maximum growth rates for resistant/sensitive cells

## Data Format

Simulation outputs are stored as `.npy` files:
- `*_cell_grid.npy`: Sparse cell positions (row, col, layer, cell_type)
- `*_run_info.npy`: 2D array of [parameter_name, value] pairs
- `*_r_at_t.npy`, `*_s_at_t.npy`: Cell counts over time

Cell types: 1-10 = resistant (growth stages), 11-20 = sensitive (growth stages)

## Common Tasks

### Plot simulation results
```bash
python3 src/plotting.py --run <run_name>              # Full colony view
python3 src/plotting.py --run <run_name> --magnify    # Magnified interface (1000 µm)
python3 src/plotting.py --run <run_name> --magnify --width 500 --scalebar 100
python3 src/plotting.py --list  # List available runs
```

### Download data
```bash
python3 scripts/download_data.py --extract
```

### Run simulation (cluster environment)
```bash
python3 scripts/run_simulation.py
```

## Dependencies

- numpy, scipy: Numerical computation
- numba: JIT compilation for performance
- ray: Distributed computing
- matplotlib: Visualization
- tqdm: Progress bars

## Notes

- Simulations are computationally intensive and typically run on clusters
- Data files are large (~10-30 MB per run as RAR archives)
- The `data/` and `results/` folders are gitignored
