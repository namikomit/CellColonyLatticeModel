# Cell Colony Lattice Model

A lattice-based simulation model for bacterial cell colony growth and collision dynamics. The model simulates the growth of resistant and sensitive cell populations, including antibiotic diffusion and cell-cell interactions.

## Features

- Lattice-based cell growth simulation with configurable parameters
- Parallel computation using Ray for efficient simulation
- Antibiotic diffusion modeling with finite difference methods
- Support for "coffee ring" and "two-drop collision" experimental setups
- Visualization tools for simulation results

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

3. (Optional) For extracting downloaded data, install unrar:
   ```bash
   # Ubuntu/Debian
   sudo apt install unrar

   # Or use the Python package
   pip install rarfile
   ```

## Downloading Data

Simulation data is hosted on ERDA and can be downloaded using the provided script:

```bash
# Download data files
python3 scripts/download_data.py

# Download and extract
python3 scripts/download_data.py --extract
```

## Usage

### Plotting Simulation Results

```bash
# List available simulation runs
python3 src/plotting.py --list

# Plot a specific run
python3 src/plotting.py --run 2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A0_0

# Save plot to file
python3 src/plotting.py --run <run_name> --save output.png

# Use custom data path
python3 src/plotting.py --data-path /path/to/data --run <run_name>
```

### Running Simulations

```bash
python3 scripts/run_simulation.py
```

Note: Running simulations requires significant computational resources and is typically done on a cluster environment.

## Project Structure

```
CellColonyLatticeModel/
├── src/
│   ├── lattice_model.py    # Core simulation functions
│   └── plotting.py         # Visualization tools
├── scripts/
│   ├── run_simulation.py   # Main simulation runner
│   └── download_data.py    # Data download utility
├── data/                   # Simulation data (not in git)
├── results/                # Generated plots (not in git)
└── docs/                   # Documentation
```

## Authors

- Brage HT
- Namiko Mitarai

## License

This project is for research purposes.
