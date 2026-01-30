#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for cell colony simulation data.

Loads and plots simulation results from numpy files.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import argparse
from pathlib import Path
from matplotlib.colors import ListedColormap

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_default_data_path() -> Path:
    """Get the default data directory path."""
    return get_project_root() / "data"


def check_data_exists(data_path: Path, run_name: str) -> bool:
    """
    Check if required data files exist.

    Returns True if all required files exist, False otherwise.
    """
    run_dir = data_path / run_name
    required_files = [
        f"{run_name}_cell_grid.npy",
        f"{run_name}_run_info.npy",
        f"{run_name}_r_at_t.npy",
        f"{run_name}_s_at_t.npy",
    ]

    for filename in required_files:
        filepath = run_dir / filename
        if not filepath.exists():
            return False
    return True


def print_data_missing_message(data_path: Path, run_name: str):
    """Print helpful message when data is missing."""
    print("=" * 60)
    print("DATA NOT FOUND")
    print("=" * 60)
    print(f"Looking for: {data_path / run_name}")
    print()
    print("The simulation data files are not present.")
    print("To download them, run:")
    print()
    print("    python3 scripts/download_data.py --extract")
    print()
    print("This will download and extract the simulation data from ERDA.")
    print("Note: You may need to install 'unrar' or 'rarfile' for extraction.")
    print("=" * 60)


def list_available_runs(data_path: Path) -> list:
    """List available simulation runs in the data directory."""
    if not data_path.exists():
        return []

    runs = []
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith("2drop_"):
            # Check if it has the required files
            if (item / f"{item.name}_cell_grid.npy").exists():
                runs.append(item.name)
    return sorted(runs)


def load_simulation_data(data_path: Path, run_name: str) -> dict:
    """Load simulation data from numpy files."""
    run_dir = data_path / run_name

    data = {
        'c_data': np.load(run_dir / f"{run_name}_cell_grid.npy", mmap_mode='r'),
        'run_info': np.load(run_dir / f"{run_name}_run_info.npy", allow_pickle=True),
        'r_at_t': np.load(run_dir / f"{run_name}_r_at_t.npy", allow_pickle=True),
        's_at_t': np.load(run_dir / f"{run_name}_s_at_t.npy", allow_pickle=True),
    }
    return data


def plot_colony(data: dict, run_name: str, scale_bar: bool = True, save_path: str = None):
    """
    Plot the cell colony grid.

    Args:
        data: Dictionary containing simulation data
        run_name: Name of the simulation run
        scale_bar: Whether to add a scale bar
        save_path: If provided, save the figure to this path
    """
    c_data = data['c_data']
    run_info = data['run_info']

    print("Run info:")
    print(run_info)

    ymin = np.min(c_data[0])
    ymax = np.max(c_data[0])
    xmin = np.min(c_data[1])
    xmax = np.max(c_data[1])

    c_grid = np.zeros((int(ymax - ymin + 1), int(xmax - xmin + 1)))
    c_grid[c_data[0] - ymin, c_data[1] - xmin] = c_data[3]
    c_grid[(c_grid < 11) & (c_grid > 0)] = 1
    c_grid[c_grid > 10] = 2

    fig, ax = plt.subplots()
    ax.matshow(c_grid, cmap=ListedColormap(['black', 'blue', 'yellow']))

    is_1drop = not re.match("2drop", run_name)

    if not is_1drop:
        title = rf"a_0: {run_info[24, 1]}, $\gamma$: {(float(run_info[16, 1])*30**2):.3f}"
        file_title = rf"2d_a{run_info[24, 1]}_g{(float(run_info[16, 1])*30**2):.3f}"
    else:
        title = rf"ratio: {run_info[14,1]}, a_0: {run_info[24, 1]}, $\gamma$: {(float(run_info[16, 1])*30**2):.3f}"
        file_title = rf"1d_r{run_info[14,1]}_a{run_info[24, 1]}_g{(float(run_info[16, 1])*30**2):.3f}"

    ax.set_title(title)

    if scale_bar and HAS_SCALEBAR:
        color = "black" if not is_1drop else "white"
        scalebar = ScaleBar(
            1,
            scale_formatter=lambda value, unit: fr"{value} $\mu$m",
            location="lower right",
            box_alpha=0,
            color=color,
            fixed_value=500
        )
        plt.gca().add_artist(scalebar)
    elif scale_bar and not HAS_SCALEBAR:
        print("Warning: matplotlib_scalebar not installed. Skipping scale bar.")
        print("Install with: pip install matplotlib-scalebar")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=800)
        print(f"Saved figure to: {save_path}")

    return fig, ax, file_title


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot cell colony simulation results"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (default: <project>/data/)"
    )
    parser.add_argument(
        "--run",
        type=str,
        default="2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A4_0",
        help="Name of simulation run to plot"
    )
    parser.add_argument(
        "--no-scale-bar",
        action="store_true",
        help="Disable scale bar"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure to this path (e.g., output.tiff)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available simulation runs"
    )

    args = parser.parse_args()

    # Determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_default_data_path()

    # List available runs if requested
    if args.list:
        runs = list_available_runs(data_path)
        if runs:
            print("Available simulation runs:")
            for run in runs:
                print(f"  {run}")
        else:
            print(f"No simulation runs found in {data_path}")
            print_data_missing_message(data_path, "")
        return

    # Check if data exists
    if not check_data_exists(data_path, args.run):
        print_data_missing_message(data_path, args.run)
        sys.exit(1)

    # Load and plot
    data = load_simulation_data(data_path, args.run)
    plot_colony(data, args.run, scale_bar=not args.no_scale_bar, save_path=args.save)

    plt.show(block=False)
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
