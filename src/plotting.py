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


def build_cell_grid(c_data):
    """
    Build a 2D cell grid from sparse cell data.

    Args:
        c_data: Sparse cell data array (4 x N) with rows, cols, layers, cell_types

    Returns:
        c_grid: 2D array with cell types (0=empty, 1=resistant, 2=sensitive)
        bounds: tuple of (ymin, ymax, xmin, xmax)
    """
    ymin = np.min(c_data[0])
    ymax = np.max(c_data[0])
    xmin = np.min(c_data[1])
    xmax = np.max(c_data[1])

    c_grid = np.zeros((int(ymax - ymin + 1), int(xmax - xmin + 1)))
    c_grid[c_data[0] - ymin, c_data[1] - xmin] = c_data[3]
    c_grid[(c_grid < 11) & (c_grid > 0)] = 1  # Resistant
    c_grid[c_grid > 10] = 2  # Sensitive

    return c_grid, (ymin, ymax, xmin, xmax)


def plot_interface_magnified(data: dict, run_name: str, save_path: str = None,
                              width_um: int = 1000, scalebar_um: int = 200):
    """
    Plot a magnified view of the interface between resistant and sensitive cells.

    Args:
        data: Dictionary containing simulation data
        run_name: Name of the simulation run
        save_path: If provided, save the figure to this path
        width_um: Width of the magnified region in micrometers (default: 1000)
        scalebar_um: Length of scale bar in micrometers (default: 200)

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    c_data = data['c_data']

    # Build full grid
    c_grid, (ymin, ymax, xmin, xmax) = build_cell_grid(c_data)

    # Find interface between resistant and sensitive cells
    resistant_mask = (c_data[3] >= 1) & (c_data[3] <= 10)
    sensitive_mask = (c_data[3] >= 11) & (c_data[3] <= 20)

    resistant_cols = c_data[1][resistant_mask]
    sensitive_cols = c_data[1][sensitive_mask]

    interface_col = (np.max(resistant_cols) + np.min(sensitive_cols)) // 2
    center_row = (ymin + ymax) // 2

    # Define magnification region
    half_width = width_um // 2
    col_start = max(0, int(interface_col - half_width - xmin))
    col_end = min(c_grid.shape[1], int(interface_col + half_width - xmin))
    row_start = max(0, int(center_row - half_width - ymin))
    row_end = min(c_grid.shape[0], int(center_row + half_width - ymin))

    # Extract magnified region
    mag_grid = c_grid[row_start:row_end, col_start:col_end]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mag_grid, cmap=ListedColormap(['black', 'blue', 'yellow']),
              interpolation='nearest', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Add scale bar (1 pixel = 1 µm since cdx=1)
    if HAS_SCALEBAR:
        scalebar = ScaleBar(
            1, 'um',
            fixed_value=scalebar_um,
            fixed_units='um',
            location='lower right',
            color='black',
            box_alpha=0,
            scale_loc='none',
            border_pad=0.5,
        )
        ax.add_artist(scalebar)
    else:
        print("Warning: matplotlib_scalebar not installed. Skipping scale bar.")

    plt.tight_layout()

    if save_path:
        # Default to TIFF format if no extension specified
        if not any(save_path.endswith(ext) for ext in ['.tiff', '.tif', '.png', '.pdf', '.eps', '.svg']):
            save_path = save_path + '.tiff'
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', pad_inches=0.02)
        print(f"Saved magnified interface to: {save_path}")

    return fig, ax


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

    c_grid, _ = build_cell_grid(c_data)

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
        help="Save figure to this path (default format: .tiff for magnified, .png otherwise)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available simulation runs"
    )
    parser.add_argument(
        "--magnify",
        action="store_true",
        help="Plot magnified view of the interface region"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Width of magnified region in micrometers (default: 1000)"
    )
    parser.add_argument(
        "--scalebar",
        type=int,
        default=200,
        help="Scale bar length in micrometers (default: 200)"
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

    if args.magnify:
        plot_interface_magnified(
            data, args.run,
            save_path=args.save,
            width_um=args.width,
            scalebar_um=args.scalebar
        )
    else:
        plot_colony(data, args.run, scale_bar=not args.no_scale_bar, save_path=args.save)

    plt.show(block=False)
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
