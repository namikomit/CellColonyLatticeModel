"""
Visualize RSR Timepoint Snapshots - OPTIMIZED FOR SPEED

Creates overlay plots of cells + antibiotic field for each saved timepoint.
Publication-ready figures showing temporal dynamics.

CONFIGURATION SECTION - Edit these to change visualization:
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import glob
import re

# Disable some matplotlib features for speed
plt.ioff()  # Turn off interactive mode

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================

# Data directory
data_dir = Path("/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/data/RSR_timeseries_2")

# Output directory
output_dir = Path("/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/results/timepoints_2")
output_dir.mkdir(parents=True, exist_ok=True)

# DISPLAY RANGE (in diffusion grid coordinates, None = auto)
# Set these to zoom into specific region, or None to show full grid
y_min_display = 600   # Start Y coordinate (None for auto)
y_max_display = 1000  # End Y coordinate (None for auto)
x_min_display = 600   # Start X coordinate (None for auto)
x_max_display = 1000  # End X coordinate (None for auto)

# Figure size
figsize = (12, 10)  # Width, height in inches

# DPI for output (higher = better quality, but slower)
dpi = 600  # High resolution for publication

# ============================================================================
# END CONFIGURATION
# ============================================================================


def load_cell_grid(filepath):
    """Load sparse cell grid and convert to dense."""
    # Try mmap first (works for most files), fallback to allow_pickle if needed
    try:
        c_data = np.load(filepath, mmap_mode='r')
    except ValueError:
        # Falls back if file contains Python objects
        c_data = np.load(filepath, allow_pickle=True)
        # Handle case where data is wrapped in 0-dimensional array
        if c_data.shape == ():
            c_data = c_data.item()
    
    ymin = np.min(c_data[0])
    ymax = np.max(c_data[0])
    xmin = np.min(c_data[1])
    xmax = np.max(c_data[1])
    
    c_grid = np.zeros((int(ymax-ymin+1), int(xmax-xmin+1)))
    c_grid[c_data[0]-ymin, c_data[1]-xmin] = c_data[3]
    
    # Simplify: 1-10 → 1 (R), 11-20 → 2 (S)
    c_grid[(c_grid < 11) & (c_grid > 0)] = 1
    c_grid[c_grid > 10] = 2
    
    return c_grid, (ymin, ymax, xmin, xmax)


def downsample_cell_grid(c_grid, ddx=30):
    """Downsample cell grid to antibiotic grid resolution."""
    ny, nx = c_grid.shape
    ny_down = ny // ddx
    nx_down = nx // ddx
    
    c_grid_down = np.zeros((ny_down, nx_down))
    
    for i in range(ny_down):
        for j in range(nx_down):
            block = c_grid[i*ddx:(i+1)*ddx, j*ddx:(j+1)*ddx]
            nonzero = block[block > 0]
            if len(nonzero) > 0:
                counts = np.bincount(nonzero.astype(int))
                c_grid_down[i, j] = np.argmax(counts)
    
    return c_grid_down


# Find all timepoint directories
timepoint_dirs = sorted(glob.glob(str(data_dir / "*_timepoints")))

print(f"Found {len(timepoint_dirs)} simulation(s) with timepoints")
print("="*60)

for timepoint_dir in timepoint_dirs:
    timepoint_dir = Path(timepoint_dir)
    run_name_base = timepoint_dir.stem.replace('_timepoints', '')
    
    print(f"\nProcessing: {run_name_base}")
    
    # Find all timepoint files for this run
    cell_files = sorted(glob.glob(str(timepoint_dir / "*_cells.npy")))
    
    print(f"  Found {len(cell_files)} timepoints")
    
    for cell_file in cell_files:
        cell_file = Path(cell_file)
        
        # Extract timepoint from filename
        match = re.search(r't([\d.]+)h', cell_file.stem)
        if not match:
            print(f"  Warning: Could not parse timepoint from {cell_file.name}")
            continue
        
        time_str = match.group(1)
        time_hours = float(time_str)
        
        # Find corresponding antibiotic file
        antibio_file = cell_file.parent / cell_file.name.replace('_cells.npy', '_antibiotic.npy')
        
        if not antibio_file.exists():
            print(f"  Warning: Missing antibiotic file for t={time_hours}h")
            continue
        
        print(f"  Processing t = {time_hours} hours...")
        
        # Load data
        c_grid, bounds = load_cell_grid(cell_file)
        a_grid = np.load(antibio_file)
        
        # Determine display range in diffusion coordinates
        if y_min_display is not None and y_max_display is not None:
            y_start = y_min_display
            y_end = y_max_display
        else:
            y_start = 0
            y_end = a_grid.shape[0]
        
        if x_min_display is not None and x_max_display is not None:
            x_start = x_min_display
            x_end = x_max_display
        else:
            x_start = 0
            x_end = a_grid.shape[1]
        
        # Crop antibiotic grid to display range
        a_grid_display = a_grid[y_start:y_end, x_start:x_end]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # === FAST RENDERING: Use imshow for everything ===
        
        # Calculate display region in micrometers
        y_start_um = y_start * 30
        y_end_um = y_end * 30
        x_start_um = x_start * 30
        x_end_um = x_end * 30
        
        # Get cell bounds with offsets
        cell_ymin, cell_ymax, cell_xmin, cell_xmax = bounds
        r_offset = 533
        c_offset = 533
        
        # Cell positions in absolute micrometers
        cell_y_start_um = (cell_ymin // 30 + r_offset) * 30
        cell_x_start_um = (cell_xmin // 30 + c_offset) * 30
        cell_y_end_um = cell_y_start_um + c_grid.shape[0]
        cell_x_end_um = cell_x_start_um + c_grid.shape[1]
        
        # Crop cell grid to display region
        y_crop_start = max(0, y_start_um - cell_y_start_um)
        y_crop_end = min(c_grid.shape[0], y_end_um - cell_y_start_um)
        x_crop_start = max(0, x_start_um - cell_x_start_um)
        x_crop_end = min(c_grid.shape[1], x_end_um - cell_x_start_um)
        
        if y_crop_end > y_crop_start and x_crop_end > x_crop_start:
            c_grid_display = c_grid[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
            
            # Actual position of cropped cells
            cells_x_start = cell_x_start_um + x_crop_start
            cells_x_end = cell_x_start_um + x_crop_end
            cells_y_start = cell_y_start_um + y_crop_start
            cells_y_end = cell_y_start_um + y_crop_end
        else:
            c_grid_display = None
        
        # Display antibiotic field (background)
        im = ax.imshow(a_grid_display, origin='lower', interpolation='bicubic',
                      extent=[x_start_um, x_end_um, y_start_um, y_end_um],
                      cmap='Reds', vmin=0, vmax=5, alpha=0.6, rasterized=True)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Antibiotic (μg/mL)', fontsize=11)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '5'])
        
        # Overlay cells at full resolution using imshow with transparency
        if c_grid_display is not None:
            # Create RGBA overlay for cells
            cell_rgba = np.zeros((*c_grid_display.shape, 4))
            
            # Resistant cells (blue)
            r_mask = (c_grid_display == 1)
            cell_rgba[r_mask] = [0, 0, 1, 1]  # Blue, fully opaque
            
            # Sensitive cells (yellow)
            s_mask = (c_grid_display == 2)
            cell_rgba[s_mask] = [1, 1, 0, 1]  # Yellow, fully opaque
            
            # Display cells
            ax.imshow(cell_rgba, origin='lower', interpolation='nearest',
                     extent=[cells_x_start, cells_x_end, cells_y_start, cells_y_end],
                     rasterized=True)
        
        # Set axis limits
        ax.set_xlim(x_start_um, x_end_um)
        ax.set_ylim(y_start_um, y_end_um)
        
        # =====================================================================
        # FAST EDGE ANALYSIS - Max/Min Y per X coordinate
        # =====================================================================
        
        all_cells = c_grid > 0
        if np.any(all_cells):
            # Get all occupied cells
            y_coords, x_coords = np.where(all_cells)
            
            # Use numpy's group-by operation (faster than loop)
            unique_x = np.unique(x_coords)
            
            # Preallocate arrays
            n_x = len(unique_x)
            max_y = np.zeros(n_x, dtype=int)
            min_y = np.zeros(n_x, dtype=int)
            
            # For each unique x, find max and min y using searchsorted + reduceat
            # Sort by x first for efficient grouping
            sort_idx = np.argsort(x_coords)
            x_sorted = x_coords[sort_idx]
            y_sorted = y_coords[sort_idx]
            
            # Find where each unique x starts
            split_idx = np.searchsorted(x_sorted, unique_x, side='left')
            split_idx = np.append(split_idx, len(x_sorted))
            
            # Get max/min for each group
            for i in range(n_x):
                y_group = y_sorted[split_idx[i]:split_idx[i+1]]
                max_y[i] = np.max(y_group)
                min_y[i] = np.min(y_group)
            
            # Get cell types at these edge positions
            top_edge_types = c_grid[max_y, unique_x]
            bottom_edge_types = c_grid[min_y, unique_x]
            
            # Count sensitive cells (type == 2)
            n_sensitive_top = np.sum(top_edge_types == 2)
            
            # Only count bottom edge if different from top (same cell shouldn't be counted twice)
            different = (max_y != min_y)
            n_sensitive_bottom = np.sum((bottom_edge_types == 2) & different)
            
            n_sensitive_at_edge = n_sensitive_top + n_sensitive_bottom
            
            print(f"    Sensitive cells at edge: {n_sensitive_at_edge} cells")
        else:
            n_sensitive_at_edge = 0
            print(f"    No cells found")
        
        # Title with time
        ax.set_title(f't = {time_hours:.1f} hours', fontsize=16, fontweight='bold')
        ax.set_xlabel('x (μm)', fontsize=12)
        ax.set_ylabel('y (μm)', fontsize=12)
        
        # Simple legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='High A'),
            Patch(facecolor='blue', label='R'),
            Patch(facecolor='yellow', label='S'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.7)
        
        # Add stats box
        stats_text = f"t = {time_hours:.1f}h\nA: {a_grid_display.min():.1f}-{a_grid_display.max():.1f}\nEdge S: {n_sensitive_at_edge}"
        ax.text(0.02, 1.2, stats_text, transform=ax.transAxes,
#        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save with full parameter set + timepoint in filename
        save_name = f"{run_name_base}_t{time_str}h.png"
        save_path = output_dir / save_name
        
        # Save with high quality
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   format='png', 
                   pil_kwargs={'compress_level': 6})  # Better compression
        plt.close('all')  # Close all figures to free memory
        
        print(f"    Saved: {save_name}")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"Display range: y=[{y_min_display}, {y_max_display}], x=[{x_min_display}, {x_max_display}]")
print("\nTo change display range, edit the CONFIGURATION section at the top of this script.")
print("="*60)
