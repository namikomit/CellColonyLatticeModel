"""
Create Panel Figure using PIL (high resolution, then downsample to 300 DPI)

Panel layout:
- Top 3 rows: Time series (0h, 1h, 2h, 8h) for w=200 at A=2, 3, 5
- Bottom 3 rows: Width comparison (w=100, 400, 800) at t=8h for A=2, 3, 5
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy.ndimage import zoom

# Configuration
data_dir = Path("/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/data/RSR_timeseries")  # <-- update to your local data directory
output_dir = Path("/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/results")  # <-- update to your local output directory
output_dir.mkdir(exist_ok=True)

# Display range (in diffusion grid coordinates)
# For time series (0h, 1h, 2h): centered smaller region
y_min_display_timeseries = 750
y_max_display_timeseries = 850
x_min_display_timeseries = 750
x_max_display_timeseries = 850

# For width comparison (8h): rectangular - top half, centered in x
y_min_display_width = 800
y_max_display_width = 895  # Taller: 200 units = 6000 μm height
x_min_display_width = 780
x_max_display_width = 820   # Narrower: 100 units = 3000 μm width

# Full page width at 300 DPI
dpi = 300
page_width_inches = 4.25  # Half page width
final_width_px = int(page_width_inches * dpi)  # 1275 pixels

# Panel layout: 4 columns, 6 rows
n_cols = 4
n_rows = 6
border = 15
gap = 5

# Calculate panel sizes
# Time series: square panels
panel_width_ts = (final_width_px - 2*border - 2*gap) // 3  # 3 columns
panel_height_ts = panel_width_ts  # Square

# Width comparison: rectangular panels (2:1 aspect ratio for 6000x3000 μm)
panel_width_wc = (final_width_px - 2*border - 3*gap) // 4  # 4 columns  
display_aspect_wc = (y_max_display_width - y_min_display_width) / (x_max_display_width - x_min_display_width)
panel_height_wc = int(panel_width_wc * display_aspect_wc)

total_height = 2*border + 3*panel_height_ts + 3*panel_height_wc + 5*gap

# Parameters
L_height = 800
g_nonnorm = 0.025
ka = 2.35

# Timepoints and widths
timepoints_series = [0.0, 1.0, 2.0]  # Only 0h, 1h, 2h (magnified)
widths_comparison = [100, 200, 400, 800]  # 4 widths at 8h
antibiotic_levels = [2.0, 3.0, 5.0]

print(f"Creating figure: {final_width_px} x {total_height} pixels")
print(f"Time series panel size: {panel_width_ts} x {panel_height_ts} pixels")
print(f"Width comparison panel size: {panel_width_wc} x {panel_height_wc} pixels")

def calculate_display_range(c_grid, bounds, center_x=True, top_half=False):
    """Calculate display range to show colony properly.
    
    Args:
        c_grid: Cell grid
        bounds: (ymin, ymax, xmin, xmax) in 1μm coordinates
        center_x: If True, center horizontally on colony
        top_half: If True, show top half; otherwise show full height
    
    Returns:
        (y_min, y_max, x_min, x_max) in diffusion grid coordinates
    """
    cell_ymin, cell_ymax, cell_xmin, cell_xmax = bounds
    r_offset = 533
    c_offset = 533
    
    # Find actual occupied region
    occupied = c_grid > 0
    if not np.any(occupied):
        # No cells, return default
        return (750, 850, 750, 850)
    
    y_coords, x_coords = np.where(occupied)
    
    # Cell bounds in micrometers (relative to cell grid origin)
    cell_y_min_um = np.min(y_coords)
    cell_y_max_um = np.max(y_coords)
    cell_x_min_um = np.min(x_coords)
    cell_x_max_um = np.max(x_coords)
    
    # Convert to diffusion grid coordinates
    diff_y_min = (cell_ymin + cell_y_min_um) // 30 + r_offset
    diff_y_max = (cell_ymin + cell_y_max_um) // 30 + r_offset
    diff_x_min = (cell_xmin + cell_x_min_um) // 30 + c_offset
    diff_x_max = (cell_xmin + cell_x_max_um) // 30 + c_offset
    
    # Calculate center
    diff_y_center = (diff_y_min + diff_y_max) // 2
    diff_x_center = (diff_x_min + diff_x_max) // 2
    
    if top_half:
        # Show top half: from center to max, centered in x
        window_height = 200  # diffusion units
        y_display_min = diff_y_center - 20  # Slight overlap with center
        y_display_max = y_display_min + window_height
    else:
        # Show full height, centered
        colony_height = diff_y_max - diff_y_min
        window_height = int(colony_height * 1.2)  # 20% margin
        y_display_min = diff_y_center - window_height // 2
        y_display_max = y_display_min + window_height
    
    if center_x:
        # Center horizontally on colony
        window_width = 200  # diffusion units
        x_display_min = diff_x_center - window_width // 2
        x_display_max = x_display_min + window_width
    else:
        x_display_min = diff_x_min - 50
        x_display_max = diff_x_max + 50
    
    return (int(y_display_min), int(y_display_max), int(x_display_min), int(x_display_max))

def load_cell_grid(filepath):
    """Load sparse cell grid."""
    c_data = np.load(filepath, allow_pickle=True)
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

def create_colormap_rgb():
    """Create RGB colormap for antibiotic (white to red)."""
    # Returns function that maps 0-5 to RGB
    def map_antibiotic(value):
        # Normalize 0-5 to 0-1
        norm = np.clip(value / 5.0, 0, 1)
        # White (low) to Red (high)
        r = int(255)
        g = int(255 * (1 - norm))
        b = int(255 * (1 - norm))
        return (r, g, b)
    return map_antibiotic

def render_panel_pil(c_grid, a_grid, bounds, y_start, y_end, x_start, x_end, width, height):
    """Render at proper absolute scale - 1 pixel per cell (1 μm), then downsample."""
    
    # Get cell bounds with offsets
    cell_ymin, cell_ymax, cell_xmin, cell_xmax = bounds
    r_offset = 533
    c_offset = 533
    
    # Display region size in micrometers
    display_width_um = (x_end - x_start) * 30  # e.g., 3000 μm
    display_height_um = (y_end - y_start) * 30
    
    # Create image at micrometers scale (1 pixel = 1 μm)
    hires_width = display_width_um
    hires_height = display_height_um
    
    # Create high-res numpy array
    img_hires = np.zeros((hires_height, hires_width, 3), dtype=np.uint8)
    
    # 1. Render antibiotic background (30 μm blocks)
    a_crop = a_grid[y_start:y_end, x_start:x_end]
    
    for i in range(a_crop.shape[0]):
        for j in range(a_crop.shape[1]):
            a_val = a_crop[i, j]
            norm = np.clip(a_val / 5.0, 0, 1)
            r = 255
            g = int(255 * (1 - norm))
            b = int(255 * (1 - norm))
            
            # Fill 30×30 μm block
            y_start_px = i * 30
            y_end_px = min((i + 1) * 30, hires_height)
            x_start_px = j * 30
            x_end_px = min((j + 1) * 30, hires_width)
            
            img_hires[y_start_px:y_end_px, x_start_px:x_end_px] = [r, g, b]
    
    # 2. Overlay cells at 1 μm resolution
    # Cell grid origin in diffusion coordinates
    cell_origin_diff_y = cell_ymin // 30 + r_offset
    cell_origin_diff_x = cell_xmin // 30 + c_offset
    
    # Display window in diffusion coordinates: y_start to y_end, x_start to x_end
    # Cell origin to display origin offset (in diffusion units)
    offset_diff_y = cell_origin_diff_y - y_start
    offset_diff_x = cell_origin_diff_x - x_start
    
    # Convert to micrometers
    offset_um_y = offset_diff_y * 30
    offset_um_x = offset_diff_x * 30
    
    # Where in the display image do cells start?
    cell_start_y = offset_um_y
    cell_start_x = offset_um_x
    
    # How much of the cell grid is visible?
    cell_end_y = min(cell_start_y + c_grid.shape[0], hires_height)
    cell_end_x = min(cell_start_x + c_grid.shape[1], hires_width)
    
    # Which part of cell grid to use
    if cell_start_y < 0:
        c_start_y = -cell_start_y
        img_start_y = 0
    else:
        c_start_y = 0
        img_start_y = cell_start_y
    
    if cell_start_x < 0:
        c_start_x = -cell_start_x
        img_start_x = 0
    else:
        c_start_x = 0
        img_start_x = cell_start_x
    
    c_end_y = c_start_y + (cell_end_y - cell_start_y)
    c_end_x = c_start_x + (cell_end_x - cell_start_x)
    
    # Bounds check
    c_end_y = min(c_end_y, c_grid.shape[0])
    c_end_x = min(c_end_x, c_grid.shape[1])
    img_end_y = img_start_y + (c_end_y - c_start_y)
    img_end_x = img_start_x + (c_end_x - c_start_x)
    
    if c_end_y > c_start_y and c_end_x > c_start_x and img_end_y > 0 and img_end_x > 0:
        c_visible = c_grid[c_start_y:c_end_y, c_start_x:c_end_x]
        
        # Ensure exact dimension match
        img_region = img_hires[img_start_y:img_end_y, img_start_x:img_end_x]
        
        # Truncate if needed
        min_h = min(c_visible.shape[0], img_region.shape[0])
        min_w = min(c_visible.shape[1], img_region.shape[1])
        
        c_visible = c_visible[:min_h, :min_w]
        img_region = img_region[:min_h, :min_w]
        
        # Apply cell colors
        resistant = (c_visible == 1)
        sensitive = (c_visible == 2)
        
        if np.any(resistant) or np.any(sensitive):
            img_region[resistant] = [0, 0, 255]
            img_region[sensitive] = [255, 255, 0]
            img_hires[img_start_y:img_start_y+min_h, img_start_x:img_start_x+min_w] = img_region
    
    # 3. Downsample to panel size
    img_pil = Image.fromarray(img_hires, mode='RGB')
    img_panel = img_pil.resize((width, height), Image.LANCZOS)
    
    return img_panel

# Create full figure
fig_img = Image.new('RGB', (final_width_px, total_height), color=(255, 255, 255))
draw = ImageDraw.Draw(fig_img)

# Try to load a font
try:
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

print("\nRendering panels...")

# Top 3 rows: Time series for w=200
for row, a0 in enumerate(antibiotic_levels):
    print(f"\nRow {row+1}: Time series w=200, A={a0}")
    
    for col, t in enumerate(timepoints_series):
        print(f"  t={t}h", end=' ', flush=True)
        
        # Load data
        run_name = f"RSR_stripe_w200_L{L_height}_A{a0}_g{g_nonnorm}_ka{ka}"
        timepoint_dir = data_dir / f"{run_name}_timepoints"
        
        cell_file = timepoint_dir / f"{run_name}_t{t}h_cells.npy"
        antibio_file = timepoint_dir / f"{run_name}_t{t}h_antibiotic.npy"
        
        if not cell_file.exists():
            print(f" SKIP (not found)")
            continue
        
        c_grid, bounds = load_cell_grid(cell_file)
        a_grid = np.load(antibio_file)
        
        # Render panel with magnified range
        panel = render_panel_pil(c_grid, a_grid, bounds,
                                 y_min_display_timeseries, y_max_display_timeseries,
                                 x_min_display_timeseries, x_max_display_timeseries,
                                 panel_width_ts, panel_height_ts)
        
        # Paste into figure
        x_pos = border + col * (panel_width_ts + gap)
        y_pos = border + row * (panel_height_ts + gap)
        fig_img.paste(panel, (x_pos, y_pos))
        
        print(" ✓")

# Bottom 3 rows: Width comparison at t=8h
for row, a0 in enumerate(antibiotic_levels):
    print(f"\nRow {row+4}: Width comparison t=8h, A={a0}")
    
    for col, w in enumerate(widths_comparison):
        print(f"  w={w}", end=' ', flush=True)
        
        # Load data
        run_name = f"RSR_stripe_w{w}_L{L_height}_A{a0}_g{g_nonnorm}_ka{ka}"
        timepoint_dir = data_dir / f"{run_name}_timepoints"
        
        cell_file = timepoint_dir / f"{run_name}_t8.0h_cells.npy"
        antibio_file = timepoint_dir / f"{run_name}_t8.0h_antibiotic.npy"
        
        if not cell_file.exists():
            print(f" SKIP (not found)")
            continue
        
        c_grid, bounds = load_cell_grid(cell_file)
        a_grid = np.load(antibio_file)
        
        # Render panel with fixed range
        panel = render_panel_pil(c_grid, a_grid, bounds,
                                 y_min_display_width, y_max_display_width,
                                 x_min_display_width, x_max_display_width,
                                 panel_width_wc, panel_height_wc)
        
        # Paste into figure
        x_pos = border + col * (panel_width_wc + gap)
        y_pos = border + 3*panel_height_ts + 3*gap + row * (panel_height_wc + gap)
        fig_img.paste(panel, (x_pos, y_pos))
        
        print(" ✓")

# Add colorbar in 4th column, top half (to avoid overlap with panels)
print("\nAdding colorbar...")
cbar_x = border + 3 * (panel_width_ts + gap) + panel_width_ts//2 - 15
cbar_y = border + 1 * (panel_height_ts + gap) + 50
cbar_width = 30
cbar_height = 200

for i in range(cbar_height):
    val = 5.0 * (1 - i / cbar_height)  # 5 at top, 0 at bottom
    norm = val / 5.0
    r = 255
    g = int(255 * (1 - norm))
    b = int(255 * (1 - norm))
    draw.rectangle([cbar_x, cbar_y + i, cbar_x + cbar_width, cbar_y + i + 1],
                   fill=(r, g, b))

# Colorbar labels
draw.text((cbar_x + cbar_width + 5, cbar_y), '5', fill=(0, 0, 0), font=font_small, anchor='lm')
draw.text((cbar_x + cbar_width + 5, cbar_y + cbar_height), '0', fill=(0, 0, 0), font=font_small, anchor='lm')
draw.text((cbar_x + cbar_width//2, cbar_y + cbar_height + 10), 'A (μg/mL)',
         fill=(0, 0, 0), font=font_small, anchor='mt')

# Save
output_file = output_dir / "panel_figure_timeseries_width_comparison.png"
fig_img.save(output_file, dpi=(dpi, dpi))

print(f"\n{'='*60}")
print(f"Panel figure saved: {output_file}")
print(f"Size: {final_width_px} x {total_height} pixels ({page_width_inches:.1f} x {total_height/dpi:.1f} inches at {dpi} DPI)")
print(f"{'='*60}")
