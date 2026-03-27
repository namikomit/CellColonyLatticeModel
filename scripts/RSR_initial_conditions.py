import numpy as np
import numpy.typing as npt

def create_RSR_stripe_geometry(c_grid: npt.NDArray, 
                                L: int, 
                                w: int, 
                                density: float = 0.9) -> tuple[npt.NDArray, int]:
    """
    Creates a rectangular initial condition with R-S-R stripe geometry.
    
    Two outer regions filled with resistant cells, middle stripe with sensitive cells.
    
    Args:
        c_grid: Empty cell grid (2D array)
        L: Height of the rectangular region [μm]
        w: Width of the middle sensitive stripe [μm]  
        density: Fraction of sites to fill with cells (default 0.9)
        
    Returns:
        c_grid: Cell grid with R-S-R geometry
        total_cells: Total number of cells placed
        
    Geometry:
        |<---- L ---->|
        
        +-------------+  ---
        | Resistant R |   ^
        +-------------+   |
        | Sensitive S |   w (middle stripe width)
        +-------------+   |
        | Resistant R |   v
        +-------------+  ---
        
    The resistant regions on top and bottom have equal width.
    Total vertical extent = L
    Middle sensitive stripe has width = w
    Each resistant stripe has width = (L - w) / 2
    
    Centered in the grid.
    """
    
    dim_y, dim_x = c_grid.shape
    
    # Calculate stripe widths
    resistant_stripe_width = int((L - w) / 2)
    
    # Check that geometry fits
    assert L <= dim_y, f"L={L} exceeds grid height {dim_y}"
    assert w < L, f"Sensitive width w={w} must be less than total height L={L}"
    assert resistant_stripe_width > 0, f"Resistant stripe width {resistant_stripe_width} must be positive"
    
    # Calculate vertical positions (centered in grid)
    y_center = dim_y // 2
    y_start = y_center - L // 2
    y_end = y_start + L
    
    # Define the three horizontal stripes
    y_bottom_R_start = y_start
    y_bottom_R_end = y_start + resistant_stripe_width
    
    y_S_start = y_bottom_R_end
    y_S_end = y_S_start + w
    
    y_top_R_start = y_S_end
    y_top_R_end = y_end
    
    # Horizontally: fill entire width of grid
    x_start = 0
    x_end = dim_x
    
    # Get all positions in each stripe
    y_all, x_all = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
    
    # Bottom resistant stripe
    mask_bottom_R = (y_all >= y_bottom_R_start) & (y_all < y_bottom_R_end) & \
                    (x_all >= x_start) & (x_all < x_end)
    positions_bottom_R = np.column_stack(np.where(mask_bottom_R))
    
    # Sensitive stripe  
    mask_S = (y_all >= y_S_start) & (y_all < y_S_end) & \
             (x_all >= x_start) & (x_all < x_end)
    positions_S = np.column_stack(np.where(mask_S))
    
    # Top resistant stripe
    mask_top_R = (y_all >= y_top_R_start) & (y_all < y_top_R_end) & \
                 (x_all >= x_start) & (x_all < x_end)
    positions_top_R = np.column_stack(np.where(mask_top_R))
    
    # Sample positions according to density
    n_bottom_R = int(len(positions_bottom_R) * density)
    n_S = int(len(positions_S) * density)
    n_top_R = int(len(positions_top_R) * density)
    
    chosen_bottom_R = positions_bottom_R[np.random.choice(len(positions_bottom_R), n_bottom_R, replace=False)]
    chosen_S = positions_S[np.random.choice(len(positions_S), n_S, replace=False)]
    chosen_top_R = positions_top_R[np.random.choice(len(positions_top_R), n_top_R, replace=False)]
    
    # Place cells
    # Resistant: random age 1-10
    for y, x in chosen_bottom_R:
        c_grid[y, x] = np.random.randint(1, 11)
    
    for y, x in chosen_top_R:
        c_grid[y, x] = np.random.randint(1, 11)
    
    # Sensitive: random age 11-20
    for y, x in chosen_S:
        c_grid[y, x] = np.random.randint(11, 21)
    
    total_cells = n_bottom_R + n_S + n_top_R
    
    print(f"Created R-S-R geometry:")
    print(f"  Total height L = {L} μm")
    print(f"  Sensitive stripe width w = {w} μm")
    print(f"  Each resistant stripe width = {resistant_stripe_width} μm")
    print(f"  Bottom R: {n_bottom_R} cells")
    print(f"  Sensitive: {n_S} cells")
    print(f"  Top R: {n_top_R} cells")
    print(f"  Total: {total_cells} cells")
    print(f"  Y range: {y_start} to {y_end}")
    
    return c_grid, total_cells


def create_RSR_vertical_stripe_geometry(c_grid: npt.NDArray, 
                                         L: int, 
                                         w: int,
                                         total_width: int,
                                         density: float = 0.9) -> tuple[npt.NDArray, int]:
    """
    Creates a rectangular initial condition with vertical R-S-R stripe geometry.
    
    Two outer regions (left and right) filled with resistant cells, 
    middle vertical stripe with sensitive cells.
    
    Args:
        c_grid: Empty cell grid (2D array)
        L: Height of the rectangular region [μm]
        w: Width of the middle sensitive stripe [μm]
        total_width: Total horizontal width of the region [μm]
        density: Fraction of sites to fill with cells (default 0.9)
        
    Returns:
        c_grid: Cell grid with vertical R-S-R geometry
        total_cells: Total number of cells placed
        
    Geometry:
        
        +---+-------+---+
        | R |   S   | R |
        | e |   e   | e |
        | s |   n   | s |
        | i |   s   | i |
        | s |   i   | s |
        | t |   t   | t |
        | a |   i   | a |
        | n |   v   | n |
        | t |   e   | t |
        +---+-------+---+
        
        |<----- total_width ----->|
        |<w/2><-- w --><w/2>|
           ^              ^
           L (height)     L (height)
    
    Each resistant region has width (total_width - w) / 2
    Middle sensitive stripe has width w
    All regions have height L
    """
    
    dim_y, dim_x = c_grid.shape
    
    # Calculate stripe widths
    resistant_stripe_width = int((total_width - w) / 2)
    
    # Check that geometry fits
    assert total_width <= dim_x, f"total_width={total_width} exceeds grid width {dim_x}"
    assert L <= dim_y, f"L={L} exceeds grid height {dim_y}"
    assert w < total_width, f"Sensitive width w={w} must be less than total_width={total_width}"
    assert resistant_stripe_width > 0, f"Resistant stripe width {resistant_stripe_width} must be positive"
    
    # Calculate positions (centered in grid)
    y_center = dim_y // 2
    y_start = y_center - L // 2
    y_end = y_start + L
    
    x_center = dim_x // 2
    x_start = x_center - total_width // 2
    x_end = x_start + total_width
    
    # Define the three vertical stripes
    x_left_R_start = x_start
    x_left_R_end = x_start + resistant_stripe_width
    
    x_S_start = x_left_R_end
    x_S_end = x_S_start + w
    
    x_right_R_start = x_S_end
    x_right_R_end = x_end
    
    # Get all positions in each stripe
    y_all, x_all = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
    
    # Left resistant stripe
    mask_left_R = (y_all >= y_start) & (y_all < y_end) & \
                  (x_all >= x_left_R_start) & (x_all < x_left_R_end)
    positions_left_R = np.column_stack(np.where(mask_left_R))
    
    # Sensitive stripe  
    mask_S = (y_all >= y_start) & (y_all < y_end) & \
             (x_all >= x_S_start) & (x_all < x_S_end)
    positions_S = np.column_stack(np.where(mask_S))
    
    # Right resistant stripe
    mask_right_R = (y_all >= y_start) & (y_all < y_end) & \
                   (x_all >= x_right_R_start) & (x_all < x_right_R_end)
    positions_right_R = np.column_stack(np.where(mask_right_R))
    
    # Sample positions according to density
    n_left_R = int(len(positions_left_R) * density)
    n_S = int(len(positions_S) * density)
    n_right_R = int(len(positions_right_R) * density)
    
    chosen_left_R = positions_left_R[np.random.choice(len(positions_left_R), n_left_R, replace=False)]
    chosen_S = positions_S[np.random.choice(len(positions_S), n_S, replace=False)]
    chosen_right_R = positions_right_R[np.random.choice(len(positions_right_R), n_right_R, replace=False)]
    
    # Place cells
    # Resistant: random age 1-10
    for y, x in chosen_left_R:
        c_grid[y, x] = np.random.randint(1, 11)
    
    for y, x in chosen_right_R:
        c_grid[y, x] = np.random.randint(1, 11)
    
    # Sensitive: random age 11-20
    for y, x in chosen_S:
        c_grid[y, x] = np.random.randint(11, 21)
    
    total_cells = n_left_R + n_S + n_right_R
    
    print(f"Created vertical R-S-R geometry:")
    print(f"  Total height L = {L} μm")
    print(f"  Total width = {total_width} μm")
    print(f"  Sensitive stripe width w = {w} μm")
    print(f"  Each resistant stripe width = {resistant_stripe_width} μm")
    print(f"  Left R: {n_left_R} cells")
    print(f"  Sensitive: {n_S} cells")
    print(f"  Right R: {n_right_R} cells")
    print(f"  Total: {total_cells} cells")
    print(f"  Y range: {y_start} to {y_end}")
    print(f"  X range: {x_start} to {x_end}")
    
    return c_grid, total_cells
