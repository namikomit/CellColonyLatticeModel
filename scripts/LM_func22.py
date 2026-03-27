import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import spmatrix
from tqdm import tqdm
import numpy.typing as npt
import numba as nb
import ray
import os


""" INITIAL CONDITIONS AND SETUP"""

def create_grids_differnt_dim(ddim_r: int, ddim_c: int, cdim_r: int, cdim_c: int, A_init: float, N_init: float, n_layers: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    
    r_offset = ddim_r // 3
    c_offset = ddim_c // 3
    # c_grid = np.zeros((cdim_r, cdim_c, n_layers), dtype=np.int8)
    c_grid = np.zeros((cdim_r // 3, cdim_c // 3, n_layers), dtype=np.int8)
    a_grid = np.zeros((ddim_r, ddim_c), dtype=float) + A_init
    n_grid = np.zeros((ddim_r, ddim_c), dtype=float) + N_init
    # snap_grid = np.zeros((cdim_r, cdim_c, no_of_snapshots), dtype=np.int8)
    
    return c_grid, a_grid, n_grid, r_offset, c_offset


def coffee_ring(c_grid: npt.ArrayLike, cell_cnt: int, resistant_fraction: float, r_inner: int, r_outer: int, frac_within: float) -> tuple[npt.NDArray, float]:
    """
    Creates a coffee ring initial configuration if r_inner > 0. For r_inner = 0 it's a drop

    Args:
        c_grid (npt.ArrayLike): _description_
        cell_cnt (int): _description_
        resistant_fraction (float): _description_
        r_inner (int): _description_
        r_outer (int): _description_

    Returns:
        npt.NDArray: cell grid with desiered initial condition
        float: density of ring or drop
    """
    # gives a fraction of R or total cnt depending on data type
    if type(resistant_fraction) == float:
        r_cnt = int(cell_cnt * resistant_fraction)
    else:
        r_cnt = resistant_fraction
    s_cnt = cell_cnt - r_cnt
    
    # get all coordinates
    y, x = np.where(c_grid == 0)
    
    center = np.array(c_grid.shape, dtype=np.int64) / 2
    
    dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    dist = dist.reshape((c_grid.shape))
    y_possible, x_possible = np.where((dist >= r_inner) & (dist <= r_outer))
    
    assert len(y_possible) >= int(cell_cnt * (1-frac_within))
    
    chosen_idx = np.random.choice(len(y_possible), int(cell_cnt * (1-frac_within)), replace=False) # pos of cells found within CRE
    
    r_cnt_cre = int(r_cnt * (1-frac_within))
    y_r = y_possible[chosen_idx[:r_cnt_cre]]
    x_r = x_possible[chosen_idx[:r_cnt_cre]]
    y_s = y_possible[chosen_idx[r_cnt_cre:]]
    x_s = x_possible[chosen_idx[r_cnt_cre:]]
    
    c_grid[y_r, x_r] = np.random.randint(1, 11, len(y_r))
    c_grid[y_s, x_s] = np.random.randint(11, 21, len(y_s))
    
    # cells where r < r_inner
    y_possible, x_possible = np.where(dist < r_inner)
    
    assert len(y_possible) >= int(cell_cnt * frac_within)
    
    chosen_idx = np.random.choice(len(y_possible), int(cell_cnt * frac_within), replace=False) # pos of cells found within CRE
    r_cnt_nocre = int(r_cnt * frac_within)
    
    y_r = y_possible[chosen_idx[:r_cnt_nocre]]
    x_r = x_possible[chosen_idx[:r_cnt_nocre]]
    y_s = y_possible[chosen_idx[r_cnt_nocre:]]
    x_s = x_possible[chosen_idx[r_cnt_nocre:]]
    
    c_grid[y_r, x_r] = np.random.randint(1, 11, len(y_r))
    c_grid[y_s, x_s] = np.random.randint(11, 21, len(y_s))
    
    
    
    
    density = cell_cnt / len(y_possible)
    
    return c_grid, density


def diffuse_drops2(cell_grid: npt.NDArray, cell_cnt: int, r_i: int, r_o: int, drop_separation: int, frac_within: float) -> npt.NDArray:
    dist_to_center = int(drop_separation / 2 + r_o)
    y_mid = int(cell_grid.shape[0] / 2)
    y, x = np.where(cell_grid == 0) # get coordinates
    
    for i in range(0, 2):
        
        # center = np.array(c_grid.shape, dtype=np.int64) / 2
        
        dist_r = np.sqrt((y - y_mid)**2 + (x - (cell_grid.shape[1] / 2 - dist_to_center))**2) # Res
        dist_s = np.sqrt((y - y_mid)**2 + (x - (cell_grid.shape[1] / 2 + dist_to_center))**2) # Res
        
        dist_r = dist_r.reshape((cell_grid.shape))
        dist_s = dist_s.reshape((cell_grid.shape))
        y_possible_r, x_possible_r = np.where((dist_r >= r_i) & (dist_r <= r_o))
        y_possible_s, x_possible_s = np.where((dist_s >= r_i) & (dist_s <= r_o))
        
        assert len(y_possible_r) >= cell_cnt / (1 - frac_within) # same for both
        
        chosen_idx_r = np.random.choice(len(y_possible_r), int(cell_cnt * (1 - frac_within)), replace=False) # comes back shuffled
        chosen_idx_s = np.random.choice(len(y_possible_s), int(cell_cnt * (1 - frac_within)), replace=False) # comes back shuffled
        y_r = y_possible_r[chosen_idx_r]
        x_r = x_possible_r[chosen_idx_r]
        y_s = y_possible_s[chosen_idx_s]
        x_s = x_possible_s[chosen_idx_s]
        
        cell_grid[y_r, x_r] = np.random.randint(1, 11, int(cell_cnt * (1 - frac_within)))
        cell_grid[y_s, x_s] = np.random.randint(11, 21, int(cell_cnt * (1 - frac_within)))
        
        density = cell_cnt / len(y_possible_r)
        
        # now place cells in middle region
        y_possible_r, x_possible_r = np.where((dist_r <= r_i))
        y_possible_s, x_possible_s = np.where((dist_s <= r_i))
        
        chosen_idx_r = np.random.choice(len(y_possible_r), int(cell_cnt * (frac_within)), replace=False) # comes back shuffled
        chosen_idx_s = np.random.choice(len(y_possible_s), int(cell_cnt * (frac_within)), replace=False) # comes back shuffled
        y_r = y_possible_r[chosen_idx_r]
        x_r = x_possible_r[chosen_idx_r]
        y_s = y_possible_s[chosen_idx_s]
        x_s = x_possible_s[chosen_idx_s]
        
        cell_grid[y_r, x_r] = np.random.randint(1, 11, int(cell_cnt * (frac_within)))
        cell_grid[y_s, x_s] = np.random.randint(11, 21, int(cell_cnt * (frac_within)))
        

    return cell_grid, density

def cCRE(lattice, ri, ro, initial_count, c, inner_frac, d):

    rows, cols = lattice.shape
    
    for i in range(2):
    
        y0 = rows // 2
        if i == 0:
            x0 = cols // 2 - d // 2 - ro
        else:
            x0 = cols //2 + d//2 + ro
            
        cells_placed = 0
        
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        annulus_mask = (distance >= ri) & (distance <= ro)
        
        annulus_positions = np.column_stack(np.where(annulus_mask))
        
        np.random.shuffle(annulus_positions)
        
        while cells_placed < int(initial_count * (1-inner_frac)):
            print(cells_placed)
            # If c = 0, cells are randomly distributed without clustering
            if c == 0:
                # Place cells in random positions in the annulus
                for pos in annulus_positions:
                    if cells_placed >= initial_count:
                        break
                    if lattice[pos[0], pos[1]] == 0:  # Check if cell is empty
                        lattice[pos[0], pos[1]] = np.random.randint(1, 11) + 10 * i  # Place cell
                        cells_placed += 1
            else:
                # For c > 0, we generate clusters
                cluster_size = int(np.random.uniform(1, c))
                for _ in range(cluster_size):
                    if cells_placed >= initial_count:
                        break
                    pos_idx = np.random.choice(len(annulus_positions))
                    y, x = annulus_positions[pos_idx]
                    
                    # Place the initial cell in the cluster
                    if lattice[y, x] == 0:
                        lattice[y, x] = np.random.randint(1, 11) + 10 * i
                        cells_placed += 1

                    # Generate nearby positions for clustering effect
                    for _ in range(cluster_size - 1):
                        if cells_placed >= initial_count:
                            break
                        # Find neighbors within one cell distance
                        neighbors = [
                            (y + dy, x + dx)
                            for dy in range(-1, 2) for dx in range(-1, 2)
                            if (0 <= y + dy < rows and 0 <= x + dx < cols)
                        ]
                        # Randomly select a neighboring position
                        ny, nx = neighbors[np.random.choice(len(neighbors))]
                        
                        # Place cell if empty
                        if lattice[ny, nx] == 0 and (ri <= np.sqrt((ny - y0)**2 + (nx - x0)**2) <= ro):
                            lattice[ny, nx] = np.random.randint(1, 11) + 10 * i
                            cells_placed += 1
        
        
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        annulus_mask = (distance < ri)
        
        # Find all possible positions within the annulus
        annulus_positions = np.column_stack(np.where(annulus_mask))
        
        # Shuffle positions for random access when c = 0
        np.random.shuffle(annulus_positions)
            
        cells_placed = 0                    
        while cells_placed < int(initial_count *inner_frac):
            print(cells_placed)
            # If c = 0, cells are randomly distributed without clustering
            if c == 0:
                # Place cells in random positions in the annulus
                for pos in annulus_positions:
                    if cells_placed >= initial_count:
                        break
                    if lattice[pos[0], pos[1]] == 0:  # Check if cell is empty
                        lattice[pos[0], pos[1]] = np.random.randint(1, 11) + 10 * i  # Place cell
                        cells_placed += 1
            else:
                # For c > 0, we generate clusters
                cluster_size = int(np.random.uniform(1, c))
                for _ in range(cluster_size):
                    if cells_placed >= initial_count:
                        break
                    pos_idx = np.random.choice(len(annulus_positions))
                    y, x = annulus_positions[pos_idx]
                    
                    # Place the initial cell in the cluster
                    if lattice[y, x] == 0:
                        lattice[y, x] = np.random.randint(1, 11) + 10 * i
                        cells_placed += 1

                    # Generate nearby positions for clustering effect
                    for _ in range(cluster_size - 1):
                        if cells_placed >= initial_count:
                            break
                        # Find neighbors within one cell distance
                        neighbors = [
                            (y + dy, x + dx)
                            for dy in range(-1, 2) for dx in range(-1, 2)
                            if (0 <= y + dy < rows and 0 <= x + dx < cols)
                        ]
                        # Randomly select a neighboring position
                        ny, nx = neighbors[np.random.choice(len(neighbors))]
                        
                        # Place cell if empty
                        if lattice[ny, nx] == 0 and (ri > np.sqrt((ny - y0)**2 + (nx - x0)**2)):
                            lattice[ny, nx] = np.random.randint(1, 11) + 10 * i
                            cells_placed += 1
    

    return lattice

def diffuse_drops(cell_grid: npt.NDArray, cell_cnt: int, r_i: int, r_o: int, drop_separation: int) -> npt.NDArray:
    dist_to_center = int(drop_separation / 2 + r_o)
    y_mid = int(cell_grid.shape[0] / 2)
    y, x = np.where(cell_grid == 0) # get coordinates
    
    for i in range(0, 2):
        
        # center = np.array(c_grid.shape, dtype=np.int64) / 2
        
        dist_r = np.sqrt((y - y_mid)**2 + (x - (cell_grid.shape[1] / 2 - dist_to_center))**2) # Res
        dist_s = np.sqrt((y - y_mid)**2 + (x - (cell_grid.shape[1] / 2 + dist_to_center))**2) # Res
        
        dist_r = dist_r.reshape((cell_grid.shape))
        dist_s = dist_s.reshape((cell_grid.shape))
        y_possible_r, x_possible_r = np.where((dist_r >= r_i) & (dist_r <= r_o))
        y_possible_s, x_possible_s = np.where((dist_s >= r_i) & (dist_s <= r_o))
        
        assert len(y_possible_r) >= cell_cnt # same for both
        
        chosen_idx_r = np.random.choice(len(y_possible_r), cell_cnt, replace=False) # comes back shuffled
        chosen_idx_s = np.random.choice(len(y_possible_s), cell_cnt, replace=False) # comes back shuffled
        y_r = y_possible_r[chosen_idx_r]
        x_r = x_possible_r[chosen_idx_r]
        y_s = y_possible_s[chosen_idx_s]
        x_s = x_possible_s[chosen_idx_s]
        
        cell_grid[y_r, x_r] = np.random.randint(1, 11, cell_cnt)
        cell_grid[y_s, x_s] = np.random.randint(11, 21, cell_cnt)
        
        density = cell_cnt / len(y_possible_r)

    return cell_grid, density


def boundary_radius(c_grid: npt.ArrayLike, initial_cell_cnt: int, distance: int, r_outer: int, r_inner: int = 0) -> npt.NDArray:
    """creates bruh

    Args:
        c_grid (npt.ArrayLike): empty 2d numpy array
        density (float): density of cells in both drops. Put to zero in order to use initial cell count
        initial_cell_cnt (int): no of initial cells in each colony
        distance (int): horizontal distance between edges of drops
        r_outer (int): outer radius of drops
        r_inner (int): inner radius of drops (CRE). Default 0 for homogenous drop

    Returns:
        npt.NDArray: cell grid with desired initial conditions
    """
    
    dim_y, dim_x = c_grid.shape

    r_furthest = r_outer
    s_furthest = r_furthest + distance

    r_center = [int(dim_y/2), int(-r_outer + r_furthest)]
    s_center = [int(dim_y/2), int(s_furthest + r_outer)]

    row = np.array([i // dim_x for i in range(dim_x*dim_y)])
    column = np.array([i % dim_x for i in range(dim_x*dim_y)])

    dist_to_center_r = np.sqrt((float(r_center[0]) - row)**2  + (float(r_center[1]) - column)**2) # flat
    dist_to_center_s = np.sqrt((float(s_center[0]) - row)**2  + (float(s_center[1]) - column)**2) # flat

    possible_spots_r = np.where((dist_to_center_r <= r_outer) & (dist_to_center_r >= r_inner))[0] 
    possible_spots_s = np.where((dist_to_center_s <= r_outer) & (dist_to_center_s >= r_inner))[0]


    assert len(possible_spots_r) >= initial_cell_cnt, "initial cell count greater than no. of available gridpoints"
    chosen_spots_r = np.random.choice(possible_spots_r, int(initial_cell_cnt / 2), replace=False) # is always a half circle, so it'll have half the cells to match density
    chosen_spots_s = np.random.choice(possible_spots_s, initial_cell_cnt, replace=False)

    c_flat = c_grid.flatten()
    c_flat[chosen_spots_r] = np.random.randint(1, 11, len(chosen_spots_r))
    c_flat[chosen_spots_s] = np.random.randint(11, 21, len(chosen_spots_s))

    c_grid = c_flat.reshape((dim_y, dim_x))

    return c_grid


""" DIFFUSION FUNCTIONS """

@ray.remote
def remote_solve_diffusion_no_flux_hack_parallel(grid_read_only: npt.NDArray, fin_diff_mat: spmatrix, val0, ddx, xb) -> npt.NDArray:
    return solve_diffusion_no_flux_hack_parallel(grid_read_only, fin_diff_mat, val0, ddx, xb)

def solve_diffusion_no_flux_hack_parallel(grid_read_only: npt.NDArray, fin_diff_mat: spmatrix, val0, ddx, xb) -> npt.NDArray:
    grid = np.ndarray.copy(grid_read_only)
    grid[:,0] = grid[:,1] - (grid[:,1] - val0) / xb * ddx
    grid[:,-1] = grid[:,-2] - (grid[:,-2] - val0) / xb * ddx
    grid[0,:] = grid[1,:] - (grid[1,:] - val0) / xb * ddx
    grid[-1,:] = grid[-2,:] - (grid[-2,:] - val0) / xb * ddx
    
    grid[1:-1,1:-1] = solve_diff_eq1(grid, fin_diff_mat)
    
    return grid

def solve_diff_eq(a_grid: npt.NDArray, n_grid: npt.NDArray, fin_diff_mat: spmatrix) -> tuple[npt.NDArray, npt.NDArray]:
    
    dim_na_r, dim_na_c = a_grid.shape
    n_grid = spsolve(fin_diff_mat, n_grid.flatten())
    a_grid = spsolve(fin_diff_mat, a_grid.flatten())
    
    n_grid = n_grid.reshape((dim_na_r, dim_na_c))
    a_grid = a_grid.reshape((dim_na_r, dim_na_c))
    
    return a_grid[1:-1,1:-1], n_grid[1:-1,1:-1]

def solve_diff_eq1(a_grid: npt.NDArray, fin_diff_mat: spmatrix) -> npt.NDArray:
    
    dim_na_r, dim_na_c = a_grid.shape
    # n_grid = spsolve(fin_diff_mat, n_grid.flatten())
    a_grid = spsolve(fin_diff_mat, a_grid.flatten())
    
    # n_grid = n_grid.reshape((dim_na_r, dim_na_c))
    a_grid = a_grid.reshape((dim_na_r, dim_na_c))
    
    return a_grid

def create_finite_differences_matrix_no_flux(ddim_r: int, ddim_c: int, D: int, dx: int, dt: float, core_cnt: int) -> tuple[spmatrix, npt.NDArray]:
    
    row_splits = np.linspace(1, ddim_r-1 + 2, core_cnt + 1, dtype=int)
    # sub_shape = (ddim_r // core_cnt + 2, ddim_c + 2)
    
    n_row, n_col = ddim_r, ddim_c  # "was +2"
    alpha = D * dt / dx**2
    diagonals = [] # list, since diagonals will be of different lengths
    
    
    
    diagonals.append((1 + 4 * alpha) * np.ones(n_row * n_col)) # site itself
    diagonals.append(- alpha * np.ones(n_row * n_col - 1)) # site immediately to the right
    diagonals.append(- alpha * np.ones(n_row * n_col - 1)) # site immediately to the left
    diagonals.append(- alpha * np.ones(n_row * n_col - n_col)) # site immediately above
    diagonals.append(- alpha * np.ones(n_row * n_col - n_col)) # site immediately below
    
    A = diags(diagonals, [0, 1, -1, n_col, -n_col], format='lil') # lil is cheaper to edit
    
    # for i in range(n_row * n_col):
    #     if (i < n_col) or ( i >= (n_row * n_col - n_col)) or (i % n_col == 0) or ((i-n_col+1) % n_col == 0):
    #         A[i,:] = 0
    #         A[i,i] = 1
    
    # this results in all rows of A summing to 1, thus conserving mass
    for i in range(n_row * n_col):
        if i < n_col: # site is top boundary
            A[i,i] -= alpha
        # if (i % n_col) == 0: # site is left boundary
        #     A[i,i] -= alpha
        if i >= (n_row * n_col - n_col): # site is bottom boundary
            A[i,i] -= alpha
        # if (i-n_col+1) % n_col == 0: # site is right boundary
        #     A[i,i] -= alpha
    A[0,0] -= alpha
    A[-1,-1] -= alpha
    # print(np.sum(A, axis = 0))        

    A = A.tocsc()
    
    return A, row_splits


def solve_diffusion_no_flux(a_grid: npt.NDArray, n_grid: npt.NDArray, fin_diff_mat: spmatrix) -> tuple[npt.NDArray, npt.NDArray]:
    
    # boundary conditions  
    n_grid[:,0] = n_grid[:,1]
    n_grid[:,-1] = n_grid[:,-2]
    n_grid[0,:] = n_grid[1,:]
    n_grid[-1,:] = n_grid[-2,:]

    a_grid[:,0] = a_grid[:,1]
    a_grid[:,-1] = a_grid[:,-2]
    a_grid[0,:] = a_grid[1,:]
    a_grid[-1,:] = a_grid[-2,:]
    
    a_grid[1:-1,1:-1], n_grid[1:-1,1:-1] = solve_diff_eq(a_grid, n_grid, fin_diff_mat)

    return a_grid, n_grid

@ray.remote
def sol_diff_no_flux_remote(grid: npt.NDArray, fin_diff_mat: spmatrix, M):
    return sol_diff_no_flux(grid, fin_diff_mat, M)

def sol_diff_no_flux(grid: npt.NDArray, fin_diff_mat: spmatrix, M) -> tuple[npt.NDArray, npt.NDArray]:
    
    x_dim, y_dim = grid.shape
    
    grid, sol_conv = cg(fin_diff_mat, grid.flatten(), x0=grid.flatten(), M=M)
    assert sol_conv == 0
    grid = grid.reshape((x_dim, y_dim))
    # a_grid, n_grid = solve_diff_eq(a_grid, n_grid, fin_diff_mat)

    return grid

def solve_diffusion_no_flux_hack(a_grid: npt.NDArray, n_grid: npt.NDArray, fin_diff_mat: spmatrix, xb: int, a0: float, n0: float, ddx: int) -> npt.NDArray:
    
    # boundary conditions  
    # n_grid[:,0] = n_grid[:,1]
    # n_grid[:,-1] = n_grid[:,-2]
    # n_grid[0,:] = n_grid[1,:]
    # n_grid[-1,:] = n_grid[-2,:]
    n_grid[:,0] = n_grid[:,1] - (n_grid[:,1] - n0) / xb * ddx
    n_grid[:,-1] = n_grid[:,-2] - (n_grid[:,-2] - n0) / xb * ddx
    n_grid[0,:] = n_grid[1,:] - (n_grid[1,:] - n0) / xb * ddx
    n_grid[-1,:] = n_grid[-2,:] - (n_grid[-2,:] - n0) / xb * ddx

    a_grid[:,0] = a_grid[:,1] - (a_grid[:,1] - a0) / xb * ddx
    a_grid[:,-1] = a_grid[:,-2] - (a_grid[:,-2] - a0) / xb * ddx
    a_grid[0,:] = a_grid[1,:] - (a_grid[1,:] - a0) / xb * ddx
    a_grid[-1,:] = a_grid[-2,:] - (a_grid[-2,:] - a0) / xb * ddx
    
    a_grid[a_grid[:,0] > a0,0] = a0
    a_grid[a_grid[:,-1] > a0,0] = a0
    a_grid[a_grid[0,:] > a0,0] = a0
    a_grid[a_grid[-1,:] > a0,0] = a0
    
    n_grid[n_grid[:,0] > n0,0] = n0
    n_grid[n_grid[:,-1] > n0,0] = n0
    n_grid[n_grid[0,:] > n0,0] = n0
    n_grid[n_grid[-1,:] > n0,0] = n0
    
    
    a_grid[1:-1,1:-1], n_grid[1:-1,1:-1] = solve_diff_eq(a_grid, n_grid, fin_diff_mat)
    
    # worker = 0 # reset for diffusion timestep
    # sub_n = np.array([n_grid[row_splits[i]-1:row_splits[i+1]+1,:] for i in range(core_cnt)])
    # sub_a = np.array([a_grid[row_splits[i]-1:row_splits[i+1]+1,:] for i in range(core_cnt)])

    # for result in pool.starmap(solve_diff_eq, [(sub_a[i], sub_n[i], fin_diff_mat) for i in range(core_cnt)]):
    #     a_grid[row_splits[worker]:row_splits[worker+1],1:-1] = result[0]
    #     n_grid[row_splits[worker]:row_splits[worker+1],1:-1] = result[1]
    #     worker += 1

    return a_grid, n_grid



def solve_diffusion_no_flux1_pure(a_grid: npt.NDArray, fin_diff_mat: spmatrix, ) -> npt.NDArray:
    
    # boundary conditions  
    # a_grid[:,0] = a_grid[:,1]
    # a_grid[:,-1] = a_grid[:,-2]
    # a_grid[0,:] = a_grid[1,:]
    # a_grid[-1,:] = a_grid[-2,:]

    # a_grid[:,0] = a_grid[:,1] + (a_grid[:,1] - a0) / xb * ddx
    # a_grid[:,-1] = a_grid[:,-2] + (a_grid[:,-2] - a0) / xb * ddx
    # a_grid[0,:] = a_grid[1,:] + (a_grid[1,:] - a0) / xb * ddx
    # a_grid[-1,:] = a_grid[-2,:] + (a_grid[-2,:] - a0) / xb * ddx
    
    # a_grid[a_grid[:,0] > a0,0] = a0
    # a_grid[a_grid[:,-1] > a0,0] = a0
    # a_grid[a_grid[0,:] > a0,0] = a0
    # a_grid[a_grid[-1,:] > a0,0] = a0
    
    
    a_grid = solve_diff_eq1(a_grid, fin_diff_mat)
    
    # worker = 0 # reset for diffusion timestep
    # sub_n = np.array([n_grid[row_splits[i]-1:row_splits[i+1]+1,:] for i in range(core_cnt)])
    # sub_a = np.array([a_grid[row_splits[i]-1:row_splits[i+1]+1,:] for i in range(core_cnt)])

    # for result in pool.starmap(solve_diff_eq, [(sub_a[i], sub_n[i], fin_diff_mat) for i in range(core_cnt)]):
    #     a_grid[row_splits[worker]:row_splits[worker+1],1:-1] = result[0]
    #     n_grid[row_splits[worker]:row_splits[worker+1],1:-1] = result[1]
    #     worker += 1

    return a_grid

""" CELL GROWTH """

@nb.njit
def cell_growth_ordered(c_grid_read: npt.NDArray, sub_a_read: npt.NDArray, sub_n_read: npt.NDArray,
                        k_N: int, k_A: int, dt: float, lambda_max_s: float, lambda_max_r: float,
                        gamma: float, mu: float, layer_penalty: float) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, int, npt.NDArray]:
    
    # copies are necessary since the objects passed to starmap are read-only (probably because they are only pointers)
    c_grid_multi = np.copy(c_grid_read)
    sub_a = np.copy(sub_a_read)
    sub_n = np.copy(sub_n_read)
    

    c_grid_dim_r, c_grid_dim_c, c_grid_dim_s = c_grid_multi.shape
    sub_d_dim_r, sub_d_dim_c = sub_a.shape
    
    # figure out which lattice points on the cell grid belongs to which potins on the a/n grids
    mod_r = c_grid_dim_r // sub_d_dim_r
    mod_c = c_grid_dim_c // sub_d_dim_c
    
    division_pos = np.zeros((c_grid_dim_r * c_grid_dim_c, 3), dtype=np.int32) # just pick a dim
    division_cnt = 0
    
    for layer in range(c_grid_dim_s):
        c_grid = c_grid_multi[:,:,layer]
        exp_layer =  np.exp(-layer / layer_penalty)
        # find bounding box for cells
        row_min = 0 
        row_max = 0
        col_min = 0
        col_max = 0

        for row in range(c_grid_dim_r):
            val = np.sum(np.abs(c_grid[row,:]))
            if val > 0:
                row_min = row
                break
        
        for row in range(c_grid_dim_r-1,-1,-1):
            val = np.sum(np.abs(c_grid[row,:]))
            if val > 0:
                row_max = row + 1
                break
        
        for col in range(c_grid_dim_c):
            val = np.sum(np.abs(c_grid[:,col]))
            if val > 0:
                col_min = col
                break
        
        for col in range(c_grid_dim_c-1,-1,-1):
            val = np.sum(np.abs(c_grid[:,col]))
            if val > 0:
                col_max = col + 1
                break
        
        # division_tries = (col_max - col_min) * (row_max - row_min)
        
        
        # division_type = np.zeros(division_tries, dtype=np.int32) # just pick a dim

        for yy_c in range(row_min, row_max): # pick a random site within bounding box
            for xx_c in range(col_min, col_max):
            # yy_c = np.random.randint(row_min, row_max)
            # xx_c = np.random.randint(col_min, col_max)
            
            # print('here')
            
                yy_d = yy_c // mod_r
                xx_d = xx_c // mod_c
                
                if (c_grid[yy_c,xx_c] - 1) // 10 == 0: # cell is resistant
                    
                    sub_a[yy_d,xx_d] -= sub_a[yy_d,xx_d] * gamma * exp_layer * dt # remove antibiotic
                    
                    if lambda_max_r * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * dt > np.random.random():

                        if c_grid[yy_c,xx_c] == 10: # divide if ready
                            division_pos[division_cnt,:] = [yy_c,xx_c,layer]
                            # division_type[division_cnt] = c_grid[yy_c,xx_c]
                            division_cnt += 1
                            c_grid[yy_c,xx_c] = 1
                        else: # else progress towards growth
                            c_grid[yy_c,xx_c] += 1

                        # consume nutrients
                        sub_n[yy_d,xx_d] -= mu * dt # constant, so it doesn't need _read
                        if sub_n[yy_d,xx_d] < 0:
                            sub_n[yy_d,xx_d] = 0
                    
                        
                elif (c_grid[yy_c,xx_c] - 1) // 10 == 1: # cell is non-resistant
                    # if lambda_max_s * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * (k_A + exp_layer * sub_a_read[yy_d,xx_d])**-1 * dt > np.random.random():
                    if lambda_max_s * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * (1 + (exp_layer * sub_a_read[yy_d,xx_d] / k_A)**-3)**-1 * dt > np.random.random():
                    

                        if c_grid[yy_c,xx_c] == 20: # ready to divide
                            division_pos[division_cnt,:] = [yy_c,xx_c, layer]
                            # division_type[division_cnt] = c_grid[yy_c,xx_c]
                            division_cnt += 1
                            c_grid[yy_c,xx_c] = 11
                        else: # progress towards growth
                            c_grid[yy_c,xx_c] += 1
                        
                        # consume nutrients
                        sub_n[yy_d,xx_d] -= mu * dt
                        if sub_n[yy_d,xx_d] < 0:
                            sub_n[yy_d,xx_d] = 0
                            
        c_grid_multi[:,:,layer] = c_grid


    # remove antibiotic
    # for xx in range(col_min, col_max): # in c_grid coordinates
    #     for yy in range(row_min, row_max):
    #         if (c_grid[yy,xx] -1) // 10 == 0:
    #             yy_d = yy // mod_r
    #             xx_d = xx // mod_c
    #             sub_a[yy_d,xx_d] -= sub_a[yy_d,xx_d] * gamma * dt
            # if a_grid[yy+1,xx+1] < 0:
            #     a_grid[yy+1,xx+1] = 0
    
    return c_grid_multi, sub_a, sub_n, division_cnt, division_pos[:division_cnt], # division_type[:division_cnt]

@nb.njit
def cell_growth_ordered_bbox(c_grid_read: npt.NDArray, sub_a_read: npt.NDArray, sub_n_read: npt.NDArray,
                        k_N: int, k_A: float, dt: float, lambda_max_s: float, lambda_max_r: float,
                        gamma: float, mu: float, layer_penalty: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, int, npt.NDArray]:
    
    
    
    # copies are necessary since the objects passed to starmap are read-only (probably because they are only pointers)
    c_grid_multi = np.copy(c_grid_read)
    sub_a = np.copy(sub_a_read)
    sub_n = np.copy(sub_n_read)
    

    c_grid_dim_r, c_grid_dim_c, c_grid_dim_s = c_grid_multi.shape
    sub_d_dim_r, sub_d_dim_c = sub_a.shape
    
    # figure out which lattice points on the cell grid belongs to which potins on the a/n grids
    mod_r = c_grid_dim_r // sub_d_dim_r
    mod_c = c_grid_dim_c // sub_d_dim_c
    
    division_pos = np.zeros((c_grid_dim_r * c_grid_dim_c, 3), dtype=np.int32) # just pick a dim
    division_cnt = 0
    
    for layer in range(c_grid_dim_s):
        c_grid = c_grid_multi[:,:,layer]
        exp_layer =  np.exp(-layer / layer_penalty)
        # find bounding box for cells
        
        
        # division_tries = (col_max - col_min) * (row_max - row_min)
        
        
        # division_type = np.zeros(division_tries, dtype=np.int32) # just pick a dim

        for yy_c in range(c_grid_dim_r): # pick a random site within bounding box
            for xx_c in range(c_grid_dim_c):
            # yy_c = np.random.randint(row_min, row_max)
            # xx_c = np.random.randint(col_min, col_max)
            
            # print('here')
            
                yy_d = yy_c // mod_r
                xx_d = xx_c // mod_c
                
                if (c_grid[yy_c,xx_c] - 1) // 10 == 0: # cell is resistant
                    
                    sub_a[yy_d,xx_d] -= sub_a[yy_d,xx_d] * gamma * exp_layer * dt # remove antibiotic
                    
                    if lambda_max_r * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * dt > np.random.random():

                        if c_grid[yy_c,xx_c] == 10: # divide if ready
                            division_pos[division_cnt,:] = [yy_c,xx_c,layer]
                            # division_type[division_cnt] = c_grid[yy_c,xx_c]
                            division_cnt += 1
                            c_grid[yy_c,xx_c] = 1
                        else: # else progress towards growth
                            c_grid[yy_c,xx_c] += 1

                        # consume nutrients
                        sub_n[yy_d,xx_d] -= mu * dt # constant, so it doesn't need _read
                        if sub_n[yy_d,xx_d] < 0:
                            sub_n[yy_d,xx_d] = 0
                    
                        
                elif (c_grid[yy_c,xx_c] - 1) // 10 == 1: # cell is non-resistant
                    if lambda_max_s * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * (1 + (exp_layer * sub_a_read[yy_d,xx_d] / k_A))**-1 * dt > np.random.random():
                    # if lambda_max_s * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * np.exp(-a_scale * sub_a_read[yy_d, xx_d] * exp_layer) * dt > np.random.random():
                    

                        if c_grid[yy_c,xx_c] == 20: # ready to divide
                            division_pos[division_cnt,:] = [yy_c, xx_c, layer]
                            # division_type[division_cnt] = c_grid[yy_c,xx_c]
                            division_cnt += 1
                            c_grid[yy_c,xx_c] = 11
                        else: # progress towards growth
                            c_grid[yy_c,xx_c] += 1
                        
                        # consume nutrients
                        sub_n[yy_d,xx_d] -= mu * dt
                        if sub_n[yy_d,xx_d] < 0:
                            sub_n[yy_d,xx_d] = 0
                            
        c_grid_multi[:,:,layer] = c_grid

    return c_grid_multi, sub_a, sub_n, division_cnt, division_pos[:division_cnt], # division_type[:division_cnt]

@nb.njit
def cell_growth_ordered_bbox_no_coord_return(c_grid_read: npt.NDArray, sub_a_read: npt.NDArray, sub_n_read: npt.NDArray,
                        k_N: int, k_A: float, dt: float, lambda_max_s: float, lambda_max_r: float,
                        gamma: float, mu: float, layer_penalty: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    
    
    
    # copies are necessary since the objects passed to starmap are read-only (probably because they are only pointers)
    c_grid_multi = np.copy(c_grid_read)
    sub_a = np.copy(sub_a_read)
    sub_n = np.copy(sub_n_read)
    

    c_grid_dim_r, c_grid_dim_c, c_grid_dim_s = c_grid_multi.shape
    sub_d_dim_r, sub_d_dim_c = sub_a.shape
    
    
    if sub_d_dim_r == 0 or sub_d_dim_c == 0:
        raise ValueError("sub_a_read or sub_n_read has a zero dimension, causing division by zero.")
    
    # figure out which lattice points on the cell grid belongs to which potins on the a/n grids
    mod_r = c_grid_dim_r // sub_d_dim_r
    mod_c = c_grid_dim_c // sub_d_dim_c
    
    # division_pos = np.zeros((c_grid_dim_r * c_grid_dim_c, 3), dtype=np.int32) # just pick a dim
    # division_cnt = 0
    
    for layer in range(c_grid_dim_s):
        c_grid = c_grid_multi[:,:,layer]
        exp_layer =  np.exp(-layer / layer_penalty)
        # find bounding box for cells
        
        
        # division_tries = (col_max - col_min) * (row_max - row_min)
        
        # division_type = np.zeros(division_tries, dtype=np.int32) # just pick a dim

        for yy_c in range(c_grid_dim_r): # pick a random site within bounding box
            for xx_c in range(c_grid_dim_c):
            # yy_c = np.random.randint(row_min, row_max)
            # xx_c = np.random.randint(col_min, col_max)
            
            # print('here')
            
                yy_d = yy_c // mod_r
                xx_d = xx_c // mod_c
                
                if (c_grid[yy_c,xx_c] - 1) // 10 == 0: # cell is resistant
                    
                    sub_a[yy_d,xx_d] -= sub_a_read[yy_d,xx_d] * gamma * exp_layer * dt # remove antibiotic
                    
                    if lambda_max_r * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * dt > np.random.random():

                        if c_grid[yy_c,xx_c] == 10: # divide if ready
                            # division_pos[division_cnt,:] = [yy_c,xx_c,layer]
                            # division_type[division_cnt] = c_grid[yy_c,xx_c]
                            # division_cnt += 1
                            c_grid[yy_c,xx_c] = -1 # mark cell as ready to divide
                        else: # else progress towards growth
                            c_grid[yy_c,xx_c] += 1

                        # consume nutrients
                        sub_n[yy_d,xx_d] -= mu * dt # constant, so it doesn't need _read
                        if sub_n[yy_d,xx_d] < 0:
                            sub_n[yy_d,xx_d] = 0
                    
                        
                elif (c_grid[yy_c,xx_c] - 1) // 10 == 1: # cell is non-resistant
                    if lambda_max_s * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * (1 + (exp_layer * sub_a_read[yy_d,xx_d] / k_A))**-1 * dt > np.random.random():
                    # if lambda_max_s * exp_layer * sub_n_read[yy_d,xx_d] / (k_N + exp_layer * sub_n_read[yy_d,xx_d]) * np.exp(-a_scale * sub_a_read[yy_d, xx_d] * exp_layer) * dt > np.random.random():
                    

                        if c_grid[yy_c,xx_c] == 20: # ready to divide
                            # division_pos[division_cnt,:] = [yy_c, xx_c, layer]
                            # division_type[division_cnt] = c_grid[yy_c,xx_c]
                            # division_cnt += 1
                            c_grid[yy_c,xx_c] = -11 # mark as ready for growth
                        else: # progress towards growth
                            c_grid[yy_c,xx_c] += 1
                        
                        # consume nutrients
                        sub_n[yy_d,xx_d] -= mu * dt
                        if sub_n[yy_d,xx_d] < 0:
                            sub_n[yy_d,xx_d] = 0
                            
        c_grid_multi[:,:,layer] = c_grid

    return c_grid_multi, sub_a, sub_n, # division_cnt, division_pos[:division_cnt], # division_type[:division_cnt]



@nb.njit
def find_cells_FNES_and_divide_parallel_multilayer(small_grid_read, rp: int, div_mask_read):
    div_y, div_x, div_z = np.where(small_grid_read[rp:-rp, rp:-rp,:] < 0)
    small_grid_height = small_grid_read.shape[2]
    r_cnt = 0
    s_cnt = 0
    small_grid = np.copy(small_grid_read)
    div_mask = np.copy(div_mask_read)
    
    if len(div_y) != 0:
        # search layer by layer
        for div_yy, div_xx, div_zz in zip(div_y, div_x, div_z): # these are relative to the sub_grid
        # for cell in range(len(div_y)):
            div_yy += rp
            div_xx += rp
            small_grid[div_yy, div_xx, div_zz] *= -1
            if small_grid[div_yy, div_xx, div_zz] == 1:
                r_cnt += 1
            else: 
                s_cnt += 1

            # div_yy = local_division_list[0,cell]
            # div_xx = local_division_list[1,cell]
            # print(div_yy, div_xx)
            for layer in range(div_mask[div_yy, div_xx], small_grid_height):
                # print(layer)
                local_mask = small_grid[div_yy-rp : div_yy + rp+1, div_xx - rp : div_xx + rp + 1, layer].astype(np.bool_) # mask for current layer
                # print(local_mask.shape)
                y_free, x_free = np.where(local_mask == 0) # pos of free sites relative to mask
                if len(y_free) == 0:
                    continue
                rand_order = np.random.permutation(len(y_free)) # random sequence of idx to shuffle free sites, so the minimum that's found isn't direction biased
                y_free = y_free[rand_order]
                x_free = x_free[rand_order]
                distance = (y_free - rp)**2 + (x_free - rp)**2 # maybe add z dim?
                local_mask[:,:] = 0
                
                closest_idx = np.argmin(distance)
                if distance[closest_idx] > rp**2:
                    continue
                # closest_dist = rp + 1 # initial distance excludes free sites outside rp
                # closest_dist_idx = -1
                # # iterate to find NES. 
                # for idx, dist in enumerate(distance):
                #     if dist < closest_dist:
                #         closest_dist = dist
                #         closest_dist_idx = idx
                #     if dist == 1:
                #         break
                
                # if closest_dist_idx == -1:
                #     continue
                
                else: # a site is availiable
                    
                    final_pos = [div_yy - rp + y_free[closest_idx], div_xx - rp + x_free[closest_idx]]
                    current_pos = np.array([div_yy, div_xx])
                    
                    
                    # small_grid = lm.push_cells(current_pos, final_pos, small_grid)
                    # print(f'free site found at {final_pos[0] + growth_box_size, final_pos[1] + growth_box_size}')
                    
                    # print(small_grid[current_pos])
                    # #pushes on end layer
                    horizontal = final_pos[1] - current_pos[1]
                    abs_horizontal = np.abs(horizontal)
                    vertical = final_pos[0] - current_pos[0]
                    abs_vertical = np.abs(vertical)
                    
                    push_array = np.zeros(abs_horizontal + abs_vertical, dtype=np.int64) # stores order of pushing. 1 is vertical, 0 is horizontal 
                    horizontal_mod = 0
                    if horizontal < 0:
                        horizontal_mod = int(1) # needs to lower index
                    elif horizontal > 0:
                        horizontal_mod = int(-1)
                    else: 
                        horizontal_mod = int(0) # straight up/down
                    
                    vertical_mod = 0
                    if vertical < 0:
                        vertical_mod = int(1) # needs to lower index
                    elif vertical > 0:
                        vertical_mod = int(-1)
                    else: 
                        vertical_mod = int(0) # straight up/down
                        
                    # print(horizontal_mod, vertical_mod) # right
                    loop_val = abs_horizontal + abs_vertical #
                    for i in range(loop_val):
                        p_vertical = abs_vertical / (abs_horizontal + abs_vertical) # order of pushing horizontal/vertical ends up being bin. dist.
                        # print(p_vertical) # correct
                        if p_vertical > np.random.random():
                            push_array[i] = 1
                            abs_vertical -= 1
                        else:
                            abs_horizontal -= 1
                            

                    push_array = np.flip(push_array)
                    # move_to = final_pos
                    # final_pos = np.asarray(final_pos, dtype=np.int64)
                    for val in push_array:
                        if val:
                            small_grid[final_pos[0], final_pos[1], layer] = small_grid[final_pos[0] + vertical_mod, final_pos[1], layer]
                            # print('moved ' + str([move_to[0] + vertical_mod + growth_box_size, move_to[1] + growth_box_size]) + 'to ' + str([move_to[0] + growth_box_size, move_to[1] + growth_box_size]))
                            final_pos = [final_pos[0] + vertical_mod, final_pos[1]]
                            
                        else:
                            small_grid[final_pos[0], final_pos[1], layer] = small_grid[final_pos[0], final_pos[1] + horizontal_mod, layer]
                            # print('moved ' + str([move_to[0] + growth_box_size, move_to[1] + horizontal_mod + growth_box_size]) + 'to ' + str([move_to[0] + growth_box_size, move_to[1] + growth_box_size]))
                            final_pos = [final_pos[0], final_pos[1] + horizontal_mod]
                    
                    # push upwards if NES was located on a different layer
                    if layer != div_zz:
                        div_mask[div_yy, div_xx] = layer # modify division mask
                        
                        # print(f'cell pushed to layer {layer}')
                        for i in range(layer, div_zz, -1):
                                small_grid[div_yy, div_xx, i] = small_grid[div_yy, div_xx, i-1] # push upwards
                    break
    
    small_grid = np.abs(small_grid)
    return small_grid, div_mask, r_cnt, s_cnt
                     

@ray.remote
def remote_do_everything(div_grid, div_mask_small, used_dsplits_row, used_dsplits_col, Gbox_Dboxes_a, Gbox_Dboxes_n,
                         dt, pushing_radius, k_A, k_N, lambda_max_r, lambda_max_s, vertical_loss):
    return do_everything(div_grid, div_mask_small, used_dsplits_row, used_dsplits_col, Gbox_Dboxes_a, Gbox_Dboxes_n,
                            dt, pushing_radius, k_A, k_N, lambda_max_r, lambda_max_s, vertical_loss)

# @ray.remote
@nb.njit
def do_everything(small_grid_read, div_mask_read, Dbox_coord_for_Gbox_row, Dbox_coord_for_Gbox_col, GDbox_choice_A, GDbox_choice_N, dt, rp, k_A, k_N, lambda_max_r, lambda_max_s, vertical_loss):
    
    sen_growth_at_cutoff = lambda_max_s / (1 + (4/k_A)**3)
    
    res_at_Dbox = np.zeros(GDbox_choice_A.shape, dtype=np.int32)
    growth_at_Dbox = np.zeros(GDbox_choice_A.shape, dtype=np.int32)
    missed_cell = False 

    # stores coordinates for dividing cells found during grid search 
    div_y = np.zeros(int((small_grid_read.shape[0] - rp) * (small_grid_read.shape[1] - rp)), dtype=np.int16)
    div_x = np.zeros(int((small_grid_read.shape[0] - rp) * (small_grid_read.shape[1] - rp)), dtype=np.int16)
    div_z = np.zeros(int((small_grid_read.shape[0] - rp) * (small_grid_read.shape[1] - rp)), dtype=np.int16)
    cells_to_divide = 0
    
    # copy for editing and returning
    small_grid = np.copy(small_grid_read)
    div_mask = np.copy(div_mask_read)
    
    sg_r, sg_c, sg_s = small_grid.shape
    
    # divide all cells. row, col (slice) corresponds to Gbox, not small_grid, therefore the range
    for row in range(rp, sg_r-rp):
        for col in range(rp, sg_c-rp):
            for sli in range(0, sg_s): # iterate over all layers/slices
                
                cell_found = False
                
                if (small_grid[row, col, sli] - 1) // 10 == 0: # cell is resistant
                    layer_penalty = np.exp(- sli / vertical_loss)
                    
                    # find correct N and A grid
                    found_row = False
                    for idx, val in enumerate(Dbox_coord_for_Gbox_row):
                        if row - rp < val:
                            cell_row_index_in_dbox = idx
                            found_row = True
                            break
                    if not found_row:
                        found_row = True
                        cell_row_index_in_dbox = len(Dbox_coord_for_Gbox_row)
                    
                    found_col = False
                    for idx, val in enumerate(Dbox_coord_for_Gbox_col):
                        if col - rp < val:
                            cell_col_index_in_dbox = idx
                            found_col = True
                            break           
                    if not found_col:
                        found_col = True
                        cell_col_index_in_dbox = len(Dbox_coord_for_Gbox_col)         

                    n_val = GDbox_choice_N[cell_row_index_in_dbox, cell_col_index_in_dbox]
                    res_at_Dbox[cell_row_index_in_dbox, cell_col_index_in_dbox] += layer_penalty # note that a res cell is at this site
                    
                    if lambda_max_r * n_val * layer_penalty / (k_N + n_val * layer_penalty) * dt > np.random.random(): # cell grows
                        growth_at_Dbox[cell_row_index_in_dbox, cell_col_index_in_dbox] += 1  # mark to remove nutrients later
                        cell_found = True
                        if small_grid[row, col, sli] < 10: # growth
                            small_grid[row, col, sli] += 1
                        else: # divide
                            small_grid[row, col, sli] = -10 # mark for division
                            
                if (small_grid[row, col, sli] - 1) // 10 == 1: # cell is sen
                    layer_penalty = np.exp(- sli / vertical_loss)
                    # find correct N and A grid
                    found_row = False
                    for idx, val in enumerate(Dbox_coord_for_Gbox_row):
                        if row - rp < val:
                            cell_row_index_in_dbox = idx
                            found_row = True
                            break
                    if not found_row:
                        found_row = True
                        cell_row_index_in_dbox = len(Dbox_coord_for_Gbox_row)
                    
                    found_col = False
                    for idx, val in enumerate(Dbox_coord_for_Gbox_col):
                        if col - rp < val:
                            cell_col_index_in_dbox = idx
                            found_col = True
                            break           
                    if not found_col:
                        found_col=True
                        cell_col_index_in_dbox = len(Dbox_coord_for_Gbox_col)  
                               
                    a_val = GDbox_choice_A[cell_row_index_in_dbox, cell_col_index_in_dbox]
                    n_val = GDbox_choice_N[cell_row_index_in_dbox, cell_col_index_in_dbox]
                    
                    # employ new condition with interpolation between a=4 and a=5. Note that this is hardcoded
                    grow_cell = False
                    if a_val < 4:
                        if lambda_max_s / (1 + (a_val/k_A)**3) * dt > np.random.random(): 
                            grow_cell = True
                    elif a_val >= 4 :
                        if sen_growth_at_cutoff * (5-a_val) * dt > np.random.random():
                            grow_cell = True
                            
                    if grow_cell:
                        growth_at_Dbox[cell_row_index_in_dbox, cell_col_index_in_dbox] += 1  # mark to remove nutrients later
                        cell_found = True
                        if small_grid[row, col, sli] < 20: # growth
                            small_grid[row, col, sli] += 1
                        else: # divide
                            small_grid[row, col, sli] = -20 # mark for division
                
                if cell_found:
                    div_y[cells_to_divide] = row
                    div_x[cells_to_divide] = col
                    div_z[cells_to_divide] = sli
                    cells_to_divide += 1
                    
                     
    # div_y, div_x, div_z = np.where(small_grid[rp:-rp, rp:-rp,:] < 0) # these are off by rp in both directions, but is fixed later
    
    # shuffle the order that cells are divided 
    rand_coord_order = np.random.permutation(cells_to_divide)
    div_y = div_y[rand_coord_order]
    div_x = div_x[rand_coord_order]
    div_z = div_z[rand_coord_order]
    # print(cells_to_divide)
    small_grid_height = small_grid_read.shape[2]
    r_cnt = 0
    s_cnt = 0
    # div_mask = np.copy(div_mask_read)
    
    if len(div_y) != 0:
        # search layer by layer
        for div_yy, div_xx, div_zz in zip(div_y, div_x, div_z): # these are relative to the Gbox    
            # div_yy += rp
            # div_xx += rp
            if small_grid[div_yy, div_xx, div_zz] > 0: # the cell previously found has been pushed, so the new cell pushed to this loc shouldn't divide
                missed_cell = True
                continue
            
            small_grid[div_yy, div_xx, div_zz] *= -1 # flip sign
            small_grid[div_yy, div_xx, div_zz] -= 9 # and reset growth steps
            


            for layer in range(div_mask[div_yy, div_xx], small_grid_height):
                
                local_mask = small_grid[div_yy-rp : div_yy + rp+1, div_xx - rp : div_xx + rp + 1, layer].astype(np.bool_) # mask for current layer
                y_free, x_free = np.where(local_mask == 0) # pos of free sites relative to mask
                
                if len(y_free) == 0: # no free sites at all, move on to next layer
                    div_mask[div_yy, div_xx] += 1
                    continue

                rand_order = np.random.permutation(len(y_free)) # random sequence of idx to shuffle free sites, so the minimum that's found isn't direction biased
                y_free = y_free[rand_order]
                x_free = x_free[rand_order]
                distance = (y_free - rp)**2 + (x_free - rp)**2 # maybe add z dim?
                local_mask[:,:] = 0 # TODO: might not be needed? Don't remember why I added it?
                
                closest_idx = np.argmin(distance) # find idx of (random) closest site - this is probably(?) the fastest way to do it
                
                if distance[closest_idx] > rp**2: # move on to next layer if it's too far away (mask is square, so there's sites in corners that are too far away by design)
                    div_mask[div_yy, div_xx] = layer + 1
                    
                    continue
 
                else: # a site is availiable and pushing commences
                    
                    # add to cell cnt
                    if small_grid[div_yy, div_xx, div_zz] == 1:
                        r_cnt += 1
                    else: 
                        s_cnt += 1
                    
                    final_pos = [div_yy - rp + y_free[closest_idx], div_xx - rp + x_free[closest_idx]]
                    current_pos = np.array([div_yy, div_xx])
                    
                    horizontal = final_pos[1] - current_pos[1]
                    abs_horizontal = np.abs(horizontal)
                    vertical = final_pos[0] - current_pos[0]
                    abs_vertical = np.abs(vertical)
                    
                    push_array = np.zeros(abs_horizontal + abs_vertical, dtype=np.int64) # stores order of pushing. 1 is vertical, 0 is horizontal 
                    horizontal_mod = 0
                    if horizontal < 0:
                        horizontal_mod = int(1) # needs to lower index
                    elif horizontal > 0:
                        horizontal_mod = int(-1)
                    else: 
                        horizontal_mod = int(0) # straight up/down
                    
                    vertical_mod = 0
                    if vertical < 0:
                        vertical_mod = int(1) # needs to lower index
                    elif vertical > 0:
                        vertical_mod = int(-1)
                    else: 
                        vertical_mod = int(0) # straight up/down
                        
                    loop_val = abs_horizontal + abs_vertical #
                    for i in range(loop_val):
                        p_vertical = abs_vertical / (abs_horizontal + abs_vertical) # order of pushing horizontal/vertical ends up being bin. dist.
                        # print(p_vertical) # correct
                        if p_vertical > np.random.random():
                            push_array[i] = 1
                            abs_vertical -= 1
                        else:
                            abs_horizontal -= 1
                            
                    push_array = np.flip(push_array)
                    
                    for val in push_array:
                        if val:
                            small_grid[final_pos[0], final_pos[1], layer] = small_grid[final_pos[0] + vertical_mod, final_pos[1], layer]
                            # print('moved ' + str([move_to[0] + vertical_mod + growth_box_size, move_to[1] + growth_box_size]) + 'to ' + str([move_to[0] + growth_box_size, move_to[1] + growth_box_size]))
                            final_pos = [final_pos[0] + vertical_mod, final_pos[1]]
                            
                        else:
                            small_grid[final_pos[0], final_pos[1], layer] = small_grid[final_pos[0], final_pos[1] + horizontal_mod, layer]
                            # print('moved ' + str([move_to[0] + growth_box_size, move_to[1] + horizontal_mod + growth_box_size]) + 'to ' + str([move_to[0] + growth_box_size, move_to[1] + growth_box_size]))
                            final_pos = [final_pos[0], final_pos[1] + horizontal_mod]
                    
                    # push upwards if NES was located on a different layer
                    if layer != div_zz:
                        div_mask[div_yy, div_xx] = layer # modify division mask
                        
                        # print(f'cell pushed to layer {layer}')
                        for i in range(layer, div_zz, -1):
                                small_grid[div_yy, div_xx, i] = small_grid[div_yy, div_xx, i-1] # push upwards
                                
                    break # out of layer loop, since cell has divided
    
    if missed_cell:
        small_grid = np.abs(small_grid) # catch any < 0 cells that might have moved. They will get the chance to divide next timestep
        
    return small_grid, div_mask, r_cnt, s_cnt, res_at_Dbox, growth_at_Dbox
    
    # return small_grid_read, div_mask, r_cnt, s_cnt, res_at_Dbox, growth_at_Dbox
    # return small_grid, div_mask, r_cnt, s_cnt, GDbox_choice_A, growth_at_Dbox
                     
   


# @nb.njit
# def push_cells_parallel(division_list: npt.NDArray, end_pos_list, sub_grid: npt.ArrayLike) -> npt.NDArray: 
#     """
#     pushes a line of cells from the dividing cell to the end position found by find_nearest_empty_site

#     Args:
#         current_pos (_type_): indexes of dividing cell
#         final_pos (_type_): indexes of pushing endpoint
#         cell_grid (_type_): cell grid

#     Returns:
#         _type_: cell grid after pushing and division
#     """
    
#     # final_pos = np.array([final_pos[0,0], final_pos[0,1]], dtype=np.int64)

#     horizontal = final_pos[1] - current_pos[1]
#     abs_horizontal = np.abs(horizontal)
#     vertical = final_pos[0] - current_pos[0]
#     abs_vertical = np.abs(vertical)
    
#     push_array = np.zeros(abs_horizontal + abs_vertical, dtype=np.int64) # stores order of pushing. 1 is vertical, 0 is horizontal 
#     horizontal_mod = 0
#     if horizontal < 0:
#         horizontal_mod = int(1)# needs to lower index
#     elif horizontal > 0:
#         horizontal_mod = int(-1)
#     else: 
#         horizontal_mod = int(0)# straight up/down
    
#     vertical_mod = 0
#     if vertical < 0:
#         vertical_mod = int(1) # needs to lower index
#     elif vertical > 0:
#         vertical_mod = int(-1)
#     else: 
#         vertical_mod = int(0) # straight up/down
        
#     # print(horizontal_mod, vertical_mod) # right
#     loop_val = abs_horizontal + abs_vertical #
#     for i in range(loop_val):
#         p_vertical = abs_vertical / (abs_horizontal + abs_vertical) # order of pushing horizontal/vertical ends up being bin. dist.
#         # print(p_vertical) # correct
#         if p_vertical > np.random.random():
#             push_array[i] = 1
#             abs_vertical -= 1
#         else:
#             abs_horizontal -= 1
            

#     push_array = np.flip(push_array)
#     # move_to = final_pos
#     # final_pos = np.asarray(final_pos, dtype=np.int64)
#     for val in push_array:
#         if val:
#             cell_grid[final_pos[0], final_pos[1]] = cell_grid[final_pos[0] + vertical_mod, final_pos[1]]
#             # print('moved ' + str([move_to[0] + vertical_mod, move_to[1]]) + 'to ' + str(move_to))
#             final_pos = [final_pos[0] + vertical_mod, final_pos[1]]
            
#         else:
#             cell_grid[final_pos[0], final_pos[1]] = cell_grid[final_pos[0], final_pos[1] + horizontal_mod]
#             # print('moved ' + str([move_to[0], move_to[1] + horizontal_mod]) + 'to ' + str(move_to))
#             final_pos = [final_pos[0], final_pos[1] + horizontal_mod]
            
#     # cell_grid[final_pos[0],final_pos[1]] -= 9
#     cell_grid[current_pos[0], current_pos[1]] = cell_grid[final_pos[0],final_pos[1]] # puts new cell in old cells place - much easier
    
#     # print('put new cell at ' + str(move_to))
#     # flip push array and work backwards to modify without having placeholders! DONE
    
#     return cell_grid



@nb.njit
def push_cells(current_pos: npt.ArrayLike, final_pos, cell_grid: npt.ArrayLike) -> npt.NDArray: 
    """
    pushes a line of cells from the dividing cell to the end position found by find_nearest_empty_site

    Args:
        current_pos (_type_): indexes of dividing cell
        final_pos (_type_): indexes of pushing endpoint
        cell_grid (_type_): cell grid

    Returns:
        _type_: cell grid after pushing and division
    """
    
    # final_pos = np.array([final_pos[0,0], final_pos[0,1]], dtype=np.int64)

    horizontal = final_pos[1] - current_pos[1]
    abs_horizontal = np.abs(horizontal)
    vertical = final_pos[0] - current_pos[0]
    abs_vertical = np.abs(vertical)
    
    push_array = np.zeros(abs_horizontal + abs_vertical, dtype=np.int64) # stores order of pushing. 1 is vertical, 0 is horizontal 
    horizontal_mod = 0
    if horizontal < 0:
        horizontal_mod = int(1)# needs to lower index
    elif horizontal > 0:
        horizontal_mod = int(-1)
    else: 
        horizontal_mod = int(0)# straight up/down
    
    vertical_mod = 0
    if vertical < 0:
        vertical_mod = int(1) # needs to lower index
    elif vertical > 0:
        vertical_mod = int(-1)
    else: 
        vertical_mod = int(0) # straight up/down
        
    # print(horizontal_mod, vertical_mod) # right
    loop_val = abs_horizontal + abs_vertical #
    for i in range(loop_val):
        p_vertical = abs_vertical / (abs_horizontal + abs_vertical) # order of pushing horizontal/vertical ends up being bin. dist.
        # print(p_vertical) # correct
        if p_vertical > np.random.random():
            push_array[i] = 1
            abs_vertical -= 1
        else:
            abs_horizontal -= 1
            

    push_array = np.flip(push_array)
    # move_to = final_pos
    # final_pos = np.asarray(final_pos, dtype=np.int64)
    for val in push_array:
        if val:
            cell_grid[final_pos[0], final_pos[1]] = cell_grid[final_pos[0] + vertical_mod, final_pos[1]]
            # print('moved ' + str([move_to[0] + vertical_mod, move_to[1]]) + 'to ' + str(move_to))
            final_pos = [final_pos[0] + vertical_mod, final_pos[1]]
            
        else:
            cell_grid[final_pos[0], final_pos[1]] = cell_grid[final_pos[0], final_pos[1] + horizontal_mod]
            # print('moved ' + str([move_to[0], move_to[1] + horizontal_mod]) + 'to ' + str(move_to))
            final_pos = [final_pos[0], final_pos[1] + horizontal_mod]
            
    # cell_grid[final_pos[0],final_pos[1]] -= 9
    cell_grid[current_pos[0], current_pos[1]] = cell_grid[final_pos[0],final_pos[1]] # puts new cell in old cells place - much easier
    
    # print('put new cell at ' + str(move_to))
    # flip push array and work backwards to modify without having placeholders! DONE
    
    return cell_grid


""" BILLION DIFFERENT FIND NEAREST EMPTY SITE FUNCTIONS """

@nb.njit
def FNES_no_roll_optimized(coord_list: npt.NDArray, c_multi: npt.NDArray, rp: int, n_neigh: int, start_layer: npt.NDArray) -> npt.NDArray: # no rolling
    
    neigh_list = np.zeros((coord_list.shape[0], int(3*n_neigh)), dtype=np.int64) - 1
    # s = int(2 * rp + 1)
    c_y, c_x, c_z = c_multi.shape
    rp_squared = rp**2
    for idx, (yy, xx, zz) in enumerate(coord_list): # loops over all cells that should divide
        
        found_neigh = 0

        for layer in range(start_layer[yy,xx], c_z): # loops over cells layer and layers above

            c_grid = np.copy(c_multi[:,:,layer])
        
            p_dim = int(2*rp + 1) # pushing dim is square
            
            local_mask = np.zeros((p_dim, p_dim), dtype=nb.bool_)
            # local_pos = np.array([r, r])
            if xx - rp < 0: # close to lh edge
                missing_cols_l = rp - xx
                missing_cols_r = 0
                local_mask[:,:missing_cols_l] = 1
                # local_mask[:,missing_cols_l:] = c_grid[yy-rp:yy+rp+1, 0:xx+rp+1]
                
            elif xx + rp >= c_x: # close to rh edge
                missing_cols_r = (xx + rp +1) % c_x
                missing_cols_l = 0
                # local_mask[:,:-missing_cols_r] = c_grid[yy-rp:yy+rp+1, xx-rp:]
                local_mask[:,-missing_cols_r:] = 1
            
            if yy - rp < 0: # close to top
                missing_rows_t = rp - yy
                missing_rows_b = 0
                local_mask[:missing_rows_t, :] = 1
            
            elif yy + rp >= c_y: # close to bot
                missing_rows_b = (yy + rp + 1) % c_y
                missing_rows_t = 0
                local_mask[-missing_rows_b:, :] = 1
            
            # print(missing_rows_t, missing_rows_b)
            # print(missing_cols_l, missing_cols_r)
                
            local_mask[missing_rows_t:local_mask.shape[0]-missing_rows_b, missing_cols_l:local_mask.shape[1]-missing_cols_r] = np.copy(c_grid[yy-rp + missing_rows_t : yy+rp+1 - missing_rows_b,
                                                                                                              xx-rp + missing_cols_l : xx+rp+1 - missing_cols_r]).astype(nb.bool_)
            # print('test')
            # else: # general, when not close to edges
            #     local_mask = np.copy(c_grid[yy - rp: yy + rp + 1, xx - rp: xx + rp + 1]).astype(nb.bool_) # copies a square centered at dividing cell
            #     missing_cols_r = 0
            #     missing_cols_l = 0
            # get local coordinates where mask is 0
            y_loc, x_loc = np.where(local_mask == 0) # can be optimized based on whether missing_cols_l/r are zero

            if len(y_loc) == 0:
                # print('no free sites at layer')
                continue
            
            # if len(y_loc) == 0: # no free sites - this is wrong, as there's more layers to be searched
            #     found_neigh = n_neigh
            #     neigh_list[idx,:] = -1
            #     break
            
            # cal distance to all free sites on layer
            dist = (y_loc - rp)**2 + (x_loc - rp)**2  # there's probably some optimized function for this - look for it when you have the time
            # dist = np.linalg.norm(np.array([y_loc, x_loc]).T - np.array([rp, rp]), axis=1)
            
            dist_argsort = np.argsort(dist)

            for d, i in zip(dist[dist_argsort], dist_argsort):
                if d <= rp_squared:
                    new_pos = np.array([y_loc[i] - rp + yy, x_loc[i] - rp + xx, layer], dtype=np.int64) # save the new position
                    
                    # if close_to_top:
                    #     new_pos[0] -= rp
                    # elif close_to_bottom:
                    #     new_pos[0] += rp
                    
                    neigh_list[idx, int(found_neigh * 3): int(found_neigh * 3) + 3] = new_pos
                    found_neigh += 1
                    if found_neigh == n_neigh:
                        break
                    
            if found_neigh == n_neigh:
                break
    
    # end_pos_global = [pos[0] - r + y_loc_close[0], pos[1] - r + x_loc_close[0],
    #                     pos[0] - r + y_loc_close[1], pos[1] - r + x_loc_close[1]]
    return neigh_list



@nb.njit
def FNES_no_roll(coord_list: npt.NDArray, c_multi: npt.NDArray, rp:int, n_neigh:int) -> npt.NDArray: # no rolling
    
    neigh_list = np.zeros((coord_list.shape[0], int(3*n_neigh)), dtype=np.int64) - 1
    # s = int(2 * rp + 1)
    c_y, c_x, c_z = c_multi.shape
    rp_squared = rp**2
    for idx, (yy, xx, zz) in enumerate(coord_list): # loops over all cells that should divide
        if yy == -1: # pos has previously been unable to divide - no need to look again
            continue
        
        found_neigh = 0

        for layer in range(zz, c_z): # loops over cells layer and layers above

            c_grid = np.copy(c_multi[:,:,layer])
        
            p_dim = int(2*rp + 1) # pushing dim is square
            
            local_mask = np.zeros((p_dim, p_dim), dtype=nb.bool_)
            # local_pos = np.array([r, r])
            if xx - rp < 0: # close to lh edge
                missing_cols_l = rp - xx
                missing_cols_r = 0
                local_mask[:,:missing_cols_l] = 1
                # local_mask[:,missing_cols_l:] = c_grid[yy-rp:yy+rp+1, 0:xx+rp+1]
                
            elif xx + rp >= c_x: # close to rh edge
                missing_cols_r = (xx + rp +1) % c_x
                missing_cols_l = 0
                # local_mask[:,:-missing_cols_r] = c_grid[yy-rp:yy+rp+1, xx-rp:]
                local_mask[:,-missing_cols_r:] = 1
            
            if yy - rp < 0: # close to top
                missing_rows_t = rp - yy
                missing_rows_b = 0
                local_mask[:missing_rows_t, :] = 1
            
            elif yy + rp >= c_y: # close to bot
                missing_rows_b = (yy + rp + 1) % c_y
                missing_rows_t = 0
                local_mask[-missing_rows_b:, :] = 1
            
            # print(missing_rows_t, missing_rows_b)
            # print(missing_cols_l, missing_cols_r)
                
            local_mask[missing_rows_t:local_mask.shape[0]-missing_rows_b, missing_cols_l:local_mask.shape[1]-missing_cols_r] = np.copy(c_grid[yy-rp + missing_rows_t : yy+rp+1 - missing_rows_b,
                                                                                                              xx-rp + missing_cols_l : xx+rp+1 - missing_cols_r]).astype(nb.bool_)
            # print('test')
            # else: # general, when not close to edges
            #     local_mask = np.copy(c_grid[yy - rp: yy + rp + 1, xx - rp: xx + rp + 1]).astype(nb.bool_) # copies a square centered at dividing cell
            #     missing_cols_r = 0
            #     missing_cols_l = 0
            # get local coordinates where mask is 0
            y_loc, x_loc = np.where(local_mask == 0) # can be optimized based on whether missing_cols_l/r are zero

            if len(y_loc) == 0:
                # print('no free sites at layer')
                continue
            
            # if len(y_loc) == 0: # no free sites - this is wrong, as there's more layers to be searched
            #     found_neigh = n_neigh
            #     neigh_list[idx,:] = -1
            #     break
            
            # cal distance to all free sites on layer
            dist = (y_loc - rp)**2 + (x_loc - rp)**2  # there's probably some optimized function for this - look for it when you have the time
            
            dist_argsort = np.argsort(dist)

            for i in dist_argsort:
                if dist[i] <= rp_squared:
                    new_pos = np.array([y_loc[i] - rp + yy, x_loc[i] - rp + xx, layer], dtype=np.int64) # save the new position
                    
                    # if close_to_top:
                    #     new_pos[0] -= rp
                    # elif close_to_bottom:
                    #     new_pos[0] += rp
                    
                    neigh_list[idx, int(found_neigh * 3): int(found_neigh * 3) + 3] = new_pos
                    found_neigh += 1
                    if found_neigh == n_neigh:
                        break
                    
            if found_neigh == n_neigh:
                break
    
    # end_pos_global = [pos[0] - r + y_loc_close[0], pos[1] - r + x_loc_close[0],
    #                     pos[0] - r + y_loc_close[1], pos[1] - r + x_loc_close[1]]
    return neigh_list


""" EARLY STOPPING CONDITIONS """

def end_sim_if_touching_border(c_grid: npt.ArrayLike) -> bool:
    # name says it all
     top = np.sum(c_grid[0,:])
     bot = np.sum(c_grid[-1,:])
     left = np.sum(c_grid[:,0])
     right = np.sum(c_grid[:,-1])
     
     end_sim = True if np.array([top, bot, left, right]).any() != 0 else False
     
     return end_sim
 
 

""" MISC FUNCTIONS """

 
def continue_run(run_name: str) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    path = rf"C:\Users\Brage\Google Drive\Studie\Masters\data\{run_name}"
    
    c_grid = np.load(f'{path}\{run_name}_cell_grid.npy')
    a_grid = np.load(f'{path}\{run_name}_antibiotic_grid.npy')
    n_grid = np.load(f'{path}\{run_name}_nutrient_grid.npy')
    
    print('Found previous run')
    print('please implement a way to assert that parameters are identical...')
    try:
        cdim_r, cdim_c, n_layers = c_grid.shape
        # layers = True
        # print(f'0 height')
    except:
        cdim_r, cdim_c = c_grid.shape
        # layers = False
    print('recreating final conditions...')
    
    for yy in tqdm(range(cdim_r)):
        for xx in range(cdim_c):
            for zz in range(n_layers):
                type = c_grid[yy,xx, zz]
                if type == 1: # R
                    c_grid[yy,xx] = np.random.randint(1, 11)
                elif type == 2: # S
                    c_grid[yy,xx] = np.random.randint(11, 21)
                
    return c_grid, a_grid, n_grid


def continue_run_general(run_name):
    # path = rf"C:\Users\Brage\Google Drive\Studie\Masters\data\{run_name}"
    current_folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(current_folder, 'data'), run_name)
    
    c_data = np.load(f'{path}\{run_name}_cell_grid.npy') # might be 3d or sparse
    a_grid = np.load(f'{path}\{run_name}_antibiotic_grid.npy') # always comes in 2d array
    n_grid = np.load(f'{path}\{run_name}_nutrient_grid.npy') # always comes in 2d array
    run_info = np.load(f'{path}\{run_name}_run_info.npy')
    
    print('Found previous run')
    print('please implement a way to assert that parameters are identical...')
    
    
    if c_data.shape[1] == 4: # data is sparse
        print('sparse c_grid data found')
    # layers = True
    # print(f'0 height')
    
    else: # no mainpulation needed UNLESS IT ONLY CONTAINS 0, 1 AND 2
        c_grid = np.copy(c_data)

        if c_grid.max() == 2: 
            print('c_grid had been made final, initializing random growth states')
            for yy in tqdm(range(cdim_r)): # not needed anymore
                for xx in range(cdim_c):
                    for zz in range(n_layers):
                        type = c_grid[yy,xx, zz]
                        if type == 1: # R
                            c_grid[yy,xx] = np.random.randint(1, 11)
                        elif type == 2: # S
                            c_grid[yy,xx] = np.random.randint(11, 21)
                
    return c_grid, a_grid, n_grid
 
 
def save_snap(c_grid: npt.NDArray, a_grid: npt.NDArray, n_grid: npt.NDArray, snappath, snap_cnt:int, final_state: bool=False) -> None:
    

    if c_grid.ndim == 3:
        fill_y, fill_x, fill_z = np.nonzero(c_grid) # builds height first
        fill_val = c_grid[fill_y, fill_x, fill_z].flatten()
        data = np.array([fill_y, fill_x, fill_z, fill_val], dtype=np.int16)
    
    elif c_grid.ndim == 2:
        fill_y, fill_x = np.nonzero(c_grid)  # builds height first
        fill_val = c_grid[fill_y, fill_x].flatten()
        data = np.array([fill_y, fill_x, fill_val], dtype=np.int16)
        
    if not final_state:
        with open(os.path.join(snappath, f"snap{snap_cnt}.npy"), "ab") as f:
            # np.savetxt(f, data[np.newaxis])
            np.save(f, data)
        
        a_loc = os.path.join(snappath, f"a{snap_cnt}.npy")
        np.save(a_loc, a_grid)
        # n_loc = os.path.join(snappath, f"n{snap_cnt}.npy")
        # np.save(n_loc, n_grid)
        snap_cnt += 1
    
    if final_state:
        return data
    else:
        return None

@nb.njit
def find_bbox(c_grid: npt.NDArray, ddx: int, prev: tuple[int, int, int, int, int], use_prev: bool, ) -> tuple[int, int, int, int, int]:
    """
    Finds the bounding box of cells and returns the closest indicies aligning with the diffusion grids.
    
    Assumes z=0 is largest for speed.
    use_prev optimized by searching outwards using the previous bbox as initial guess
    

    Args:
        c_grid (npt.NDArray): _description_
        ddx (int): _description_

    Returns:
        tuple[slice, slice, slice]: _description_
    """
    cy, cx, cz = c_grid.shape

    if use_prev: # use previous bbox and iterate outwards instead
        prow_min, prow_max, pcol_min, pcol_max, pz_max = prev
        run_z = False
        
        for row in range(prow_min, -1 , -1):
            val = np.sum(c_grid[row,:,0].astype(np.bool_))
            if val == 0:
                row_min = row
                break
        
        for row in range(prow_max, cy, 1):
            val = np.sum(c_grid[row,:,0].astype(np.bool_))
            if val == 0:
                row_max = row
                break
        
        for col in range(pcol_min, -1, -1):
            val = np.sum(c_grid[:,col,0].astype(np.bool_))
            if val == 0:
                col_min = col
                break
            
        for col in range(pcol_max, cx, 1):
            val = np.sum(c_grid[:,col,0].astype(np.bool_))
            if val == 0:
                col_max = col
                break
        
        for height in range(pz_max, cz, 1):
            run_z = True
            val = np.sum(c_grid[:,:,height].astype(np.bool_))
            if val == 0:
                z_max = height
                break
        if not run_z:
            z_max = cz
            
    else:
        row_min = 0 
        row_max = 0
        col_min = 0
        col_max = 0
        z_max = cz
        
        for row in range(cy):
            val = np.sum(c_grid[row,:,0].astype(np.bool_))
            if val > 0:
                row_min = row
                break
        
        for row in range(cy-1,-1,-1):
            val = np.sum(c_grid[row,:,0].astype(np.bool_))
            if val > 0:
                row_max = row
                break
        
        for col in range(cx):
            val = np.sum(c_grid[:,col,0].astype(np.bool_))
            if val > 0:
                col_min = col
                break
        
        for col in range(cx-1,-1,-1):
            val = np.sum(c_grid[:,col,0].astype(np.bool_))
            if val > 0:
                col_max = col
                break
        
        for height in range(cz):
            val = np.sum(c_grid[:,:,height].astype(np.bool_))
            if val == 0:
                z_max = height
                break
        
    # r_min_off = r_min % ddx
    # row_min_m = row_min - row_min % ddx
    # row_max_m = row_max + ddx - (row_max % ddx)
    # col_min_m = col_min - col_min % ddx
    # col_max_m = col_max +ddx - (col_max % ddx) 
    
    best_guess = np.array([row_min, row_max, col_min, col_max, z_max], dtype=np.int64)
    return row_min, row_max, col_min, col_max, z_max, best_guess
        
@nb.njit
def find_bbox_growth_2drop(c_grid: npt.NDArray, type: str) -> tuple[int, int, int, int, int]:
    """
    Finds the bounding box of the specified cell type and returns the closest indicies aligning with the diffusion grids.
    
    Assumes z=0 is largest for speed.
    use_prev optimized by searching outwards using the previous bbox as initial guess
    
    Args:
        c_grid (npt.NDArray): _description_
        ddx (int): _description_

    Returns:
        tuple[slice, slice, slice]: _description_
    """
    cy, cx, cz = c_grid.shape
    
    if type == 'r':
        min_val = 1
        max_val = 10
    elif type == 's':
        min_val = 11
        max_val = 21
    else:
        min_val = 1
        max_val = 21
        
    row_min = 0
    for row in range(0, cy):
        val = np.any((c_grid[row, :, 0] >= min_val) & (c_grid[row, :, 0] <= max_val))
        if val:
            row_min = row - 1
            break
        
    row_max = cy
    for row in range(cy-1, -1, -1):
        val = np.any((c_grid[row, :, 0] >= min_val) & (c_grid[row, :, 0] <= max_val))
        if val:
            row_max = row + 1
            break
    
    col_min = 0
    for col in range(0, cx):
        val = np.any((c_grid[:, col, 0] >= min_val) & (c_grid[:, col, 0] <= max_val))
        if val:
            col_min = col - 1
            break
        
    col_max = cx
    for col in range(cx-1, -1, -1):
        val = np.any((c_grid[:, col, 0] >= min_val) & (c_grid[:, col, 0] <= max_val))
        if val:
            col_max = col + 1
            break

        
    z_max = cz
    for height in range(cz):
        val = np.any((c_grid[:, :, height] >= min_val) & (c_grid[:, :, height] <= max_val))
        if not val:
            z_max = height
            break
    
    best_guess = np.array([row_min+1, row_max-1, col_min+1, col_max-1, z_max], dtype=np.int64)
    return row_min, row_max, col_min, col_max, z_max #, best_guess
        

@nb.njit
def find_bbox_growth(c_grid: npt.NDArray, ) -> tuple[int, int, int, int, int]:
    """
    Finds the bounding box of cells and returns the closest indicies aligning with the diffusion grids.
    
    Assumes z=0 is largest for speed.
    use_prev optimized by searching outwards using the previous bbox as initial guess
    

    Args:
        c_grid (npt.NDArray): _description_
        ddx (int): _description_

    Returns:
        tuple[slice, slice, slice]: _description_
    """
    cy, cx, cz = c_grid.shape

    # if use_prev: # use previous bbox and iterate outwards instead
    #     prow_min, prow_max, pcol_min, pcol_max, pz_max = prev
    #     run_z = False
        
    #     for row in range(prow_min+1, -1 , -1):
    #         val = np.sum(c_grid[row,:,0].astype(np.bool_))
    #         if val == 0:
    #             row_min = row
    #             break
        
    #     for row in range(prow_max, cy, 1):
    #         val = np.sum(c_grid[row,:,0].astype(np.bool_))
    #         if val == 0:
    #             row_max = row
    #             break
        
    #     for col in range(pcol_min, -1, -1):
    #         val = np.sum(c_grid[:,col,0].astype(np.bool_))
    #         if val == 0:
    #             col_min = col
    #             break
            
    #     for col in range(pcol_max, cx, 1):
    #         val = np.sum(c_grid[:,col,0].astype(np.bool_))
    #         if val == 0:
    #             col_max = col
    #             break
        
    #     for height in range(pz_max, cz, 1):
    #         run_z = True
    #         val = np.sum(c_grid[:,:,height].astype(np.bool_))
    #         if val == 0:
    #             z_max = height
    #             break
    #     if not run_z:
    #         z_max = cz
    if False:
        None
        
    else:
        row_min = 0 
        row_max = 0
        col_min = 0
        col_max = 0
        z_max = cz
        
        for row in range(cy):
            val = np.sum(c_grid[row,:,0].astype(np.bool_))
            if val > 0:
                row_min = row - 1
                break
        
        for row in range(cy-1,-1,-1):
            val = np.sum(c_grid[row,:,0].astype(np.bool_))
            if val > 0:
                row_max = row + 1
                break
        
        for col in range(cx):
            val = np.sum(c_grid[:,col,0].astype(np.bool_))
            if val > 0:
                col_min = col - 1
                break
        
        for col in range(cx-1,-1,-1):
            val = np.sum(c_grid[:,col,0].astype(np.bool_))
            if val > 0:
                col_max = col + 1
                break
        
        for height in range(cz):
            val = np.sum(c_grid[:,:,height].astype(np.bool_))
            if val == 0:
                z_max = height
                break
        
    # r_min_off = r_min % ddx
    # row_min_m = row_min - row_min % ddx
    # row_max_m = row_max + ddx - (row_max % ddx)
    # col_min_m = col_min - col_min % ddx
    # col_max_m = col_max +ddx - (col_max % ddx) 
    
    # best_guess = np.array([row_min, row_max, col_min, col_max, z_max], dtype=np.int64)
    return row_min, row_max, col_min, col_max, z_max



def bbox_splits(c_grid, a_grid, n_grid, core_cnt, ddx, prev: tuple[int, int, int, int, int], use_prev: bool):
    row_min, row_max, col_min, col_max, z_max, best_guess = find_bbox(c_grid, ddx, prev, use_prev) # find bbox of cells - corresponding to the indices on c_grid
    
    bbox_diffusion = (slice(int(row_min/ddx)+1,int(row_max/ddx)+1), slice(int(col_min/ddx)+1,int(col_max/ddx)+1), slice(0,z_max)) # +1 due to ghost cells
    bbox_cell = (slice(row_min, row_max), slice(col_min, col_max), slice(0,z_max))
    solution_found = False
    empty_count = 0
    
    while not solution_found:
        
        core_cnt -= empty_count
        split_c = np.array_split(c_grid[bbox_cell], core_cnt)
        split_a = np.array_split(a_grid[bbox_diffusion[:2]], core_cnt)
        split_n = np.array_split(n_grid[bbox_diffusion[:2]], core_cnt)

        split_points_d = [int(split_a[i].shape[0]) for i in range(len(split_a))] # +1 due to ghost cells

        # print(split_points_d)
        row_split_d = [int(row_min / ddx) + int(np.sum(split_points_d[:i])) for i in range(len(split_a))] # wo/ ghost cells, so 0 corresponds to 0 on cell grid
        row_split_d.append(row_split_d[-1] + split_points_d[-1])

        # print(row_split_d)

        split_points_c = [int(split_a[i].shape[0] * ddx) for i in range(len(split_a))] 
        row_split_c = [row_min + int(np.sum(split_points_c[:i])) for i in range(len(split_a))]
        row_split_c.append(row_split_c[-1] + split_points_c[-1])

        # print(row_split_c)
        split_c = [c_grid[row_split_c[i]:row_split_c[i+1],col_min:col_max, 0:z_max] for i in range(len(split_points_c))]
        
        empty_count = sum(arr.size == 0 for arr in split_a)
        
        if empty_count > 0:
            solution_found = True # workaround which works as long as there's always one or more row of ddx per coress
            # print(f'sol not found. empty cnt = {empty_count}')
        else:
            solution_found = True
            
    n_proc = len(split_a)
    
    return split_c, split_a, split_n, split_points_c, split_points_d, row_split_c, row_split_d, row_min, row_max, col_min, col_max, z_max, best_guess, n_proc
 
 
 
 
def bbox_splits2(c_grid, a_grid, n_grid, core_cnt, ddx, prev: tuple[int, int, int, int, int], use_prev: bool):
    
    row_min, row_max, col_min, col_max, z_max, best_guess = find_bbox(c_grid, ddx, prev, use_prev) # find bbox of cells - corresponding to the indices on c_grid
    
    bbox_diffusion = np.array([[row_min // ddx + 1, row_max // ddx + 1],
                               [col_min // ddx + 1, col_max // ddx + 1]],
                               dtype=np.int64)
    
    bbox_cell = np.array([[row_min - row_min % ddx, row_max + ddx - row_max % ddx],
                         [col_min - col_min % ddx, col_max + ddx - col_max % ddx],
                         [0, z_max]], dtype=np.int64)
    

    current_row = bbox_cell[0,0]
    
    # divide grids into sub grids. These are lists of arrays where rows are separated
    split_a = np.array_split(a_grid[bbox_diffusion[0,0]: bbox_diffusion[0,1], bbox_diffusion[1,0]:bbox_diffusion[1,1]], core_cnt)
    split_n = np.array_split(n_grid[bbox_diffusion[0,0]: bbox_diffusion[0,1], bbox_diffusion[1,0]:bbox_diffusion[1,1]], core_cnt)
    split_c = []
    split_c_loc = np.zeros(core_cnt+1, dtype=np.int64)
    split_d_loc = np.zeros(core_cnt+1, dtype=np.int64)
    
    # now match the number of rows in each d_split with ddx*n_rows in c
    for idx, split in enumerate(split_a):
        n_rows_d = split.shape[0]
        split_d_loc[idx+1] = split_d_loc[idx] + n_rows_d
        split_c_loc[idx+1] = split_c_loc[idx] + n_rows_d * ddx
        split_c.append(c_grid[current_row: current_row + n_rows_d * ddx, bbox_cell[1,0]:bbox_cell[1,1], bbox_cell[2,0]:bbox_cell[2,1]])
        current_row += n_rows_d * ddx

    return bbox_cell, bbox_diffusion, split_c, split_a, split_n, split_c_loc, split_d_loc, z_max, best_guess
 
 
 
