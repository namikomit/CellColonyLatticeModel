"""
RSR Stripe Closure Experiment with Timepoint Snapshots

Saves complete spatial data (cells + antibiotic) at specified timepoints
for creating publication-quality temporal sequences.
"""

import re
import os
import numpy as np
import LM_func22 as lm
import ray
from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator

# Import the RSR initial condition function
import sys
sys.path.append('/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/code')
from RSR_initial_conditions import create_RSR_vertical_stripe_geometry

env = "cluster"
working_dir = r"/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/code"

def apply_preconditioner(v):
    return LU_comp.solve(v)

" RAY SETUP "
# Use custom temp directory to avoid filling up /tmp
# Path must be short (< 107 bytes for Unix socket limit)
import tempfile
ray_temp_dir = "/tmp/ray_brage"  #Short path that won't fill /tmp
os.makedirs(ray_temp_dir, exist_ok=True)

ray.init(
    runtime_env={"working_dir": working_dir},
    _temp_dir=ray_temp_dir
)

# ============================================================================
# EXPERIMENTAL PARAMETERS
# ============================================================================

# Random seed (set to None for random, or set to integer for reproducibility)
random_seed = None  # Change to e.g. 42 for reproducible runs

# Geometry parameters
L_height = 800              
total_width = 1500          
initial_density = 0.9       

# Sensitive stripe widths to test
sensitive_widths = [100, 200, 400, 800]  # [μm]

# Simulation time
simulation_hours = 8.1       

# TIMEPOINTS TO SAVE (in hours) - EDIT THIS LIST!
timepoints_to_save = [0, 1, 2, 4, 6, 8, 10]  # hours

# ============================================================================

# Convert timepoints to timestep indices
dt = 0.5  # minutes
timepoints_timesteps = [int(t * 60 / dt) for t in timepoints_to_save]

print(f"Will save snapshots at timepoints: {timepoints_to_save} hours")
print(f"Corresponding to timesteps: {timepoints_timesteps}")

# Run each width configuration
for w_sensitive in sensitive_widths:
    
    print(f"\n{'='*60}")
    print(f"RUNNING: Sensitive stripe width = {w_sensitive} μm")
    print(f"{'='*60}\n")
    
    " RUN NAMING "
    exp_setup = "RSR_stripe"
    
    " COMPUTATIONAL PARAMETERS "
    core_cnt = 25
    wrk_r = int(np.sqrt(core_cnt))
    n_layers = 1
    pushing_radius = 50
    timesteps = int(simulation_hours * 60 / dt)
    
    " DIFFUSION GRID "
    ddim_r = 1600
    ddim_c = 1600
    ddx = 30
    r_offset, c_offset = ddim_r // 3, ddim_c // 3
    
    " DIFFUSION PARAMETERS "
    D = 200 * 60
    vertical_loss = 3
    xb = 100
    
    " ANTIBIOTIC PARAMETERS "
    a0 = 2.0
    n0 = 1.0
    add_a_at_t = 0
    
    " CELL PARAMETERS "
    lambda_r = 1.93
    lambda_s = 1.93
    lambda_max_r = 10 * lambda_r / (np.log(2) * 60)
    lambda_max_s = 10 * lambda_s / (np.log(2) * 60)
    
    " ANTIBIOTIC DEGRADATION "
    g_nonnorm = 0.025
    gamma = g_nonnorm / ddx**2
    
    " ANTIBIOTIC INHIBITION "
    k_A = 2.35
    
    " NUTRIENT LIMITATION "
    k_N = 0.05
    mu = 0.01
    
    " RUN NAME "
    run_name = f"{exp_setup}_w{w_sensitive}_L{L_height}_A{a0}_g{g_nonnorm}_ka{k_A}"
    
    " CREATE GRIDS "
    c_grid, a_grid, n_grid, r_offset, c_offset = lm.create_grids_differnt_dim(
        ddim_r, ddim_c, ddim_r*ddx, ddim_c*ddx, a0, n0, n_layers
    )
    
    # Create RSR stripe geometry
    c_grid[:,:,0], initial_cell_cnt = create_RSR_vertical_stripe_geometry(
        c_grid[:,:,0], 
        L=L_height, 
        w=w_sensitive,
        total_width=total_width,
        density=initial_density
    )
    
    div_mask = np.zeros(c_grid.shape[0:2], dtype=np.int8)
    
    # Create diffusion matrix
    fin_diff_mat, _ = lm.create_finite_differences_matrix_no_flux(
        ddim_r, ddim_c, D, ddx, dt, core_cnt
    )
    LU_comp = spilu(fin_diff_mat.tocsc())
    M = LinearOperator(fin_diff_mat.shape, matvec=apply_preconditioner)
    
    " SAVE RUN INFO "
    run_info = np.array([
        ['exp_setup', exp_setup],
        ['dt', dt],
        ['timesteps', timesteps],
        ['simulation_hours', simulation_hours],
        ['ddx', ddx],
        ['ddim_r', ddim_r],
        ['ddim_c', ddim_c],
        ['D', D],
        ['vertical_loss', vertical_loss],
        ['pushing_radius', pushing_radius],
        ['cell_cnt', initial_cell_cnt],
        ['k_A', k_A],
        ['k_N', k_N],
        ['gamma', gamma],
        ['lambda_max_r', lambda_max_r],
        ['lambda_max_s', lambda_max_s],
        ['init_A', a0],
        ['init_N', n0],
        ['n_layers', n_layers],
        ['x_bound', xb],
        ['add_a_at_t', add_a_at_t],
        ['L_height', L_height],
        ['sensitive_width', w_sensitive],
        ['total_width', total_width],
        ['initial_density', initial_density],
        ['timepoints_hours', str(timepoints_to_save)],
    ], dtype=object)
    
    " SETUP SAVING "
    save_dir = r"/nbi/nbicmplx/cell/mitarai/IndirectResistance/Brage/CellColonyLatticeModel/data/RSR_timeseries_2"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectory for this run's timepoints
    timepoints_dir = os.path.join(save_dir, run_name + "_timepoints")
    os.makedirs(timepoints_dir, exist_ok=True)
    
    " INITIALIZE TRACKING "
    ran_order = np.arange(0, 9, 1)
    diff_splits = np.arange(0, ddim_c*ddx, ddx)
    r_cnt = 0
    s_cnt = 0
    
    # Track populations over time
    r_cnt_at_t = np.zeros(timesteps)
    s_cnt_at_t = np.zeros(timesteps)
    sensitive_width_at_t = np.zeros(timesteps)
    
    " MAIN SIMULATION LOOP "
    print(f"Starting simulation: {timesteps} timesteps ({simulation_hours} hours)")
    
    for t in tqdm(range(timesteps)):
        
        # Find bounding box
        row_min, row_max, col_min, col_max, z_max = lm.find_bbox_growth(c_grid)
        
        if row_max == 0 or col_max == 0:
            print(f"No cells found at timestep {t}")
            break
        
        max_height = (z_max == n_layers)
        layer_buffer = 1
        
        # Use original code's splitting logic
        largest_dim = np.max([row_max-row_min, col_max-col_min])
        row_max = row_min + largest_dim 
        col_max = col_min + largest_dim
        
        growth_box_size = int(np.ceil((row_max - row_min) / (3 * wrk_r)))
        
        row_max += growth_box_size - ((row_max - row_min) % growth_box_size)
        col_max += growth_box_size - ((col_max - col_min) % growth_box_size)
        
        row_splits = np.arange(row_min, row_max+1, growth_box_size)
        col_splits = np.arange(col_min, col_max+1, growth_box_size)
        
        # Process cells
        strains = ['r', 's', 'rs']
        
        for strain in strains:
            for row_mod in range(0, 3):
                for col_mod in range(0, 3):
                    
                    rows = np.arange(0, 3 * wrk_r, 3, dtype=np.int64) + row_mod
                    cols = np.arange(0, 3 * wrk_r, 3, dtype=np.int64) + col_mod
                    
                    Gbox_ul_row = np.array([row_splits[rows[wrk // wrk_r]] 
                                           for wrk in range(core_cnt)])
                    Gbox_ul_col = np.array([col_splits[cols[wrk % wrk_r]] 
                                           for wrk in range(core_cnt)])
                    
                    ul_dbox_row = Gbox_ul_row // ddx
                    ul_dbox_col = Gbox_ul_col // ddx
                    
                    used_dsplits_row = [np.arange(ul_dbox_row[i] * ddx, 
                                                  row_splits[rows[i // wrk_r] + 1], ddx) 
                                       for i in range(core_cnt)]
                    used_dsplits_col = [np.arange(ul_dbox_col[i] * ddx, 
                                                  col_splits[cols[i % wrk_r] + 1], ddx) 
                                       for i in range(core_cnt)]
                    
                    Gbox_Dboxes_a = [a_grid[r_offset + ul_dbox_row[i]: 
                                            r_offset + ul_dbox_row[i] + len(used_dsplits_row[i]),
                                           c_offset + ul_dbox_col[i]: 
                                           c_offset + ul_dbox_col[i] + len(used_dsplits_col[i])]
                                    for i in range(core_cnt)]
                    
                    Gbox_Dboxes_n = [n_grid[r_offset + ul_dbox_row[i]: 
                                            r_offset + ul_dbox_row[i] + len(used_dsplits_row[i]),
                                           c_offset + ul_dbox_col[i]: 
                                           c_offset + ul_dbox_col[i] + len(used_dsplits_col[i])]
                                    for i in range(core_cnt)]
                    
                    if max_height:
                        div_grid = [c_grid[row_splits[rows[wrk // wrk_r]] - pushing_radius: 
                                          row_splits[rows[wrk // wrk_r] + 1] + pushing_radius,
                                          col_splits[cols[wrk % wrk_r]] - pushing_radius: 
                                          col_splits[cols[wrk % wrk_r] + 1] + pushing_radius, 0:z_max]
                                   for wrk in range(core_cnt)]
                    else:
                        div_grid = [c_grid[row_splits[rows[wrk // wrk_r]] - pushing_radius: 
                                          row_splits[rows[wrk // wrk_r] + 1] + pushing_radius,
                                          col_splits[cols[wrk % wrk_r]] - pushing_radius: 
                                          col_splits[cols[wrk % wrk_r] + 1] + pushing_radius, 
                                          0:z_max + layer_buffer]
                                   for wrk in range(core_cnt)]
                    
                    div_mask_small = [div_mask[row_splits[rows[wrk // wrk_r]] - pushing_radius: 
                                               row_splits[rows[wrk // wrk_r] + 1] + pushing_radius,
                                              col_splits[cols[wrk % wrk_r]] - pushing_radius: 
                                              col_splits[cols[wrk % wrk_r] + 1] + pushing_radius]
                                     for wrk in range(core_cnt)]
                    
                    task_args = [(div_grid[i], div_mask_small[i], 
                                 used_dsplits_row[i], used_dsplits_col[i],
                                 Gbox_Dboxes_a[i], Gbox_Dboxes_n[i], 
                                 dt, pushing_radius, k_A, k_N,
                                 lambda_max_r, lambda_max_s, vertical_loss) 
                                for i in range(core_cnt)]
                    
                    futures = [lm.remote_do_everything.remote(*args) for args in task_args]
                    results = ray.get(futures)
                    
                    for worker, result in enumerate(results):
                        if max_height:
                            c_grid[row_splits[rows[worker // wrk_r]] - pushing_radius: 
                                  row_splits[rows[worker // wrk_r] + 1] + pushing_radius,
                                  col_splits[cols[worker % wrk_r]] - pushing_radius: 
                                  col_splits[cols[worker % wrk_r] + 1] + pushing_radius, 
                                  0:z_max] = result[0]
                        else:
                            c_grid[row_splits[rows[worker // wrk_r]] - pushing_radius: 
                                  row_splits[rows[worker // wrk_r] + 1] + pushing_radius,
                                  col_splits[cols[worker % wrk_r]] - pushing_radius: 
                                  col_splits[cols[worker % wrk_r] + 1] + pushing_radius, 
                                  0:z_max + layer_buffer] = result[0]
                        
                        div_mask[row_splits[rows[worker // wrk_r]] - pushing_radius: 
                                row_splits[rows[worker // wrk_r] + 1] + pushing_radius,
                                col_splits[cols[worker % wrk_r]] - pushing_radius: 
                                col_splits[cols[worker % wrk_r] + 1] + pushing_radius] = result[1]
                        
                        # Antibiotic degradation
                        a_grid[r_offset + ul_dbox_row[worker]: 
                              r_offset + ul_dbox_row[worker] + Gbox_Dboxes_a[worker].shape[0],
                              c_offset + ul_dbox_col[worker]: 
                              c_offset + ul_dbox_col[worker] + Gbox_Dboxes_a[worker].shape[1]] -= \
                            result[4] * gamma * Gbox_Dboxes_a[worker] * dt
                        
                        sub_a = a_grid[r_offset + ul_dbox_row[worker]: 
                                      r_offset + ul_dbox_row[worker] + Gbox_Dboxes_a[worker].shape[0],
                                      c_offset + ul_dbox_col[worker]: 
                                      c_offset + ul_dbox_col[worker] + Gbox_Dboxes_a[worker].shape[1]]
                        sub_a[sub_a < 0] = 0
                        
                        r_cnt += result[2]
                        s_cnt += result[3]
                        
                        n_grid[r_offset + ul_dbox_row[worker]: 
                              r_offset + ul_dbox_row[worker] + Gbox_Dboxes_n[worker].shape[0],
                              c_offset + ul_dbox_col[worker]: 
                              c_offset + ul_dbox_col[worker] + Gbox_Dboxes_n[worker].shape[1]] -= \
                            result[5] * mu
        
        # Diffusion step (AFTER cell growth)
        if add_a_at_t <= t * dt:
            a_grid = lm.sol_diff_no_flux(a_grid, fin_diff_mat, M)
        
        # Track populations
        r_cnt_at_t[t] = r_cnt
        s_cnt_at_t[t] = s_cnt
        
        # SAVE SNAPSHOT at specified timepoints (AFTER diffusion)
        if t in timepoints_timesteps:
            time_hours = t * dt / 60
            print(f"\n  Saving snapshot at t = {time_hours:.1f} hours (timestep {t})")
            
            # Save cell grid (use final_state=True to get data returned)
            cell_snapshot_file = os.path.join(timepoints_dir, f"{run_name}_t{time_hours:.1f}h_cells.npy")
            np.save(cell_snapshot_file, 
                   lm.save_snap(c_grid, a_grid, n_grid, timepoints_dir, t, final_state=True))
            
            # Save antibiotic grid
            antibio_snapshot_file = os.path.join(timepoints_dir, f"{run_name}_t{time_hours:.1f}h_antibiotic.npy")
            np.save(antibio_snapshot_file, a_grid.copy())
            
            print(f"    Saved: {run_name}_t{time_hours:.1f}h_*.npy")
        
        # Track sensitive stripe width
        if t % 10 == 0:
            sensitive_cells = (c_grid[:,:,0] >= 11) & (c_grid[:,:,0] <= 20)
            if np.any(sensitive_cells):
                cols_with_s = np.any(sensitive_cells, axis=0)
                x_coords = np.where(cols_with_s)[0]
                if len(x_coords) > 0:
                    x_min_s = np.min(x_coords)
                    x_max_s = np.max(x_coords)
                    sensitive_width_at_t[t] = x_max_s - x_min_s
    
    " SAVE FINAL DATA "
    print(f"\nSimulation complete. Saving summary data...")
    
    np.save(os.path.join(save_dir, run_name + '_run_info.npy'), run_info)
    np.save(os.path.join(save_dir, run_name + '_r_at_t.npy'), r_cnt_at_t)
    np.save(os.path.join(save_dir, run_name + '_s_at_t.npy'), s_cnt_at_t)
    np.save(os.path.join(save_dir, run_name + '_sensitive_width_at_t.npy'), sensitive_width_at_t)
    
    print(f"Saved to: {save_dir}")
    print(f"Timepoint snapshots in: {timepoints_dir}")

print("\n" + "="*60)
print("ALL SIMULATIONS COMPLETE!")
print("="*60)
