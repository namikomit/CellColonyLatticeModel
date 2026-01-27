import re
import os
import numpy as np
import LM_func22 as lm
import ray
from tqdm import tqdm
from scipy.sparse.linalg import spilu, LinearOperator

env = "cluster"
working_dir = r"/nbi/user-scratch/c/chv625/LM" # path to directory where this and LM_func22.py are found

def apply_preconditioner(v):
    return LU_comp.solve(v)

" RAY SETUP STUFF "
ray.init(runtime_env={"working_dir": working_dir})

a_conc = [0] # initial concentrations of antibiotica

for run_no, a_init in enumerate(a_conc):
    
    " RUN NAMING AND PARAMETERS"
    exp_setup = "2drop" # 2drop for collision, "1_drop_co" for mixed


    " GENERAL COMPUTATIONAL PARAMETERS "
    core_cnt = 25 # no of cores used for simulation
    wrk_r = int(np.sqrt(core_cnt))
    dt = 0.5 # temporal step size [min]
    timesteps = int(24 * 60 / dt) # 24h simulation time
    snap_rate = timesteps # saves the initial state, not used as default


    " GRID AND STEPSIZE PARAMETERS "
    ddim_r = 1600 # no. of diffusion rows
    ddim_c = 1600 # no. of diffusion cols
    n_layers = 1 # no. of layers that the colony can build to, not used as default
    ddx = 30 # grid size for diffusion [mu m]
    cdx = 1 # grid size for cells [mu m]
    
    r_offset, c_offset = ddim_r // 3, ddim_c // 3

    " DIFFUSION PARAMETERS "
    D = 200 * 60 # diffusion coefficient (mu m^2 / min)
    vertical_loss = 3 # not used for article sims
    xb = 100 # not used - will be removed

    " INITIAL CONDITIONS "
    initial_cell_cnt = int(150_000 * 0.5) # total no. of cell at t=0
    initial_density = 0 # not used - will be removed
    resistant_fraction = 0.33
    r_inner = 1000 # inner radius of CRE
    r_outer = 1100 # outer radius of CRE
    frac_within = 0.2 # fraction of cells located within the inner radius of the CRE
    
    add_a_at_t = 0 # minute at which antibiotics are added
    if add_a_at_t == 0:
        a0 = a_init
    else:
        a0 = 0
        
    n0 = 1 # initial nutrient count

    if exp_setup != "2drop":
        drop_separation = 0
    else:
        drop_separation = 0

    " CELL PARAMETERS "
    lambda_r = 1.93 # growth rate in [h^-1]
    lambda_s = 1.93 # growth rate in [h^-1]
    lambda_max_r = 10 * lambda_r / (np.log(2) * 60) # growth rate scaled for sim
    lambda_max_s = 10 * lambda_s / (np.log(2) * 60) # growth rate scaled for sim
    g_nonnorm = 0.01 # not scaled antibiotic removal
    gamma = g_nonnorm / ddx**2 # antibiotic removal scaled for sim
    mu = 0.0 / ddx**2 # nutrient consumption rate scaled for sim
    pushing_radius = 50 # [mu m]
    k_A = 2.35 # mug/mL from best fit of Mireia data
    k_N = 0.0


    if exp_setup == '2drop':
        run_name = f"{exp_setup}_CRE{r_outer}{r_inner}_rp{pushing_radius}_icc{initial_cell_cnt}_g{g_nonnorm}_ka{k_A}_A{a_init}"
        if lambda_max_s == 0:
            run_name = f"{exp_setup}_CRE{r_outer}{r_inner}_rp{pushing_radius}_icc{initial_cell_cnt}_g{g_nonnorm}_ka{k_A}_NOS_A{a_init}"
        elif lambda_max_r != lambda_max_s:
            run_name = f"{exp_setup}_CRE{r_outer}{r_inner}_rp{pushing_radius}_icc{initial_cell_cnt}_g{g_nonnorm}_ka{k_A}_ldiff_A{a_init}"
    else:
        run_name = f"{exp_setup}_CRE{r_inner}_r{resistant_fraction}_icc{initial_cell_cnt}_g{g_nonnorm}_ka{k_A}_A{a_init}"
        if lambda_max_r != lambda_max_s:
            run_name = f"{exp_setup}_CRE{r_inner}_r{resistant_fraction}_icc{initial_cell_cnt}_g{g_nonnorm}_ka{k_A}_ldiff_A{a_init}"
            
        
        



    # if ddim_r % core_cnt != 0:
    #     ddim_r += core_cnt - ddim_r % core_cnt
    #     ddim_c += core_cnt - ddim_c % core_cnt
        
    #     print(f'ddim_r is now  {ddim_r}')
    #     print(f'ddim_c is now  {ddim_c}')

    " INITIALIZE GRIDS AND INITIAL CONDITIONS "
    if True:
        c_grid, a_grid, n_grid, r_offset, c_offset = lm.create_grids_differnt_dim(ddim_r, ddim_c, ddim_r*ddx, ddim_c*ddx, a0, n0, n_layers)
        div_mask = np.zeros(c_grid.shape[0:2], dtype=np.int8)
        if exp_setup == "1drop_co":
            c_grid[:,:,0], initial_density = lm.coffee_ring(c_grid[:,:,0], initial_cell_cnt, resistant_fraction, r_inner, r_outer, frac_within)
        if exp_setup == "2drop":
            c_grid[:,:,0], initial_density = lm.diffuse_drops2(c_grid[:,:,0], initial_cell_cnt, r_inner, r_outer, drop_separation, frac_within)
        
        fin_diff_mat, _ = lm.create_finite_differences_matrix_no_flux(ddim_r, ddim_c, D, ddx, dt, core_cnt)
        LU_comp = spilu(fin_diff_mat.tocsc())
        M = LinearOperator(fin_diff_mat.shape, matvec=apply_preconditioner)

    " LIST OF RUN PARAMETERS TO SAVE "

    run_info = np.array([['cdim_r', int(ddim_r * ddx // 3)],
                    ['cdim_c', int(ddim_c * ddx // 3)],
                    ['ddim_r', ddim_r],
                    ['ddim_c', ddim_c],
                    ['timesteps', timesteps],
                    ['core_cnt', core_cnt], # 5
                    ['cdx', cdx],
                    ['dt', dt], # 7
                    ['D', D],
                    ['ddx', ddx],
                    ['pushing radius', pushing_radius],
                    ['timesteps', timesteps],
                    ['initial cell cnt, ', initial_cell_cnt],
                    ['initial density', initial_density],
                    ['resistant fraction, ', resistant_fraction],
                    ['experimental setup', exp_setup], 
                    ['gamma', gamma],
                    ['mu', mu],
                    ['k_A', k_A],
                    ['k_N', k_N], 
                    ['lambda_max_r', lambda_max_r], 
                    ['lambda_max_s', lambda_max_s],
                    ['r inner', r_inner],
                    ['r outer', r_outer],
                    ['init_A', a0],
                    ['init_N', n0],
                    ['drop separation', drop_separation],
                    ['n_layers', n_layers],
                    ['vertical diff loss', vertical_loss],
                    ['x_bound', xb],
                    ['add_a_at_t', add_a_at_t],
                    ])


    " INITIALIZING RANDOM STUFF THAT'S CONSTANT THROUGHOUT SIM "
    ran_order = np.arange(0, 9, 1) # used to determine the order of division in Gboxes
    diff_splits = np.arange(0, ddim_c*ddx, ddx) # on entire grid, assuming col will be largest, which is true for all setups
    init_vals = [a0, n0]
    r_cnt = len(np.where((c_grid > 0) & (c_grid < 11))[0])
    s_cnt = len(np.where((c_grid > 10) & (c_grid < 21))[0])
    r_cnt_at_t = np.zeros(timesteps, dtype=np.int64)
    s_cnt_at_t = np.zeros(timesteps, dtype=np.int64)
    height_at_t = np.zeros(timesteps, dtype=np.int64)
    snap_cnt = 0
    only_a_snap_cnt = 0
    
    layer_buffer = 5
    
    if exp_setup == "2drop":
        strains = ['r', 's']
    elif exp_setup == "1drop_co":
        strains = ['rs']
    
    collision = False
    sen_col_min = ddim_c * ddx
    res_col_max = 0
    collision_time = 0
    a_at_sen_edge = np.zeros(timesteps, dtype=float) # for 2drop
    
    if exp_setup == '1drop_co':
        range_exp = np.zeros((4, timesteps), dtype=np.int64)
    
    # a_at_colony_center
    

    " CREATE DIRECTORY TO SAVE DATA "
    folder_created = False
    run_no_for_save = 0

    current_folder = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(current_folder, 'data')

    # Extract base run_name without existing numbers
    base_run_name = re.sub(r'_\d+$', '', run_name)  # Removes the last _number if it exists

    while not folder_created:
        run_name = f'{base_run_name}_{run_no_for_save}'
        newpath = os.path.join(data_folder, run_name)
        snappath = os.path.join(newpath, "snaps")


        if not os.path.exists(newpath):
            os.makedirs(newpath)
            print(f'made directory in {newpath}')
            folder_created = True
        else:
            print(f'folder already exists in {newpath}, incrementing run_no_for_save')
            run_no_for_save += 1
            base_run_name = re.sub(r'_\d+$', '', run_name)  # Removes the last _number if it exists

    

    # # newpath = snap_loc = rf"C:\Users\Brage\Google Drive\Studie\Masters\data\{run_name}\snaps"
    # if not os.path.exists(snappath):
    #     os.makedirs(snappath)
    #     print(f'made directory in {newpath}')


    " Timesteps start here "

    for t in tqdm(range(timesteps)):
        
        if add_a_at_t == t * dt: # antibiotics are added at later time
            a_grid[:,:] = a0

        # r_min, r_max, c_min, c_max, z_max = lm.find_bbox_growth(c_grid)
        for cell_strain in strains:
            r_min, r_max, c_min, c_max, z_max = lm.find_bbox_growth_2drop(c_grid, type=cell_strain)
            height_at_t[t] = z_max
            # print(r_min, r_max, c_min, c_max, z_max)
            if (z_max >= n_layers - layer_buffer):
                max_height = True
                z_max = n_layers
            else:
                max_height = False
                
            if exp_setup == "2drop":
                if cell_strain == 'r': 
                    res_col_max = c_max
                elif cell_strain == 's':
                    sen_col_min = c_min # save for comparison later
                    a_at_sen_edge[t] = a_grid[int(ddx/2) + r_offset, c_min // ddx + c_offset]
                elif cell_strain == 'rs': # collision has happened, keep on saving conc at border
                    a_at_sen_edge[t] = a_grid[int(ddx/2) + r_offset, sen_col_min // ddx + c_offset]
                    
            if exp_setup == '1drop_co':
                range_exp[:, t] = np.array([r_min, r_max, c_min, c_max])
                
            largest_dim = np.max([r_max-r_min, c_max-c_min])
            
            # make square
            r_max = r_min + largest_dim 
            c_max = c_min + largest_dim
            growth_box_size = int(np.ceil((r_max - r_min) / (3 * int(np.sqrt(core_cnt))))) # size of the area where the dividing cels are located

            r_max += growth_box_size - ((r_max - r_min) % growth_box_size)
            c_max += growth_box_size - ((c_max - c_min) % growth_box_size)

            # TODO: change function such that there will only be 3 * n_wrk + 1 points
            row_splits = np.arange(r_min, r_max+1, growth_box_size)
            col_splits = np.arange(c_min, c_max+1, growth_box_size)

            "This part is kinda dumb, but we're basically mapping the Dboxes to the Gboxes and creating lists that refers to when the Dboxes changes in relation to local Gbox coordinates"

            # pick the correct splits

            diff_splits_row = diff_splits[int(np.floor(row_splits[0]/ddx)):int(np.ceil(row_splits[-1]/ddx)+1)]
            diff_splits_col = diff_splits[int(np.floor(col_splits[0]/ddx)):int(np.ceil(col_splits[-1]/ddx)+1)]

            # map ul corner of each G box to the ul corner of the D box it resides in

            ul_dbox_row = row_splits // ddx
            ul_dbox_col = col_splits // ddx

            # these should be accessed from the Gbox position on the lattice ie. row 0, col2 with mod and //
            first_offset_row = row_splits - ul_dbox_row * ddx # vertical (row) offset of each Gbox ul corner compared to ul Dbox
            first_offset_col = col_splits - ul_dbox_col * ddx # horizontal (col) offset of each Gbox ul corner compared to ul Dbox

            # transform the Dbox splits to Gbox coordinates. if cell_x < Gbox_Dsplit, then cell feel this N conc. Must be lists because the number of Dboxes is different for each Gbox
            gbox_dsplits_row = [np.arange(ddx - first_offset_row[i], growth_box_size, ddx) for i in range(len(first_offset_row))] # NOTE: last one [-1] isn't used, 
            gbox_dsplits_col = [np.arange(ddx - first_offset_col[i], growth_box_size, ddx) for i in range(len(first_offset_col))] # NOTE: last one [-1] isn't used,
            
            # now grab the concentration of N and A
            np.random.shuffle(ran_order)
            
            for div_order in ran_order:
                # div_order = 4
                row_mod = div_order // 3 
                col_mod = div_order % 3

                # row, col indexes for the chosen Gboxes
                rows = np.arange(0, 3 * wrk_r, 3, dtype=np.int64) + row_mod
                cols = np.arange(0, 3 * wrk_r, 3, dtype=np.int64) + col_mod

                used_dsplits_row = [gbox_dsplits_row[row] for row in rows for col in cols]
                used_dsplits_col = [gbox_dsplits_col[col] for row in rows for col in cols]

                " Fill chosen parts of the diff grid with known combination to verify that they're matched"

                Gbox_Dboxes_a = [a_grid[ul_dbox_row[row]  + r_offset: ul_dbox_row[row]+len(gbox_dsplits_row[row])+1 + r_offset,
                            ul_dbox_col[col] + c_offset: ul_dbox_col[col]+len(gbox_dsplits_col[col])+1+ c_offset] for row in rows for col in cols]

                Gbox_Dboxes_n = [n_grid[ul_dbox_row[row]  + r_offset: ul_dbox_row[row]+len(gbox_dsplits_row[row])+1 + r_offset,
                            ul_dbox_col[col] + c_offset: ul_dbox_col[col]+len(gbox_dsplits_col[col])+1+ c_offset] for row in rows for col in cols]

                Gbox_Dboxes_ul_row = np.array([ul_dbox_row[row] for row in rows for col in cols]).flatten()
                Gbox_Dboxes_ul_col = np.array([ul_dbox_col[col] for row in rows for col in cols]).flatten()

                " ensure that the Dboxes for each Gbox are equal to the a grid when indexed using the global coordiantes"
                # for i in range(core_cnt):
                #     assert a_grid[Gbox_Dboxes_ul_row[i] : Gbox_Dboxes_ul_row[i] + Gbox_Dboxes_a[i].shape[0],
                #                 Gbox_Dboxes_ul_col[i] : Gbox_Dboxes_ul_col[i] + Gbox_Dboxes_a[i].shape[1]].all() == Gbox_Dboxes_a[i].all()
                if max_height:
                    div_grid = [c_grid[row_splits[row_mod + (wrk // wrk_r) * 3] - pushing_radius : row_splits[row_mod + (wrk // wrk_r) * 3 +1] + pushing_radius,
                            col_splits[col_mod + (wrk % wrk_r)*3]- pushing_radius : col_splits[col_mod + (wrk % wrk_r)*3 + 1] + pushing_radius ,0:z_max]
                            for wrk in np.arange(0, core_cnt, 1, dtype=np.int64)]
                else:
                    # divides grid into the sub_div grids +- rp. These thus contain all the lattice points needed to divide all cells found within the sub_div grids
                    div_grid = [c_grid[row_splits[row_mod + (wrk // wrk_r) * 3] - pushing_radius : row_splits[row_mod + (wrk // wrk_r) * 3 +1] + pushing_radius,
                                col_splits[col_mod + (wrk % wrk_r)*3]- pushing_radius : col_splits[col_mod + (wrk % wrk_r)*3 + 1] + pushing_radius ,0:z_max+layer_buffer]
                                for wrk in np.arange(0, core_cnt, 1, dtype=np.int64)]

                div_mask_small= [div_mask[row_splits[row_mod + (wrk // wrk_r) * 3] -pushing_radius : row_splits[row_mod + (wrk // wrk_r) * 3 +1] +pushing_radius,
                                col_splits[col_mod + (wrk % wrk_r)*3]-pushing_radius : col_splits[col_mod + (wrk % wrk_r)*3 + 1] +pushing_radius]
                            for wrk in np.arange(0, core_cnt, 1, dtype=np.int64)]

                " initialize parallel cell growth "
                
                task_args = [(div_grid[i], div_mask_small[i], used_dsplits_row[i], used_dsplits_col[i], 
                    Gbox_Dboxes_a[i], Gbox_Dboxes_n[i], dt, pushing_radius, k_A, k_N, 
                    lambda_max_r, lambda_max_s, vertical_loss) for i in range(core_cnt)]
                
                futures = [lm.remote_do_everything.remote(*args) for args in task_args]
                
                results = ray.get(futures)  # Retrieve results from Ray workers
                worker = 0
                for result in results:
                    # update c_grid
                    if max_height:
                        c_grid[row_splits[row_mod + (worker // wrk_r) * 3] - pushing_radius : row_splits[row_mod + (worker // wrk_r) * 3 + 1] + pushing_radius,
                                col_splits[col_mod + (worker % wrk_r) * 3] - pushing_radius : col_splits[col_mod + (worker % wrk_r) * 3 + 1] + pushing_radius, 0:z_max] = result[0]
                    else:
                        c_grid[row_splits[row_mod + (worker // wrk_r) * 3] - pushing_radius : row_splits[row_mod + (worker // wrk_r) * 3 + 1] + pushing_radius,
                            col_splits[col_mod + (worker % wrk_r) * 3] - pushing_radius : col_splits[col_mod + (worker % wrk_r) * 3 + 1] + pushing_radius, 0:z_max + layer_buffer] = result[0]
                    

                    # update div_mask
                    div_mask[row_splits[row_mod + (worker // wrk_r) * 3] - pushing_radius : row_splits[row_mod + (worker // wrk_r) * 3 + 1] + pushing_radius,
                            col_splits[col_mod + (worker % wrk_r) * 3] - pushing_radius : col_splits[col_mod + (worker % wrk_r) * 3 + 1] + pushing_radius] = result[1]
                    
                    # update a_grid
                    a_grid[Gbox_Dboxes_ul_row[worker] + r_offset: Gbox_Dboxes_ul_row[worker] + Gbox_Dboxes_a[worker].shape[0]+ r_offset,
                        Gbox_Dboxes_ul_col[worker] + c_offset: Gbox_Dboxes_ul_col[worker] + Gbox_Dboxes_a[worker].shape[1]+ c_offset] -= result[4] * gamma * Gbox_Dboxes_a[worker] * dt

                    sub_a = a_grid[Gbox_Dboxes_ul_row[worker] + r_offset: Gbox_Dboxes_ul_row[worker] + Gbox_Dboxes_a[worker].shape[0]+ r_offset,
                        Gbox_Dboxes_ul_col[worker] + c_offset: Gbox_Dboxes_ul_col[worker] + Gbox_Dboxes_a[worker].shape[1]+ c_offset]
                    
                    sub_a[sub_a < 0] = 0
                    
                    # update n_grid
                    n_grid[Gbox_Dboxes_ul_row[worker]  + r_offset: Gbox_Dboxes_ul_row[worker] + Gbox_Dboxes_a[worker].shape[0] + r_offset,
                        Gbox_Dboxes_ul_col[worker] + c_offset: Gbox_Dboxes_ul_col[worker] + Gbox_Dboxes_a[worker].shape[1]+ c_offset] -= result[5] * mu  # not scaled with dt

                    sub_n = n_grid[Gbox_Dboxes_ul_row[worker]  + r_offset: Gbox_Dboxes_ul_row[worker] + Gbox_Dboxes_a[worker].shape[0] + r_offset,
                        Gbox_Dboxes_ul_col[worker] + c_offset: Gbox_Dboxes_ul_col[worker] + Gbox_Dboxes_a[worker].shape[1]+ c_offset]
                    
                    sub_n[sub_n < 0] = 0
                    
                    # update cell counts
                    r_cnt += result[2]
                    s_cnt += result[3]
                    
                    worker += 1
            
            
            
             
            if exp_setup == '2drop':
                if (collision == False) and (sen_col_min <= res_col_max): # collision has happened, and the cells should be divided at the same time
                    strains = ['rs']
                    collision_time = t
                    print(f'collision happened at timestep {t}')
                    collision = True
            " Saving snapshots during simulation "
            #     if (collision) and ((t - collision_time) % snap_rate == 0) and (snap_cnt < 11):
            #         lm.save_snap(c_grid, a_grid, n_grid, snappath, snap_cnt)
            #         snap_cnt += 1
            #     if not collision and (t % snap_rate == 0):
            #         a_loc = os.path.join(snappath, f"only_a{only_a_snap_cnt}.npy")
            #         np.save(a_loc, a_grid)
            #         only_a_snap_cnt += 1
                    
            # elif exp_setup == '1drop_co':
            #     if (t % (60 / dt) == 0) and (snap_cnt < 10): # save snapshot every hour
            #         lm.save_snap(c_grid, a_grid, n_grid, snappath, snap_cnt)
            #         snap_cnt += 1
        
        " DIFFUSION TIMESTEP "
        
        # diff_grids = [a_grid, n_grid]
        
        # diffusion_task_args = [(diff_grids[i], fin_diff_mat, M) for i in range(2)]
        
        # diffusion_futures = [lm.sol_diff_no_flux_remote.remote(*args) for args in diffusion_task_args]
            
        # results = ray.get(diffusion_futures)
        
        # maybe consider shared memory?
        # a_grid = np.ndarray.copy(results[0])
        # n_grid = np.ndarray.copy(results[1])
        
        if add_a_at_t <= t * dt:
            a_grid = lm.sol_diff_no_flux(a_grid, fin_diff_mat, M) # solves diffusion equation for A with no flux boundary conditions
        
        # n_grid = lm.sol_diff_no_flux(n_grid, fin_diff_mat, M)
        
        " SAVING SNAPSHORTS DURING SIM "
        r_cnt_at_t[t] += r_cnt
        s_cnt_at_t[t] += s_cnt
        

        

                
    " --------------------------------------------------------------------------------- SIMULATION TERMINATES HERE -------------------------------------------------------------------------------------- "

    " SAVE STUFF "

    # path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # savestring = path + rf'\data'
    save_path = os.path.dirname(snappath) # one step up

    np.save(os.path.join(save_path, run_name + rf'_run_info.npy'), run_info)
    np.save(os.path.join(save_path, run_name + rf'_cell_grid.npy'), lm.save_snap(c_grid, a_grid, n_grid, snappath, snap_cnt, final_state=True)) # returns c_grid as sparse
    # np.save(os.path.join(save_path, run_name + rf'_nutrient_grid.npy'), n_grid)
    np.save(os.path.join(save_path, run_name + rf'_antibiotic_grid.npy'), a_grid)
    np.save(os.path.join(save_path, run_name + rf'_r_at_t.npy'), r_cnt_at_t)
    np.save(os.path.join(save_path, run_name + rf'_s_at_t.npy'), s_cnt_at_t)
    # np.save(os.path.join(save_path, run_name + rf'_runtime.npy'), time_per_step)
    np.save(os.path.join(save_path, run_name + rf'_div_mask.npy'), lm.save_snap(div_mask, a_grid, n_grid, snappath, snap_cnt, final_state=True))
    
    if n_layers > 1:
        np.save(os.path.join(save_path, run_name + rf'_h_at_t.npy'), height_at_t)
    
    if exp_setup == "2drop":
        np.save(os.path.join(save_path, run_name + rf'_a_at_sen_edge.npy'), a_at_sen_edge)
        run_info = np.append(run_info, np.array([['collision_timestep', collision_time]]), axis=0)
        
    if exp_setup == '1drop_co':
        np.save(os.path.join(save_path, run_name + rf'_range_exp.npy'), range_exp)

    ray.shutdown()
    print('TERMINATED GRACEFULLY')





