import numpy as np
import matplotlib.pyplot as plt
import re
import os
from matplotlib.colors import ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar # might need to install this package

# ---- SET THESE BEFORE RUNNING ------------------------------------------------
# Path to the folder containing simulation run directories
main_path = rf"E:\cell_lattice\data\no_pushing"   # <-- update to your local path

# Name of the simulation run folder to plot
run_name = "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A4_0"  # <-- update to your run name
# ------------------------------------------------------------------------------
scale_bar = True


c_data = np.load(rf"{main_path}\{run_name}\{run_name}_cell_grid.npy", mmap_mode='r')
run_info = np.load(rf"{main_path}\{run_name}\{run_name}_run_info.npy", allow_pickle=True)
r_at_t = np.load(rf"{main_path}\{run_name}\{run_name}_r_at_t.npy", allow_pickle=True)
s_at_t = np.load(rf"{main_path}\{run_name}\{run_name}_s_at_t.npy", allow_pickle=True)


print(run_info)
# print(r_at_t)
ymin = np.min(c_data[0])
ymax = np.max(c_data[0])

xmin = np.min(c_data[1])
xmax = np.max(c_data[1])

c_grid = np.zeros((int(ymax-ymin+1), int(xmax-xmin+1)))
c_grid[c_data[0]-ymin, c_data[1]-xmin] = c_data[3]
c_grid[(c_grid < 11) & (c_grid > 0)] = 1
c_grid[c_grid > 10] = 2

fig1, ax1 = plt.subplots()
ax1.matshow(c_grid, cmap=ListedColormap(['black', 'blue', 'yellow']))

is_1drop = not re.match("2drop", run_name)

if not is_1drop:
	title = rf"a_0: {run_info[24, 1]}, $\gamma$: {(float(run_info[16, 1])*30**2):.3f}"
	file_title = rf"2d_a{run_info[24, 1]}_g{(float(run_info[16, 1])*30**2):.3f}"
	ax1.set_title(title)
else:
	title = rf"ratio: {run_info[14,1]}, a_0: {run_info[24, 1]}, $\gamma$: {(float(run_info[16, 1])*30**2):.3f}"
	file_title = rf"1d_r{run_info[14,1]}_a{run_info[24, 1]}_g{(float(run_info[16, 1])*30**2):.3f}"
	ax1.set_title(title)

if scale_bar and not is_1drop:
	scalebar = ScaleBar(1,scale_formatter=lambda value, unit: fr"{value} $\mu$m", location="lower right", box_alpha=0, color="black", fixed_value=500) # 1 pixel = 0.2 meter
	plt.gca().add_artist(scalebar)

if scale_bar and is_1drop:
	scalebar = ScaleBar(1,scale_formatter=lambda value, unit: fr"{value} $\mu$m", location="lower right", box_alpha=0, color="white", fixed_value=500) # 1 pixel = 0.2 meter
	plt.gca().add_artist(scalebar)

plt.tight_layout()
# plt.savefig(rf"E:\cell_lattice\tiffs\{file_title}.tiff",
# 			dpi=800)

plt.show(block=False)

input("Press Enter to continue...")