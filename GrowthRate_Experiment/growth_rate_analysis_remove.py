import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from functools import partial
import pandas as pd
from collections import defaultdict
import re
import os
from scipy.stats import linregress




" DATA "

df = pd.read_csv('250522_CmResistance_GrowthAssay.csv', sep=';')
print(df.columns)
time = df['Time'].values
print("time:", time.shape)


# Group columns by concentration
tb194_cols = [col for col in df.columns if col.startswith('TB194_Cm')]
cm_col_dict = defaultdict(list)
for col in tb194_cols:
    conc = int(re.search(r'Cm(\d+)', col).group(1))
    cm_col_dict[conc].append(col)

# Sort concentrations
sorted_concs = sorted(cm_col_dict.keys())

# Pool all replicates for each concentration, stacking them side by side
tb194_data = np.hstack([df[cm_col_dict[conc]].values for conc in sorted_concs])

# Optionally, keep track of which columns belong to which concentration
tb194_col_indices = []

for conc in sorted_concs:
    n_reps = len(cm_col_dict[conc])
    tb194_col_indices.append((conc, list(range(n_reps))))  # always [0, 1, ..., n_reps-1]



tb204_cols = [col for col in df.columns if col.startswith('TB204_Cm')]
cm_col_dict = defaultdict(list)
for col in tb204_cols:
    conc = int(re.search(r'Cm(\d+)', col).group(1))
    cm_col_dict[conc].append(col)

# Sort concentrations
sorted_concs = sorted(cm_col_dict.keys())

# Pool all replicates for each concentration, stacking them side by side
tb204_data = np.hstack([df[cm_col_dict[conc]].values for conc in sorted_concs])

# Optionally, keep track of which columns belong to which concentration
tb204_col_indices = []
for conc in sorted_concs:
    n_reps = len(cm_col_dict[conc])
    tb194_col_indices.append((conc, list(range(n_reps))))  # always [0, 1, ..., n_reps-1]


# Find the index of Cm=0 in sorted_concs
# cm10_idx = sorted_concs.index(10)
# n_reps = len(cm_col_dict[10])

# # Get the columns for Cm=0 (they are the first n_reps columns in tb194_data)
# col_start = sum(len(cm_col_dict[c]) for c in sorted_concs[:cm10_idx])
# tb204_cm10_data = tb204_data[:, col_start:col_start + n_reps]

# # Plot each replicate
# for i in range(n_reps):
#     plt.semilogy(time, tb204_cm10_data[:, i], label=f'Replicate {i+1}')


# plt.xlabel('Time')
# plt.ylabel('OD or measurement')
# plt.title('TB204 Cm=10 Growth Curves')
# plt.legend()
# plt.show()

# colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_concs)))  # or use any colormap you like

# col_start = 0
# for idx, conc in enumerate(sorted_concs):
#     n_reps = len(cm_col_dict[conc])
#     # Get the data for this concentration (all replicates)
#     data = tb194_data[:, col_start:col_start + n_reps]
#     for rep in range(n_reps):
#         plt.semilogy(time, data[:, rep], color=colors[idx], label=f'Cm={conc}' if rep == 0 else None)
#     col_start += n_reps

# plt.xlabel('Time')
# plt.ylabel('OD or measurement (log scale)')
# plt.title('TB194 Growth Curves by Cm Concentration')
# plt.legend()
# plt.xlim(0, 300)
# plt.show()


# colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_concs)))  # or use any colormap you like

# col_start = 0
# for idx, conc in enumerate(sorted_concs):
#     n_reps = len(cm_col_dict[conc])
#     # Get the data for this concentration (all replicates)
#     data = tb204_data[:, col_start:col_start + n_reps]
#     for rep in range(n_reps):
#         plt.semilogy(time, data[:, rep], color=colors[idx], label=f'Cm={conc}' if rep == 0 else None)
#     col_start += n_reps

# plt.xlabel('Time')
# plt.ylabel('OD or measurement (log scale)')
# plt.title('TB204 Growth Curves by Cm Concentration')
# plt.legend()
# plt.show()

# --- For TB194 ---
col_start = 0
for conc in sorted_concs:
    n_reps = len(cm_col_dict[conc])
    data = tb194_data[:, col_start:col_start + n_reps]
    plt.figure()
    for rep in range(n_reps):
        plt.semilogy(time, data[:, rep], label=f'Replicate {rep+1}')
    plt.xlabel('Time')
    plt.ylabel('OD or measurement (log scale)')
    plt.title(f'TB194 Cm={conc} Growth Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots_remove/TB194_Cm{conc}.pdf')
    plt.close()
    col_start += n_reps

# --- For TB204 ---
col_start = 0
for conc in sorted_concs:
    n_reps = len(cm_col_dict[conc])
    data = tb204_data[:, col_start:col_start + n_reps]
    plt.figure()
    for rep in range(n_reps):
        plt.semilogy(time, data[:, rep], label=f'Replicate {rep+1}')
    plt.xlabel('Time')
    plt.ylabel('OD or measurement (log scale)')
    plt.title(f'TB204 Cm={conc} Growth Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots_remove/TB204_Cm{conc}.pdf')
    plt.close()
    col_start += n_reps
    
# Indices for Cm=0 and Cm=1 in sorted_concs
cm_values = [0, 1]
colors = ['tab:blue', 'tab:orange']

col_start = 0
for idx, conc in enumerate(sorted_concs):
    n_reps = len(cm_col_dict[conc])
    if conc in cm_values:
        data = tb204_data[:, col_start:col_start + n_reps]
        for rep in range(n_reps):
            plt.semilogy(time, data[:, rep], color=colors[cm_values.index(conc)],
                         label=f'Cm={conc} Rep={rep+1}' if rep == 0 else None)
    col_start += n_reps

plt.xlabel('Time')
plt.ylabel('OD or measurement (log scale)')
plt.title('TB204 Growth Curves for Cm=0 and Cm=1')
plt.xlim(0, 300)
plt.legend()
plt.show()
    
# Example: t_min and t_max for each Cm (must match sorted_concs order)
t_min = [90, 90, 90, 90, 90, 90, 90, 90, 90, 90]  # example values
t_max = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150]

growth_rates = []
fit_errors = []

col_start = 0
for idx, conc in enumerate(sorted_concs):
    n_reps = len(cm_col_dict[conc])
    data = tb194_data[:, col_start:col_start + n_reps]
    for rep in range(n_reps):
        y = data[:, rep]
        #mask = (time >= t_min[idx]) & (time <= t_max[idx])
        #time_fit = time[mask]
        #y_fit = y[mask]
                # Find start and end indices for fitting
        above_005 = np.where(y > 0.05)[0]
        above_05 = np.where(y > 0.5)[0]
        if len(above_005) == 0 or len(above_05) == 0:
            continue  # skip if thresholds not reached
        start_idx = above_005[0]
        end_idx = above_05[0]
        if conc ==7:
            above_008 = np.where(y > 0.08)[0]
            above_05 = np.where(y > 0.5)[0]
            if len(above_008) == 0 or len(above_05) == 0:
                continue  # skip if thresholds not reached
            start_idx = above_008[0]
            end_idx = above_05[0]
        if end_idx <= start_idx:
            continue  # skip if range is invalid
        time_fit = time[start_idx:end_idx+1]
        y_fit = y[start_idx:end_idx+1]
        log_y_fit = np.log(y_fit)
        res = linregress(time_fit, log_y_fit)
        slope = res.slope
        slope_se = res.stderr
        intercept = res.intercept  # <-- add this line
        growth_rates.append({'Cm': conc, 'rep': rep+1, 'rate': slope})
        fit_errors.append({'Cm': conc, 'rep': rep+1, 'fit_se': slope_se})
        plt.semilogy(time, y, label=f'Rep {rep+1} data')
        plt.semilogy(time_fit, np.exp(slope*time_fit + intercept), '--', label=f'Rep {rep+1} (rate={slope:.5f})')
    plt.xlabel('Time')
    plt.ylabel('OD (log scale)')
    plt.title(f'TB194 Cm={conc} Growth Curves and Fits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots_remove/TB194_Cm{conc}_fits.pdf')
    plt.close()
    col_start += n_reps
# Save all replicate growth rates for TB194
df_194 = pd.DataFrame(growth_rates)  # growth_rates from TB194 loop
df_194.to_csv('TB194_growth_rates.csv', index=False)

# Print all growth rates
for gr in growth_rates:
    print(f"Cm={gr['Cm']} Rep={gr['rep']} Growth rate: {gr['rate']:.5f} 1/time")
 
# Group growth rates by Cm
cm_to_rates = {}
cm_to_fit_se = {}
for entry, fit in zip(growth_rates, fit_errors):
    cm = entry['Cm']
    rate = entry['rate']
    fit_se = fit['fit_se']
    cm_to_rates.setdefault(cm, []).append(rate)
    cm_to_fit_se.setdefault(cm, []).append(fit_se)

summary_194 = []
for cm in cm_to_rates:
    rates = np.array(cm_to_rates[cm])
    fit_ses = np.array(cm_to_fit_se[cm])
    mean = np.mean(rates)
    std = np.std(rates, ddof=1)
    # Combine errors: sqrt( (std/sqrt(n))^2 + (mean_fit_se/sqrt(n))^2 )
    n = len(rates)
    mean_fit_se = np.mean(fit_ses)
    combined_se = np.sqrt((std / np.sqrt(n))**2 + (mean_fit_se / np.sqrt(n))**2)
    summary_194.append({'Cm': cm, 'mean_rate': mean, 'std_rate': std, 'n': n, 'fit_se': mean_fit_se, 'combined_se': combined_se})
df_summary_194 = pd.DataFrame(summary_194)
df_summary_194.to_csv('TB194_growth_rate_summary.csv', index=False)

# Example: t_min and t_max for each Cm (must match sorted_concs order)

# Save all replicate growth rates for TB204
df_204 = pd.DataFrame(growth_rates)  # growth_rates from TB204 loop
df_204.to_csv('TB204_growth_rates.csv', index=False) 

t_min = [100, 130, 130, 130, 270, 1200, 1200, 1000, 1200, 1200]  # example values
t_max = [150, 170, 230, 230, 450, 1400, 1400, 1200, 1400, 1400]

growth_rates = []

col_start = 0
for idx, conc in enumerate(sorted_concs):
    n_reps = len(cm_col_dict[conc])
    data = tb204_data[:, col_start:col_start + n_reps]
    for rep in range(n_reps):
        if rep not in [0, 1, 3]:
            continue  # Skip replicates not 0, 1, or 3
        y = data[:, rep]
        if conc >= 5: # For Cm >= 5, use t_min and t_max
            mask = (time >= t_min[idx]) & (time <= t_max[idx])
            time_fit = time[mask]
            y_fit = y[mask]
            if conc==5:
                if rep ==0:
                    above_009 = np.where(y > 0.09)[0]
                    above_02 = np.where(y > 0.2)[0]
                    if len(above_009) == 0 or len(above_02) == 0:
                        continue  # skip if thresholds not reached
                    start_idx = above_009[0]
                    end_idx = above_02[0]
                    time_fit = time[start_idx:end_idx+1]
                    y_fit = y[start_idx:end_idx+1]
                elif rep ==2:
                    above_003 = np.where(y > 0.03)[0]
                    above_03 = np.where(y > 0.3)[0]
                    if len(above_003) == 0 or len(above_03) == 0:
                        continue  # skip if thresholds not reached
                    start_idx = above_003[0]
                    end_idx = above_03[0]
                    time_fit = time[start_idx:end_idx+1]
                    y_fit = y[start_idx:end_idx+1]
            elif conc ==6:  
                if rep ==2:
                    above_003 = np.where(y > 0.03)[0]
                    above_03 = np.where(y > 0.3)[0]
                    if len(above_003) == 0 or len(above_03) == 0:
                        continue  # skip if thresholds not reached
                    start_idx = above_003[0]
                    end_idx = above_03[0]
                    time_fit = time[start_idx:end_idx+1]
                    y_fit = y[start_idx:end_idx+1]
            elif conc ==7:  
                if rep ==2:
                    above_005 = np.where(y > 0.05)[0]
                    above_03 = np.where(y > 0.3)[0]
                    if len(above_005) == 0 or len(above_03) == 0:
                        continue  # skip if thresholds not reached
                    start_idx = above_005[0]
                    end_idx = above_03[0]
                    time_fit = time[start_idx:end_idx+1]
                    y_fit = y[start_idx:end_idx+1]
            elif conc ==8:     
                if rep ==2:
                    above_003 = np.where(y > 0.03)[0]
                    above_01 = np.where(y > 0.1)[0]
                    if len(above_003) == 0 or len(above_01) == 0:
                        continue  # skip if thresholds not reached
                    start_idx = above_003[0]
                    end_idx = above_01[0]   
                    time_fit = time[start_idx:end_idx+1]
                    y_fit = y[start_idx:end_idx+1]    
        else:
            if conc ==0:# Find start and end indices for fitting
                above_005 = np.where(y > 0.05)[0]
                above_04 = np.where(y > 0.4)[0]
                if len(above_005) == 0 or len(above_04) == 0:
                    continue  # skip if thresholds not reached
                start_idx = above_005[0]
                end_idx = above_04[0]
            elif conc in [1,2] :
                above_003 = np.where(y > 0.03)[0]
                above_03 = np.where(y > 0.3)[0]
                if len(above_003) == 0 or len(above_03) == 0:
                    continue  # skip if thresholds not reached
                start_idx = above_003[0]
                end_idx = above_03[0]
            elif conc in [3,4] :
                above_002 = np.where(y > 0.02)[0]
                above_02 = np.where(y > 0.2)[0]
                if len(above_002) == 0 or len(above_02) == 0:
                    continue  # skip if thresholds not reached
                start_idx = above_002[0]
                end_idx = above_02[0]                

            if end_idx <= start_idx:
                continue  # skip if range is invalid
            time_fit = time[start_idx:end_idx+1]
            y_fit = y[start_idx:end_idx+1]
        log_y_fit = np.log(y_fit)
        res = linregress(time_fit, log_y_fit)
        slope = res.slope
        slope_se = res.stderr
        intercept = res.intercept  # <-- add this line
        growth_rates.append({'Cm': conc, 'rep': rep+1, 'rate': slope})
        fit_errors.append({'Cm': conc, 'rep': rep+1, 'fit_se': slope_se})
        plt.semilogy(time, y, label=f'Rep {rep+1} data')
        plt.semilogy(time_fit, np.exp(slope*time_fit + intercept), '--', label=f'Rep {rep+1} (rate={slope:.5f})')
    plt.xlabel('Time')
    plt.ylabel('OD (log scale)')
    plt.title(f'TB204 Cm={conc} Growth Curves and Fits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots_remove/TB204_Cm{conc}_fits.pdf')
    plt.close()
    col_start += n_reps


# Print all growth rates
for gr in growth_rates:
    print(f"Cm={gr['Cm']} Rep={gr['rep']} Growth rate: {gr['rate']:.5f} 1/time")
    
    

# Group growth rates by Cm
cm_to_rates = {}
cm_to_fit_se = {}
for entry, fit in zip(growth_rates, fit_errors):
    cm = entry['Cm']
    rate = entry['rate']
    fit_se = fit['fit_se']
    cm_to_rates.setdefault(cm, []).append(rate)
    cm_to_fit_se.setdefault(cm, []).append(fit_se)

summary_204 = []
for cm in cm_to_rates:
    rates = np.array(cm_to_rates[cm])
    fit_ses = np.array(cm_to_fit_se[cm])
    mean = np.mean(rates)
    std = np.std(rates, ddof=1)
    # Combine errors: sqrt( (std/sqrt(n))^2 + (mean_fit_se/sqrt(n))^2 )
    n = len(rates)
    mean_fit_se = np.mean(fit_ses)
    combined_se = np.sqrt((std / np.sqrt(n))**2 + (mean_fit_se / np.sqrt(n))**2)
    summary_204.append({'Cm': cm, 'mean_rate': mean, 'std_rate': std, 'n': n, 'fit_se': mean_fit_se, 'combined_se': combined_se})
df_summary_204 = pd.DataFrame(summary_204)
df_summary_204.to_csv('TB204_growth_rate_summary.csv', index=False)

# Both on one plot
df_194 = pd.read_csv('TB194_growth_rate_summary.csv')
df_204 = pd.read_csv('TB204_growth_rate_summary.csv')

df_194['mean_rate_scaled'] = df_194['mean_rate'] * 60
df_194['combined_se_scaled'] = df_194['combined_se'] * 60
df_204['mean_rate_scaled'] = df_204['mean_rate'] * 60
df_204['combined_se_scaled'] = df_204['combined_se'] * 60

plt.errorbar(df_194['Cm'], df_194['mean_rate_scaled'], yerr=df_194['combined_se_scaled'],
             fmt='o-', capsize=4, label='TB194 (scaled)')
plt.errorbar(df_204['Cm'], df_204['mean_rate_scaled'], yerr=df_204['combined_se_scaled'],
             fmt='s-', capsize=4, label='TB204 (scaled)')

#plt.errorbar(df_194['Cm'], df_194['mean_rate'], yerr=df_194['combined_se'], fmt='o-', capsize=4, label='TB194')
#plt.errorbar(df_204['Cm'], df_204['mean_rate'], yerr=df_204['combined_se'], fmt='s-', capsize=4, label='TB204')
x=[0,1,3]
y=[1.4, 0.8, 0.38]
plt.scatter(x, y, color='red', marker='x', s=80, label='previous data')

plt.xlabel('Cm')
plt.ylabel('Mean Growth Rate')
plt.title('Mean Growth Rate vs Cm (TB194 & TB204)')
plt.legend()
plt.grid(True)
plt.savefig('plots_remove/MeanGrowthRate_vs_Cm_TB194_TB204.pdf')  # <-- Add this line
plt.show()

#Calculate the 6h ratio, then calculate W assuming the growth fold is 500
# Calculate the mean value at time=360 for each Cm
# Find the index of time=360
time_idx = np.argmin(np.abs(time - 360))

cm_means_204 = []
cm_ses_204 = []
cm_list_204 = []

for cm in sorted_concs:
    n_reps = len(cm_col_dict[cm])
    col_start = sum(len(cm_col_dict[c]) for c in sorted_concs[:sorted_concs.index(cm)])
    data = tb204_data[:, col_start:col_start + n_reps]
    rep_indices = [i for i in range(data.shape[1]) if i != 2]
    values_at_360 = data[time_idx, rep_indices]
    mean_val = np.mean(values_at_360)
    se_val = np.std(values_at_360, ddof=1) / np.sqrt(len(rep_indices))
    cm_means_204.append(mean_val)
    cm_ses_204.append(se_val)
    cm_list_204.append(cm)
    print(f"Cm={cm}: mean={mean_val:.4f}, SE={se_val:.4f}, n={n_reps}")
    


cm_means_194 = []
cm_ses_194 = []
cm_list_194  = []

for cm in sorted_concs:
    n_reps = len(cm_col_dict[cm])
    col_start = sum(len(cm_col_dict[c]) for c in sorted_concs[:sorted_concs.index(cm)])
    data = tb194_data[:, col_start:col_start + n_reps]
    values_at_360 = data[time_idx, :]
    mean_val = np.mean(values_at_360)
    se_val = np.std(values_at_360, ddof=1) / np.sqrt(n_reps)
    cm_means_194.append(mean_val)
    cm_ses_194.append(se_val)
    cm_list_194.append(cm)
    print(f"Cm={cm}: mean={mean_val:.4f}, SE={se_val:.4f}, n={n_reps}")

def ratio_A_over_AplusB(A, se_A, B, se_B):
    R = A / (A + B)
    dA = B / (A + B)**2
    dB = -A / (A + B)**2
    se_R = np.sqrt((dA * se_A)**2 + (dB * se_B)**2)
    return R, se_R

R_list = []
se_R_list = []

for cm in sorted_concs:
    R, se_R = ratio_A_over_AplusB(cm_means_204[cm_list_204.index(cm)],
                                   cm_ses_204[cm_list_204.index(cm)],
                                   cm_means_194[cm_list_194.index(cm)],
                                   cm_ses_194[cm_list_194.index(cm)])
    R_list.append(np.log(R)/500)
    se_R_list.append((se_R/R)/500)
    print(f"Cm={cm}: R={R:.4f}, SE={se_R:.4f}")
    print(f"Cm={cm}: R={R_list[-1]:.4f}, SE={se_R_list[-1]:.4f}")
# Optionally, plot
plt.errorbar(cm_list_194, cm_means_194, yerr=cm_ses_194, fmt='o-', capsize=4, label='TB194 at 360 min')
plt.errorbar(cm_list_204, cm_means_204, yerr=cm_ses_204, fmt='o-', capsize=4,label='TB204 at 360 min')
plt.xlabel('Cm')
plt.ylabel('Mean value at time=360')
plt.title('Mean value at time=360 vs Cm')
plt.show()

plt.errorbar(cm_list_194, R_list, yerr=se_R_list, fmt='o-', capsize=4, label='Ratio at 360 min/500')
plt.xlabel('Cm')
plt.ylabel('s at 6h')
plt.title('s at time=360 vs Cm')
plt.show()