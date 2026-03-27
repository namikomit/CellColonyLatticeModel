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


# Both on one plot
df_194 = pd.read_csv('TB194_growth_rate_summary.csv')
df_204 = pd.read_csv('TB204_growth_rate_summary.csv')

df_194['mean_rate_scaled'] = df_194['mean_rate'] * 60
df_194['combined_se_scaled'] = df_194['combined_se'] * 60
df_204['mean_rate_scaled'] = df_204['mean_rate'] * 60
df_204['combined_se_scaled'] = df_204['combined_se'] * 60

#df_204.loc[df_204['Cm'] >= 5, 'mean_rate_scaled'] = 0.0
#df_204.loc[df_204['Cm'] >= 5, 'combined_se_scaled'] = 0.0  # (optional: also set error bar to zero)

for cm, rate in zip(df_204['Cm'], df_204['mean_rate_scaled']):
    print(f"Cm={cm}: growth rate  = {rate:.4f}")


plt.errorbar(df_194['Cm'], df_194['mean_rate_scaled'], yerr=df_194['combined_se_scaled'],
             fmt='o-', capsize=4, label='TB194 ')
plt.errorbar(df_204['Cm'], df_204['mean_rate_scaled'], yerr=df_204['combined_se_scaled'],
             fmt='s-', capsize=4, label='TB204 ')

#plt.errorbar(df_194['Cm'], df_194['mean_rate'], yerr=df_194['combined_se'], fmt='o-', capsize=4, label='TB194')
#plt.errorbar(df_204['Cm'], df_204['mean_rate'], yerr=df_204['combined_se'], fmt='s-', capsize=4, label='TB204')
x=[0,1,3]
y=[1.4, 0.8, 0.38]
#plt.scatter(x, y, color='red', marker='x', s=80, label='previous data')

def sen_growth_func(a, ka, ls):
    return ls / (1 + a / ka) 

def sen_growth2_func(a, ka, ls):
    return ls / (1 + (a / ka)**3) 

# Fit and plot for TB204
a_cont = np.linspace(0, max(df_194['Cm'].max(), df_204['Cm'].max()) + 0.5, 100)
popt_204, _ = curve_fit(
    lambda a, ka: sen_growth2_func(a, ka, (df_204['mean_rate_scaled'].iloc[0]+df_194['mean_rate_scaled'].iloc[0])/2),
    df_204['Cm'], df_204['mean_rate_scaled'], sigma=df_204['combined_se_scaled']
)
# Calculate the average initial growth rate
g0 = (df_204['mean_rate_scaled'].iloc[0] + df_194['mean_rate_scaled'].iloc[0]) / 2
def sen_growth2_to_zero(cm, ka, ls, cm_cut=5, transition=1.0):
    """
    Interpolates sen_growth2_func to zero at cm_cut with a linear transition of width 'transition'.
    """
    y = np.zeros_like(cm)
    # Region 1: Use sen_growth2_func
    mask1 = cm <= (cm_cut - transition)
    y[mask1] = sen_growth2_func(cm[mask1], ka, ls)
    # Region 2: Linear transition
    mask2 = (cm > (cm_cut - transition)) & (cm < cm_cut)
    y1 = sen_growth2_func(cm_cut - transition, ka, ls)
    y2 = 0
    y[mask2] = y1 + (y2 - y1) * (cm[mask2] - (cm_cut - transition)) / transition
    # Region 3: Zero
    # y[cm >= cm_cut] = 0 (already set by zeros_like)
    return y

# Usage:
cm_vals = np.linspace(0, 10, 200)
y_interp = sen_growth2_to_zero(cm_vals, popt_204[0], g0, cm_cut=5, transition=1.0)


plt.plot(cm_vals, y_interp, 'c-', lw=2, label=f'(g(0)/[1+(cm/ka)^3], \nka={popt_204[0]:.2f}, g0={g0:.2f}) \ninterpolated')
plt.axhline(y=g0, color='k', linestyle='--', label=f'g0 = {g0:.2f}')

#plt.plot(a_cont, sen_growth2_func(a_cont, popt_204[0], (df_204['mean_rate_scaled'].iloc[0]+df_194['mean_rate_scaled'].iloc[0])/2),
        # 'g--', label=f'TB204 fit ')
#plt.plot(a_cont, sen_growth_func(a_cont, 0.5, (df_204['mean_rate_scaled'].iloc[0]+df_194['mean_rate_scaled'].iloc[0])/2),
#         'k:', label='sen_growth_func (g(0)/(1+cm/ka),ka=0.5)')
plt.xlabel('Cm')
plt.ylabel('Mean exponential Growth Rate (1/h)')
#plt.title('Mean Growth Rate vs Cm (TB194 & TB204)')
plt.legend(ncol=2, loc='center right')
#plt.grid(True)
plt.savefig('plots_remove/MeanGrowthRate_vs_Cm_TB194_TB204_fit.pdf')  # <-- Add this line
plt.show()


