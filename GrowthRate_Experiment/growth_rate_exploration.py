import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from functools import partial
import pandas as pd

def sen_growth_func(a, ka, ls):
    return ls / (1 + a / ka) 

def sen_growth2_func(a, ka, ls):
    return ls / (1 + (a / ka)**3) 

def exp_growth(t, lamb, n0):
    return n0 * np.exp(lamb * t)

def lin_fit(t, lamb, n0):
    return n0 + lamb * t

" DATA "

df_194 = pd.read_csv('TB194_growth_rate_summary.csv')
df_204 = pd.read_csv('TB204_growth_rate_summary.csv')

df_194['mean_rate_scaled'] = df_194['mean_rate'] * 60
df_194['combined_se_scaled'] = df_194['combined_se'] * 60
df_204['mean_rate_scaled'] = df_204['mean_rate'] * 60
df_204['combined_se_scaled'] = df_204['combined_se'] * 60

a_line = df_194['Cm']


" FIT "
" PLOT RESULTING FIT "

# growth rates as func. of A
plt.figure(1)

plt.errorbar(df_194['Cm'], df_194['mean_rate_scaled'], yerr=df_194['combined_se_scaled'],
             fmt='o-', capsize=4, label='TB194 (scaled)')
plt.errorbar(df_204['Cm'], df_204['mean_rate_scaled'], yerr=df_204['combined_se_scaled'],
             fmt='s-', capsize=4, label='TB204 (scaled)')


sen_growth_fixed_ls = partial(sen_growth_func, ls=df_204['mean_rate_scaled'].iloc[0])
popt, cov = curve_fit(sen_growth_fixed_ls, a_line, df_204['mean_rate_scaled'], sigma=df_204['combined_se_scaled'])
sen_growth2_fixed_ls = partial(sen_growth2_func, ls=df_204['mean_rate_scaled'].iloc[0])
popt2, cov = curve_fit(sen_growth2_fixed_ls, a_line, df_204['mean_rate_scaled'], sigma=df_204['combined_se_scaled'])

print(f'{popt = }')
print(f'{popt2 = }')

plt.plot(a_cont := np.linspace(0, a_line[-1]+0.1, 100), sen_growth_func(a_cont, popt, df_194['mean_rate_scaled'].iloc[0]), '--', label=f'Best fit, $k_a=${popt[0]:.2f}') 
plt.plot(a_cont := np.linspace(0, a_line[-1]+0.1, 100), sen_growth2_func(a_cont, popt, df_194['mean_rate_scaled'].iloc[0]), '--', label=f'Best fit, $k_a=${popt2[0]:.2f}') 

plt.legend()

fig2 = plt.figure(figsize=(8, 4))
bax2 = brokenaxes(xlims=((0, 6), (23.8, 24.05)), hspace=.05) 


