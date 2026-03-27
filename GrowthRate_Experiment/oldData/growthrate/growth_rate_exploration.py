import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from functools import partial

def sen_growth_func(a, ka, ls):
    return ls / (1 + a / ka) 

def sen_growth2_func(a, ka, ls):
    return ls / (1 + (a / ka)**3) 

def exp_growth(t, lamb, n0):
    return n0 * np.exp(lamb * t)

def lin_fit(t, lamb, n0):
    return n0 + lamb * t

" DATA "

time = np.array([0, 45, 90, 130, 175, 220, 265, 310, 1440]) /60  # [h]

RES0 = np.array([0.064, 0.107, 0.345, 0.82, 2.3, 3.3, 4.3, 5.4, 5.4])
RES6 = np.array([0.064, 0.107, 0.342, 0.816, 2.16, 3.32, 4.3, 5.6, 6])

SEN0 = np.array([0.057, 0.082, 0.233, 0.614, 1.4, 2.8, 4.1, 4.5, 5.3])
SEN1 = np.array([0.057, 0.074, 0.125, 0.24, 0.43, 0.62, 0.82, 1.1, 3.1])
SEN3 = np.array([0.057, 0.067, 0.1, 0.13, 0.18, 0.228, 0.3, 0.377, 2.8])
SEN6 = np.array([0.057, 0.065, 0.085, 0.102, 0.117, 0.124, 0.126, 0.13, 0.96])

a_line = np.array([0, 1, 3, 6])

# pick a range to fit
fit_a0_min = 1 # for sen a=0
fit_a_min = 4 # for sen a != 0
fit_range_a0 = slice(fit_a0_min,4)
fit_range_a = slice(fit_a_min,8)

" INITIAL PLOT OF OD VALUES "
# broken x-axis
fig = plt.figure(figsize=(8, 4))
bax = brokenaxes(xlims=((0, 6), (23.8, 24.05)), hspace=.05) 

# plot OD curves
bax.plot(time, SEN0, '-o', label='SEN0')
bax.plot(time, SEN1, '-o', label='SEN1')
bax.plot(time, SEN3, '-o', label='SEN3')
bax.plot(time, SEN6, '-o', label='SEN6')
bax.plot(time, RES0, '-o', label='RES0', alpha=0.1)
bax.plot(time, RES6, '-o', label='RES6', alpha=0.1)

# highlight fit region
bax.axvspan(time[fit_range_a][0], time[fit_range_a][-1], color='lightblue', alpha=0.4, label='fit region')

bax.set_yscale('log')
bax.set_xlabel('t [h]')
bax.set_ylabel('OD600')
bax.legend(loc='upper left')

" FIT "

# arrays to fill growth rate values into
sen_growth = np.zeros(len(a_line))
sen_growth_sig = np.zeros(len(a_line))
res_growth = np.zeros(2)
res_growth_sig = np.zeros(2)

# fitting to exponential growth, using n(t)=n0 exp(lambda * t), with n0 = 0 at start of fit ([fit_range][0])
sen_growth[0], sen_growth_sig[0] = curve_fit(partial(exp_growth, n0=SEN0[fit_range_a0][0]), time[fit_range_a0] - time[fit_a0_min], SEN0[fit_range_a0]) 
sen_growth[1], sen_growth_sig[1] = curve_fit(partial(exp_growth, n0=SEN1[fit_range_a0][0]), time[fit_range_a0] - time[fit_a0_min], SEN1[fit_range_a0])
sen_growth[2], sen_growth_sig[2] = curve_fit(partial(exp_growth, n0=SEN3[fit_range_a][0]), time[fit_range_a] - time[fit_a_min], SEN3[fit_range_a])
sen_growth[3], sen_growth_sig[3] = curve_fit(partial(exp_growth, n0=SEN6[fit_range_a][0]), time[fit_range_a] - time[fit_a_min], SEN6[fit_range_a])

res_growth[0], res_growth_sig[0] = curve_fit(partial(exp_growth, n0=RES0[fit_range_a0][0]), time[fit_range_a0] - time[fit_a0_min], RES0[fit_range_a0])
res_growth[1], res_growth_sig[1] = curve_fit(partial(exp_growth, n0=RES6[fit_range_a0][0]), time[fit_range_a0] - time[fit_a0_min], RES6[fit_range_a0])


# fitting to exponential growth, using n(t)=n0 exp(lambda * t), with n0 = 0 at start of fit ([fit_range][0])
# sen_growth[0], sen_growth_sig[0] = least_squares(partial(lin_fit, n0=np.log(SEN0[fit_range_a0][0])), time[fit_range_a0], np.log(SEN0[fit_range_a0]), loss="soft_l1")
# sen_growth[1], sen_growth_sig[1] = curve_fit(partial(lin_fit, n0=np.log(SEN1[fit_range_a][0])), time[fit_range_a], np.log(SEN1[fit_range_a]))
# sen_growth[2], sen_growth_sig[2] = curve_fit(partial(lin_fit, n0=np.log(SEN3[fit_range_a][0])), time[fit_range_a], np.log(SEN3[fit_range_a]))
# sen_growth[3], sen_growth_sig[3] = curve_fit(partial(lin_fit, n0=np.log(SEN6[fit_range_a][0])), time[fit_range_a], np.log(SEN6[fit_range_a]))

# res_growth[0], res_growth_sig[0] = curve_fit(partial(lin_fit, n0=np.log(RES0[fit_range_a0][0])), time[fit_range_a0], np.log(RES0[fit_range_a0]))
# res_growth[1], res_growth_sig[1] = curve_fit(partial(lin_fit, n0=np.log(RES6[fit_range_a0][0])), time[fit_range_a0], np.log(RES6[fit_range_a0]))



# sen_growth /= np.log(2)
# sen_growth_sig /= np.log(2)

# res_growth /= np.log(2)
# res_growth_sig /= np.log(2)

print(f'{sen_growth / 60 =}')
print(f'{res_growth / 60 =}')

# print(np.sqrt(sen_growth_sig))

" PLOT RESULTING FIT "

# growth rates as func. of A
plt.figure(2)
plt.errorbar([0,6], res_growth, yerr=np.sqrt(res_growth_sig), fmt='o', label='RES', capsize=5)
plt.errorbar(a_line, sen_growth, yerr=np.sqrt(sen_growth_sig), fmt ='o', label='SEN', capsize=5)

sen_growth_fixed_ls = partial(sen_growth_func, ls=sen_growth[0])
popt, cov = curve_fit(sen_growth_fixed_ls, a_line, sen_growth, sigma=np.sqrt(sen_growth_sig))
sen_growth2_fixed_ls = partial(sen_growth2_func, ls=sen_growth[0])
popt2, cov = curve_fit(sen_growth2_fixed_ls, a_line, sen_growth, sigma=np.sqrt(sen_growth_sig))


print(f'{popt = }')
print(f'{popt2 = }')

plt.plot(a_cont := np.linspace(0, a_line[-1]+0.1, 100), sen_growth_func(a_cont, popt, sen_growth[0]), '--', label=f'Best fit, $k_a=${popt[0]:.2f}') 
plt.plot(a_cont := np.linspace(0, a_line[-1]+0.1, 100), sen_growth2_func(a_cont, popt, sen_growth[0]), '--', label=f'Best fit, $k_a=${popt2[0]:.2f}') 

plt.legend()

fig2 = plt.figure(figsize=(8, 4))
bax2 = brokenaxes(xlims=((0, 6), (23.8, 24.05)), hspace=.05) 


# plot data and resulting fits
# plot OD curves
incl_label = False
line_SEN0 = bax2.plot(time, SEN0, 'o', label='SEN0' if incl_label else None)[0][0]
line_SEN1 = bax2.plot(time, SEN1, 'o', label='SEN1' if incl_label else None)[0][0]
line_SEN3 = bax2.plot(time, SEN3, 'o', label='SEN3' if incl_label else None)[0][0]
line_SEN6 = bax2.plot(time, SEN6, 'o', label='SEN6' if incl_label else None)[0][0]
line_RES0 = bax2.plot(time, RES0, 'o', label='RES0' if incl_label else None, alpha=0.1)[0][0]
line_RES6 = bax2.plot(time, RES6, 'o', label='RES6' if incl_label else None, alpha=0.1)[0][0]

# plot fits
bax2.plot(time[fit_range_a0], exp_growth(time[fit_range_a0] - time[fit_range_a0][0], lamb=sen_growth[0], n0=SEN0[fit_range_a0][0]), label='SEN0 fit', color=line_SEN0.get_color())
bax2.plot(time[fit_range_a0], exp_growth(time[fit_range_a0] - time[fit_range_a0][0], lamb=sen_growth[1], n0=SEN1[fit_range_a0][0]), label='SEN1 fit', color=line_SEN1.get_color())
bax2.plot(time[fit_range_a], exp_growth(time[fit_range_a] - time[fit_range_a][0], lamb=sen_growth[2], n0=SEN3[fit_range_a][0]), label='SEN3 fit', color=line_SEN3.get_color())
bax2.plot(time[fit_range_a], exp_growth(time[fit_range_a] - time[fit_range_a][0], lamb=sen_growth[3], n0=SEN6[fit_range_a][0]), label='SEN6 fit', color=line_SEN6.get_color())

bax2.plot(time[fit_range_a0], exp_growth(time[fit_range_a0] - time[fit_range_a0][0], lamb=res_growth[0], n0=RES0[fit_range_a0][0]), label='RES0 fit', color=line_RES0.get_color(), alpha=0.1)
bax2.plot(time[fit_range_a0], exp_growth(time[fit_range_a0] - time[fit_range_a0][0], lamb=res_growth[1], n0=RES6[fit_range_a0][0]), label='RES6 fit', color=line_RES6.get_color(), alpha=0.1)

# highlight fit region
bax2.axvspan(time[fit_range_a][0], time[fit_range_a][-1], color='lightblue', alpha=0.4, label='fit region')

bax2.set_yscale('log')
bax2.set_xlabel('t [h]')
bax2.set_ylabel('OD600')
bax2.legend(loc='upper left')


if False:
    first_point = 1
    last_point = 4

    # range_i = slice(1,-5)
    # range_f = slice(2,-4)
    range_i = slice(first_point, last_point)
    range_f = slice(first_point+1, last_point+1)

    # ln(OD(t2) / OD(t1)) / (t2 - t1) for each data point in the specified range
    sen_mean_rates = np.array([
        np.mean( (np.log(SEN0[range_f] / SEN0[range_i]) / (time[range_f]- time[range_i]) )), 
        np.mean( (np.log(SEN1[range_f] / SEN1[range_i]) / (time[range_f]- time[range_i]) )), 
        np.mean( (np.log(SEN3[range_f] / SEN3[range_i]) / (time[range_f]- time[range_i]) )), 
        np.mean( (np.log(SEN6[range_f] / SEN6[range_i]) / (time[range_f]- time[range_i]) )), 
    ])

    res_mean_rates = np.array([
        np.mean( (np.log(RES0[range_f] / RES0[range_i]) / (time[range_f]- time[range_i]) )), 
        np.mean( (np.log(RES6[range_f] / RES6[range_i]) / (time[range_f]- time[range_i]) )), 
    ])


    sen_std = np.array([
        np.std( (np.log(SEN0[range_f] / SEN0[range_i]) / (time[range_f]- time[range_i]) ) ),
        np.std( (np.log(SEN1[range_f] / SEN1[range_i]) / (time[range_f]- time[range_i]) ) ),
        np.std( (np.log(SEN3[range_f] / SEN3[range_i]) / (time[range_f]- time[range_i]) ) ),
        np.std( (np.log(SEN6[range_f] / SEN6[range_i]) / (time[range_f]- time[range_i]) ) )
    ])

    res_std = np.array([
        np.std( (np.log(RES0[range_f] / RES0[range_i]) / (time[range_f]- time[range_i]) ) ),
        np.std( (np.log(RES6[range_f] / RES6[range_i]) / (time[range_f]- time[range_i]) ) )
    ])




    # define sensitive growth rate at 0 antibiotics
    ls = sen_mean_rates[0]

    # fit
    sen_growth_fixed_ls = partial(sen_growth, ls=ls)
    popt, cov = curve_fit(sen_growth_fixed_ls, a_line, sen_mean_rates, sigma=sen_std)
    print(popt)

    sen_growth2_fixed_ls = partial(sen_growth2, ls=ls)
    popt2, cov2 = curve_fit(sen_growth2_fixed_ls, a_line, sen_mean_rates, sigma=sen_std)
    print(popt2)

    plt.figure(2)
    plt.errorbar([0,6], res_mean_rates, yerr=res_std, fmt='o', label='RES', capsize=5)
    plt.errorbar(a_line, sen_mean_rates, yerr=sen_std, xerr=None, fmt ='o', label='SEN', capsize=5)
    plt.plot(a_cont := np.linspace(0, a_line[-1], 100), sen_growth(a_cont, popt, ls), '--', label='Best fit') 
    plt.plot(a_cont := np.linspace(0, a_line[-1], 100), sen_growth2(a_cont, popt2, ls), '--', label='Best fit2') 
    # plt.plot(np.linspace(0, a_line[-1], 100), ls / (1 + np.linspace(0, a_line[-1], 100) / popt), '--', label='Best fit') 
    plt.legend()
    plt.ylabel(r'$\lambda$ [h$^{-1}$]')
    plt.xlabel(r'Cm [$\mu$g/mL]')
    
    print('sen0', sen_mean_rates)
    print('res0',res_mean_rates)

plt.show(block=False)
input('[enter]')

