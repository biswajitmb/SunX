'''
This script will help in deciding the grid values of Loop length and mean of lognormal distribution.

Biswajit, Jun.24.2026
'''

import sunx as ar
import matplotlib.pyplot as plt
import numpy as np
import importlib
importlib.reload(ar)
import astropy.units as u
from matplotlib.colors import LogNorm

#importlib.reload(ar)
mu_original= 12.84 #12.99
sigma = 1.1 #0.907
duration= 18816000 #10000

#Lets consider L_half grid in between 0.1 Mm (or L = 0.2Mm) to 500Mm (or L = 1000Mm). This came from 
log_L_halfs = np.arange(start=5.7, stop=8.8, step=0.05)
tau=40

tau_half = tau / 2

L_halfs = 10**log_L_halfs / 1.0e6 # in Mm

#L_halfs = [L_halfs[10]]

#mu_original_all = np.arange(start=12.99-4, stop=12.99+8, step=0.1)
mu_original_all = np.arange(start=4, stop=19, step=0.1)


L_all = []
mu_all = []
Q_all = []
Q_bkg = []
Vol_Q_all = []
for l in range(len(L_halfs)):
    for q in range(len(mu_original_all)):
        L_half = L_halfs[l]
        #distribution is in the unit of J/m2 from Shanwlee_et.al_2025
        unit_conv_fact = 1000/(L_half*1.0e8) #erg/cm3
        mu = mu_original_all[q] + np.log(unit_conv_fact)

        L_all+=[L_half*2*1.0e8]
        mu_all += [mu]
        Q_bkg += [ar.H_back(L_half*2*1.0e8,0.3e6)]

        #Decide energy grids:
        average_Q_vol = np.average(np.random.lognormal(mean=mu, sigma=sigma, size=int(2*duration)))
        median_Q = (average_Q_vol * L_halfs[l]*1.0e8 / (tau_half)) #erg/cm2/s
        print('Average energy budget (1.0e7 erg/cm2/s): ',median_Q/1.0e7)


        Q_all+= [median_Q] #erg/cm2/s 
        Vol_Q_all += [average_Q_vol/tau_half] ##erg/cm3/s

        #plt.plot(L_halfs[l]*1.0e8*2,mu,'*')
        #sc = axs.scatter(L_halfs[l]*1.0e6, mu_original_all[q], c=median_Q,cmap='viridis', s=50)

        '''

        plt.close('all')
        time,heat, Peak_time, Peak_heat, Mean_energy_flux = ar.util.nanoflareprof_logNormal(
           mu=mu,               #log(E) of nanoflare energy
           sigma = sigma,       #sigma of log-normal distribution
           E_low = 0.02,        #Lower log(energy) of log-normal distribition
           E_high = 1.4,        ##Upper log(energy) of log-normal distribition
           L_half=L_half,       #Half loop length in Mm
           delay_param = [6.37446112, 1.73054859, 3.71667007e4,  -2.69852103,-6.94515675,9.66192460e8, 2500, 10, 4860, 1000, 10000],
           qbkg=1.0e-5,
           tau_half=int(tau//2),
           dur=duration,
           HeatingFunction = True,
           Test=False,
           )

        plt.show()
        '''

def get_lognorm(data, pmin=5, pmax=99):
    valid = data[np.isfinite(data) & (data > 0)]
    if len(valid) == 0:
        return None
    vmin = np.percentile(valid, pmin)
    vmax = np.percentile(valid, pmax)
    if vmin <= 0 or vmax <= 0:
        return None
    return LogNorm(vmin=vmin, vmax=vmax)

def forward_transform(x):
    return np.interp(x, L_all, Q_bkg)
def inverse_transform(x):
    return np.interp(x, Q_bkg, L_all)


fig, axs = plt.subplots(1, 1, figsize=(8, 4))

norm = get_lognorm(np.array(Q_all))


sc = axs.scatter(L_all, Vol_Q_all, c=Q_all,cmap='jet', s=50,norm=norm,marker='s')
cbar = fig.colorbar(sc,ax=axs)
cbar.set_label('Area Average <Q> (erg cm$^{-2}$ s$^{-1}$)')

#ax_top = axs.secondary_xaxis('top', functions=(forward_transform, inverse_transform))

plt.xlabel('Loop Length (cm)')
plt.ylabel('Volumetric <Q> (erg cm$^{-3}$ s$^{-1}$)')
plt.xscale('log')
plt.yscale('log')

plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 5))

norm = get_lognorm(np.array(L_all))
sc = axs.scatter(Q_bkg, Vol_Q_all, c=L_all,cmap='jet', s=50,norm=norm,marker='s')
cbar = fig.colorbar(sc,ax=axs)
cbar.set_label('L (cm)')
#plt.loglog(Q_bkg,Vol_Q_all,'s',alpha=0.2)
plt.xlabel('background <Q> (erg cm$^{-3}$ s$^{-1}$)')
plt.ylabel('Avg. Volumetric <Q> (erg cm$^{-3}$ s$^{-1}$)')

plt.xscale('log')
plt.yscale('log')
plt.show()



