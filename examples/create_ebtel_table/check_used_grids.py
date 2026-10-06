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

log_file = '/Users/bmondal/BM_Works/ISWAT/EBTEL_table/codes/ebtel_model/outputs/product/ebtel_run_log.txt'
#log_file = '/Volumes/Working/product/ebtel_run_log.txt'

log_data = np.loadtxt(log_file,skiprows=1)

l_ind = log_data[:,0]
mu_ind = log_data[:,1]
L_half_Mm = log_data[:,2]
mu_orig = log_data[:,3]
effective_duration = log_data[:,4]
status = log_data[:,5]
Avg_Nanoflare_heating_rate = log_data[:,6]
bkg_heating_rate = log_data[:,7]

#distribution is in the unit of J/m2 from Shanwlee_et.al_2025
unit_conv_fact = 1000/(L_half_Mm*1.0e8) #erg/cm3
mu_all = mu_orig + np.log(unit_conv_fact)
L_all = L_half_Mm*2*1.0e8 #cm

Avg_Q = (Avg_Nanoflare_heating_rate * L_all) #erg/cm2/s
Vol_Q_all = Avg_Nanoflare_heating_rate

L_unique = np.unique(L_all)

L_2d = np.stack([L_all[L_all == l] for l in L_unique])
Q_2d = np.stack([Vol_Q_all[L_all == l] for l in L_unique])

NL, NQ = L_2d.shape

ind_bad = np.where(status==1)
L_all[ind_bad] = np.nan
Avg_Q[ind_bad] = np.nan

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

norm = get_lognorm(np.array(Avg_Q))


sc = axs.scatter(L_all, Vol_Q_all, c=Avg_Q,cmap='jet', s=50,norm=norm,marker='s')
cbar = fig.colorbar(sc,ax=axs)
cbar.set_label('Area Average <Q> (erg cm$^{-2}$ s$^{-1}$)')

#ax_top = axs.secondary_xaxis('top', functions=(forward_transform, inverse_transform))

plt.xlabel('Loop Length (cm)')
plt.ylabel('Volumetric <Q> (erg cm$^{-3}$ s$^{-1}$)')
plt.xscale('log')
plt.yscale('log')

plt.show()

'''
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
'''


