import matplotlib.pyplot as plt
import numpy as np
import importlib
import os,glob
from scipy.io import readsav
from matplotlib.colors import LogNorm


idl_file = '/Users/bmondal/BM_Works/softwares/ssw/packages/GX_SIMULATOR/euv/ebtel/ebtel_scale=0.2_alpha=-2.5.sav'


da = readsav(idl_file)

lrun = da['lrun']
qrun = da['qrun']

L_all = lrun.flatten()
Vol_Q_all = qrun.flatten()
Avg_Q = Vol_Q_all*L_all

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

logt = da['LOGTDEM']
plt.plot(logt,da['DDM_TR_RUN'][10,10,:],label='TR')
plt.plot(logt,da['DDM_COR_RUN'][10,10,:],label='Coronal')
plt.yscale('log')
plt.ylabel('DDM')
plt.xlabel('logT')
plt.legend()
plt.show()


