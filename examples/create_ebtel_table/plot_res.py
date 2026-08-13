import sunx as ar
import matplotlib.pyplot as plt
import numpy as np
import importlib
importlib.reload(ar)
import astropy.units as u
import os
from scipy.io import readsav
from scipy import interpolate
from scipy.ndimage import shift


DataDir = './outputs'

#files = ['Lhalf50.00_mu-2.5849_coronal','Lhalf50.00_mu-2.5849_photospheric','Lhalf50.00_mu-2.5849_powerlaw']
files = ['Lhalf50.00_mu-2.5849_coronal_dur18816000.0','Lhalf50.00_mu-2.5849_coronal_dur1000000.0','Lhalf50.00_mu-2.5849_coronal_dur100000.0']

plt.close('all')

lab = ['1.9e7','1.0e6','1.0e5']

color=['r','b','g']
dem_all = []
logt_all = []
label = []
for i in range(len(files)):
    f = os.path.join(DataDir,files[i])
    data = ar.util.load_obj(f)

    logt = np.log10(data['dem_temperature'].value)
    dem_cor = data['dem_corona']
    dem_tr = data['dem_tr']
   
    ind = data['time'].value > 10000
    dem_cor = data['dem_corona'][ind]
    dem_tr = data['dem_tr'][ind]

    delta_t = np.gradient(data['time'][ind])
    dem_avg_total = np.average(dem_cor+dem_tr,axis=0,weights=delta_t)
    dem_avg_tr = np.average(dem_tr,axis=0,weights=delta_t)
    dem_avg_corona = np.average(dem_cor,axis=0,weights=delta_t)
 
    label+= [files[i].split('_')[-1]]
    #plt.plot(logt,dem_avg_corona,label='Coronal ('+files[i].split('_')[-1]+')',color=color[i],ls='-')
    #plt.plot(logt,dem_avg_total,label='Total ('+files[i].split('_')[-1]+')',color=color[i],ls='--')
    plt.plot(logt,dem_avg_corona,label=lab[i],color=color[i],ls='-')
    plt.plot(logt,dem_avg_total,color=color[i],ls='--')

    logt_all+= [logt]
    dem_all += [dem_avg_total]

plt.yscale('log')
plt.ylabel('DEM (cm$^{-5}$ K$^{-1}$)')
plt.xlabel('logT')
plt.legend()
plt.show()

#Lets predict AIA intensities for different loss functions:

aia_rsp = '/Users/bmondal/BM_Works/ISWAT/EBTEL_table/codes/data/aia_tresp/aia_tresp_30072021.sav'

tresp = readsav(aia_rsp)

def Dem2EM(DEM_logT,DEM_Map):
    '''
    inputs:
        DEM_Map -> 1D array, dimension = [logT]
        DEM_logT -> DEM logT grids
    outputs: EM
    '''
    DEM_logT = DEM_logT
    DEM_Map = DEM_Map
    dT = (shift(DEM_logT, -1, cval=0.0) - shift(DEM_logT, 1, cval=0.0)) * 0.5
    ntemps = len(DEM_logT)
    dT[0] = DEM_logT[1] - DEM_logT[0]
    dT[ntemps-1] = (DEM_logT[ntemps-1]-DEM_logT[ntemps-2])

    Model_EM = DEM_Map*0
    Model_EM = (DEM_Map * (10**DEM_logT) *np.log(10.) * dT)
    #import pdb; pdb.set_trace()
    return Model_EM

print(f"{'AIA Channel':>10} {'I ('+label[0]+')':>15} {'I ('+label[1]+')':>15} {'I ('+label[2]+')':>15}")
print("-" * 60)


chn = tresp['channels'].astype('str')
for i in range(len(chn)):
    trsp = tresp['tr'][i,:]
    logt = tresp['logt']

    intpFunc = interpolate.interp1d(logt , trsp, bounds_error=False, fill_value=0)
    I_all = []
    for jj in range(len(dem_all)):
        ebtel_logT = logt_all[jj]
        ebtel_em = Dem2EM(logt_all[jj],dem_all[jj].value)
        tresp__ = intpFunc(ebtel_logT)
        I = np.sum(tresp__*ebtel_em)
        I_all+= [I]
    print(f"{chn[i]:10} {I_all[0]:15.4f} {I_all[1]:15.4f} {I_all[2]:15.4f}")




