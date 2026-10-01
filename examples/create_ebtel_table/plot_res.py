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


DataDir = '/Users/bmondal/BM_Works/ISWAT/EBTEL_table/codes/ebtel_model/outputs/test'

#files = ['Lhalf50.00_mu-2.5849_coronal','Lhalf50.00_mu-2.5849_photospheric','Lhalf50.00_mu-2.5849_powerlaw']
#files = ['Lhalf50.00_mu-2.5849_coronal_dur18816000.0','Lhalf50.00_mu-2.5849_coronal_dur1000000.0','Lhalf50.00_mu-2.5849_coronal_dur100000.0']
#files = ['LInd000000_MuInd000000_Lhalf50.12_mu-2.5873_power_law_k1e-6','LInd000000_MuInd000000_Lhalf50.12_mu-2.5873_power_law_koriginal','LInd000000_MuInd000000_Lhalf50.12_mu-2.5873_power_law_k4.06e-7']

files = ['LInd000000_MuInd000000_Lhalf50.12_mu-2.5873_power_law_0.2c1','LInd000000_MuInd000000_Lhalf50.12_mu-2.5873_power_law_0.5c1','LInd000000_MuInd000000_Lhalf50.12_mu-2.5873_power_law_1.0c1']

def klimchuk_rad2(log_temperature):
        if( log_temperature <= 4.72 ):
            chi = 1.2e-31
            alpha = 2.0
        elif( log_temperature <= 6.1 ):
            chi = 3.31e-22
            alpha = 0.0
        elif( log_temperature <= 6.6 ):
            chi = 4.67e-13;
            alpha = -3/2;
        elif( log_temperature <= 6.95 ):
            chi = 1.35e-24
            alpha = 1/4.0
        elif( log_temperature <= 7.5 ):
            chi = 6.61e-16
            alpha = -1.0
        else :
            #// NOTE: free-free radiation is included in the parameter values for log_10 T > 7.63
            chi = 6.61e-28#3.72e-27
            alpha = 3./5.#1.0/2.0

        return (chi * (10.0**(alpha*log_temperature)))

plt.close('all')

#lab = ['1.9e7','1.0e6','1.0e5']
lab = ['1.0e-6','8.12e-7','4.06e-7']
lab = ['c1/5','c1/2','c1']

color=['r','b','g']
dem_all = []
dem_all_cor = []
dem_all_tr = []
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
    plt.plot(logt,dem_avg_corona,label=lab[i],color=color[i],ls='-',alpha=0.5)
    plt.plot(logt,dem_avg_tr,color=color[i],ls='--',alpha=0.5)

    logt_all+= [logt]
    dem_all += [dem_avg_total]
    dem_all_cor += [dem_avg_corona]
    dem_all_tr += [dem_avg_tr]

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
    R_loss = []
    for jj in range(len(dem_all)):
        ebtel_logT = logt_all[jj]
        ebtel_em = Dem2EM(logt_all[jj],dem_all[jj].value)
        tresp__ = intpFunc(ebtel_logT)
        #ind = np.where(ebtel_logT<5.0)[0]
        #ebtel_em = ebtel_em[ind]; tresp__ = tresp__[ind]
        I = np.sum(tresp__*ebtel_em)
        I_all+= [I]
    print(f"{chn[i]:10} {I_all[0]:15.4f} {I_all[1]:15.4f} {I_all[2]:15.4f}")
    
R_loss = []
R_loss_tr = []
#Calculate radiation loss:
for jj in range(len(dem_all)):
    ebtel_logT = logt_all[jj]
    ebtel_em_cor = Dem2EM(logt_all[jj],dem_all_cor[jj].value)
    ebtel_em_tr = Dem2EM(logt_all[jj],dem_all_tr[jj].value)
    r_loss_ = []
    for ii in ebtel_logT: r_loss_+=[klimchuk_rad2(ii)]
    R_loss += [np.sum(ebtel_em_cor*np.array(r_loss_))]
    R_loss_tr += [np.sum(ebtel_em_tr*np.array(r_loss_))]

