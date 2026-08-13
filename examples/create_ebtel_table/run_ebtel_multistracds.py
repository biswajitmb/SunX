#!/usr/bin/env python
# coding: utf-8

# 
#  # 🔬 Create EBTEL library for radom log-normal energy distribution and random delay distribution 
#  
#  **Purpose**  
#  This notebook generates **EBTEL DEMs** using random log-normal energy distribution and random delay distribution. Delay distribution follow a mixed function of log-normal till 1000s and after that a broken powerlaw.
#  
#  ---
#  
#  ## What this notebook does
#  
#  - Create random heating function for **log-normal** distribution
#  - Create random delay distribution for **log-normal** and **broken-powerlaw** distribution
#  - run EBTEL++ single fluid model
#  
#  ---
#  
#  ## Technical Details
#  
#  - **Code:** EBTEL++  
#  
#  ---
#  
#  📅 **Last modified:** Apr-29-2026  
#  ✍️ **Modified by:** Biswajit Mondal
#  

# In[7]:


#get_ipython().run_line_magic('matplotlib', 'widget')
import sunx as ar
import matplotlib.pyplot as plt
import numpy as np
import importlib
importlib.reload(ar)
import astropy.units as u
import time as tm
from joblib import Parallel, delayed
# Check Radom Energy distribution


#importlib.reload(ar)
mu=12.84 #12.99
sigma = 1.1 #0.907
duration= 2.0e6 #18816000
tau=50

OutDir = './outputs/product'

Parallel_run_Ncore = 8

log_L_halfs = np.arange(start=5.7, stop=8.8, step=0.05)
mu_original_all = np.arange(start=4, stop=19, step=0.1)


def process_r(l_ind,mu_ind, L_half,mu):

    print('%START: (l_ind,mu_ind) ',format('%d'%l_ind),'\t',format('%d'%mu_ind))
    start_time = tm.perf_counter()

    #distribution is in the unit of J/m2 from Shanwlee_et.al_2025
    unit_conv_fact = 1000/(L_half*1.0e8) #erg/cm3
    mu = mu + np.log(unit_conv_fact)


    qbkg = ar.util.H_back_loopTop(L_half*1.0e8,1.0e6)
    time,heat, Peak_time, Peak_heat, Mean_energy_flux = ar.util.nanoflareprof_logNormal(
        mu=mu,               #log(E) of nanoflare energy
        sigma = sigma,       #sigma of log-normal distribution
        E_low = 0.02,        #Lower log(energy) of log-normal distribition
        E_high = 3.0,        ##Upper log(energy) of log-normal distribition
        L_half=L_half,       #Half loop length in Mm
        #delay_param = [6.37446112, 1.73054859, 3.71667007e4,  -2.69852103,-6.94515675,9.66192460e8, 2500, 10, 4860, 1000, 10000],
        delay_param = [
                        6.31786474e+00, 1.71177384e+00, 3.30854734e+04,
                        -2.70969539e+00, -7.10910333e+00,  9.53518227e+08,
                        2500, 10, 5000, 1000, 100000],
        qbkg=qbkg,
        tau_half=int(tau//2),
        dur=duration,
        HeatingFunction = True,
        Test=False,
        )


    # Run EBTEL for a single heating distribution and plot the results
    sim = ar.fieldalign_model(configfile='NA')


    #abundance = 'photospheric'
    abundance = 'coronal'

    OutFile_name = 'LInd'+format('%0.6d'%l_ind)+'_'+'MuInd'+format('%0.6d'%mu_ind)+'_Lhalf'+format('%0.2f'%L_half)+'_mu'+format('%0.4f'%mu)+'_'+abundance+'_Dur'+format('%0.1f'%duration)

    result = sim.run_ebtelv0p5_general(
            L_half,         #Loop half length in Mm
            Peak_heat,      #heating energy distribution
            Peak_time,      #delay times
            BKG_T=1.0e6,    #To mentain loop-top temperature in Kelvin
            tau=tau,
            SimulationTime=duration,
            electron_ion_partition = 1,
            dem_logTnbins = 20,
            dem_logTmin = 4.5,
            dem_logTmax = 7.5,
            OutDir = OutDir,
            OutFile_name = OutFile_name,
            #Out_phys = False,
            partition = 1,
            saturation_limit=None,
            force_single_fluid=True,
            c1_conduction=6.0,
            c1_radiation=0.6,
            use_c1_gravity_correction=True,
            use_c1_radiation_correction=True,
            surface_gravity=1.0,
            helium_to_hydrogen_ratio=0.075,
            radiative_loss= abundance,#'power_law', 
            loop_length_ratio_tr_total=0.15,
            area_ratio_tr_corona=1.0,
            area_ratio_0_corona=1.0
            )
    end_time = tm.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Computation time: {elapsed_time:.6f} seconds")
    print('='*60)

    return l_ind, mu_ind, result


L_halfs = 10**log_L_halfs / 1.0e6 #in Mm

tasks = [(l_ind,mu_ind) for l_ind in range(len(L_halfs)) for mu_ind in range(len(mu_original_all))]

results = Parallel(n_jobs=Parallel_run_Ncore, backend="threading")
        (delayed(process_r)(l_ind,mu_ind, L_halfs[l_ind] , mu_original_all[mu_ind])
        for l_ind,mu_ind in tasks
        )    


