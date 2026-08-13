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
# Check Radom Energy distribution


#importlib.reload(ar)
mu=12.84 #12.99
sigma = 1.1 #0.907
sim_times = [1.0e6]#18816000, 1.0e6,1.0e5]
#duration=18816000/10
L_half = 100#50
tau=40

#distribution is in the unit of J/m2 from Shanwlee_et.al_2025
unit_conv_fact = 1000/(L_half*1.0e8) #erg/cm3
mu = mu + np.log(unit_conv_fact)

plt.close('all')

OutDir = './outputs'

for t in range(len(sim_times)):
    duration = sim_times[t]

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
       qbkg=1.94e-6,
       tau_half=int(tau//2),
       dur=duration,
       HeatingFunction = True,
       Test=True,
       )
    
    
    # Run EBTEL for a single heating distribution and plot the results
    
    # In[34]:
    
    
    sim = ar.fieldalign_model(configfile='NA')
    
    
    abundance_all = ['power_law','photospheric','coronal']
    #abundance = 'photospheric'
    #abundance = 'coronal'
    
    for i in range(2,len(abundance_all)):
        abundance = abundance_all[i]
        OutFile_name = 'Lhalf'+format('%0.2f'%L_half)+'_mu'+format('%0.4f'%mu)+'_'+abundance+'_dur'+format('%0.1f'%duration)

        start_time = tm.perf_counter()    
        result = sim.run_ebtelv0p5_general(
                L_half,         #Loop half length in Mm
                Peak_heat,      #heating energy distribution
                Peak_time,      #delay times
                BKG_T=1.0e6,    #Average loop background temperature in Kelvin
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

# In[37]:


data = ar.util.load_obj('test')
### --- Plot T, density, heating ------
#plt.close('all')
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(data['time'], data['heat'])
axes[0].set_ylabel('Heat (erg cm$^{-3}$ s$^{-1}$)')
axes[1].plot(data['time'], data['electron_temperature']/1.0e6, label='electron')
#axes[1].plot(data['time'], data['ion_temperature'], label='ion')
axes[1].set_ylabel('T (MK)')
axes[2].plot(data['time'], data['density'])
axes[2].set_ylabel('n (cm$^{-3}$)'); axes[2].set_xlabel('Time (s)')
axes[2].set_xlim([1000,max(data['time'])])
axes[2].set_yscale('log')
axes[1].legend()
plt.show()

###--- Plot DEM -----
delta_t = np.gradient(data['time'])
dem_avg_total = np.average(data['dem_tr']+data['dem_corona'],axis=0,weights=delta_t)
dem_avg_tr = np.average(data['dem_tr'],axis=0,weights=delta_t)
dem_avg_corona = np.average(data['dem_corona'],axis=0,weights=delta_t)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(data['dem_temperature'], dem_avg_total, label='Total')
ax.plot(data['dem_temperature'], dem_avg_tr, label='TR')
ax.plot(data['dem_temperature'], dem_avg_corona, label='Corona')
ax.set_xlim([10**(4.5), 10**(7.5)])
ax.set_ylim([10**(20.0), 10**(23.5)])
ax.set_xlabel('log(T)'); 
ax.set_ylabel('DEM (cm$^{-5}$ K$^{-1}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

plt.show()

'''
# Lets create multiple distribution with different mean values

# In[45]:


log_L_halfs = np.arange(start=6, stop=10.6, step=0.1)
L_halfs = 10**log_L_halfs / 1.0e6 # in Mm

mu_original=12.99
unit_conv_fact = 1000/(L_halfs*1.0e8) #erg/cm3
mu2 = mu_original + np.log(unit_conv_fact)
np.exp(mu2),L_halfs


# In[41]:


log_L_halfs = np.array([6])#np.arange(start=6, stop=10.6, step=0.1)
L_halfs = 10**log_L_halfs / 1.0e6 # in Mm

mu_original= 12.84#12.99
sigma = 1.1 #0.907
duration=18816000
tau=40


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#axs = [axs]
for l in range(len(L_halfs)):
    L_half = L_halfs[l]
    #distribution is in the unit of J/m2 from Shanwlee_et.al_2025
    unit_conv_fact = 1000/(L_half*1.0e8) #erg/cm3
    mu = mu_original + np.log(unit_conv_fact)

    time,heat, Peak_time, Peak_heat, Mean_energy_flux = ar.util.nanoflareprof_logNormal(
       mu=mu,               #log(E) of nanoflare energy
       sigma = sigma,       #sigma of log-normal distribution
       E_low = 0.02,        #Lower log(energy) of log-normal distribition
       E_high = 1.4,        ##Upper log(energy) of log-normal distribition
       L_half=L_half,       #Half loop length in Mm
       #delay_param = [6.37446112, 1.73054859, 3.71667007e4,  -2.69852103,-6.94515675,9.66192460e8, 2500, 10, 4860, 1000, 10000],
       delay_param = [
                   6.31786474e+00, 1.71177384e+00, 3.30854734e+04,
                   -2.70969539e+00, -7.10910333e+00,  9.53518227e+08,
                   2500, 10, 5000, 1000, 100000],
       qbkg=1.0e-5,
       tau_half=tau/2,
       dur=duration,
       HeatingFunction = True,
       Test=False,
       )

    a,b=np.histogram(np.array(Peak_heat),bins='auto')
    xE = (b[1::]+b[0:-1])/2.0
    dx = (b[1::]-b[0:-1])
    yE = a/dx    
    axs[0].set_title('Energy distribution')

    axs[0].step(xE,yE, alpha=0.3, label = '$\mu$:'+format('%0.4f'%(10**mu))+', L:'+format('%0.2f'%(2*L_half)))

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('erg cm$^{-3}$ s$^{-1}$')
    axs[0].set_ylabel('N/bin')
    axs[0].legend()
    
    
    ##Plot delay distribution
    #N = np.diff(Peak_time)
    ##a,b,=np.histogram(N[N<delay_param[9]],bins='auto')
    #a,b,=np.histogram(N,bins='auto')
    #delay = (b[1::]+b[0:-1])/2.0
    #dx = (b[1::]-b[0:-1])
    #n = a/dx
    #axs[1].scatter(delay,n)
    #axs[1].set_title('Delay distribution')
    #axs[1].set_yscale('log'); axs[1].set_xscale('log')
    #axs[1].set_ylabel('N/bin')
    #axs[1].set_xlabel('Delay')
    #axs[1].legend()
     
    sim = ar.fieldalign_model(configfile='NA')
    result = sim.run_ebtelv0p5_general(
        L_half,         #Loop half length in Mm
        Peak_heat,      #heating energy distribution
        Peak_time,      #delay times
        BKG_T=0.3e6,    #Average loop background temperature in Kelvin
        tau=tau,
        SimulationTime=duration,
        electron_ion_partition = 1,
        dem_logTnbins = 20,
        dem_logTmin = 4.5,
        dem_logTmax = 7.5,
        OutDir = './',
        OutFile_name = 'test',
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
        radiative_loss= 'photospheric',#'power_law', 
        loop_length_ratio_tr_total=0.15,
        area_ratio_tr_corona=1.0,
        area_ratio_0_corona=1.0
        )

    ###--- Plot DEM -----
    delta_t = np.gradient(result.time)
    dem_avg_total = np.average(result.dem_tr+result.dem_corona,axis=0,weights=delta_t)
    dem_avg_tr = np.average(result.dem_tr,axis=0,weights=delta_t)
    dem_avg_corona = np.average(result.dem_corona,axis=0,weights=delta_t)

    
    ax = axs[1]
    line, = ax.plot(result.dem_temperature, dem_avg_total,alpha=0.3, label = '$\mu$:'+format('%0.4f'%(10**mu))+', L:'+format('%0.2f'%(2*L_half)))
    #ax.plot(data['dem_temperature'], dem_avg_tr, label='TR')
    line, = ax.plot(result.dem_temperature, dem_avg_corona,alpha=0.3,color=line.get_color(),ls='--')
    ax.set_xlim([10**(4.5), 10**(7.5)])
    ax.set_ylim([10**(20.0), 10**(23.5)])
    ax.set_xlabel('log(T)'); 
    ax.set_ylabel('DEM (cm$^{-5}$ K$^{-1}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('DEM')
    ax.legend()
plt.show()


# In[ ]:





# In[6]:

'''
