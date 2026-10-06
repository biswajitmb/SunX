import matplotlib.pyplot as plt
import numpy as np
import importlib
import os,glob
import sunx as ar
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

OutDir = '/Volumes/Working/ebtel/outputs'
OutFile = 'ebtel_runs_final.h5'

ebtel_dir = '/Users/bmondal/BM_Works/ISWAT/EBTEL_table/codes/ebtel_model/outputs/product/'
ebtel_dir = '/Volumes/Working/ebtel/product/'
ebtel_output_files = np.sort(glob.glob(os.path.join(ebtel_dir,'*.pkl')))

log_file = os.path.join(ebtel_dir,'ebtel_run_log.txt')

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
Vol_Q_all = Avg_Nanoflare_heating_rate #erg/cm3/s

L_unique = np.unique(L_all)

L_2d = np.stack([L_all[L_all == l] for l in L_unique])
Q_2d = np.stack([Vol_Q_all[L_all == l] for l in L_unique])
Status_2d = np.stack([status[L_all == l] for l in L_unique])
l_ind_2d = np.stack([l_ind[L_all == l] for l in L_unique])
mu_ind_2d = np.stack([mu_ind[L_all == l] for l in L_unique])


def ebtel_coronal_dem_python(
    temperature,
    density,
    logT_mid,
    loop_length=1,
    c2=0.9,
):
    """
    Python implementation of the EBTEL++ coronal DEM calculation
    from dem.cpp.

    Parameters
    ----------
    temperature : array_like, shape (nt,)
        Electron temperature from EBTEL++ [K].

    density : array_like, shape (nt,)
        Electron/number density from EBTEL++ [cm^-3].

    logT_mid : array_like, shape (nT,)
        log10(T) values at which the DEM should be evaluated.

        Example
        -------
        logT_mid = np.arange(4.0, 8.51, 0.01)

    loop_length : float
        Total Loop length used in the DEM calculation [cm].

        IMPORTANT:
        This should correspond to the length used by the EBTEL++
        DEM calculation, not a value in Mm.

        For example:
            50 Mm = 50e8 cm

    Note: If loop_length = 1, the Unit of DEM will be cm^-6 K^-1 and DDM unit will be cm^{-3} K^{-1}

    c2 : float, optional
        EBTEL c2 parameter = T_average / T_apex.
        Default is 0.9.

    Returns
    -------
    dem_corona : ndarray, shape (nt, nT)
        Coronal DEM [cm^-5 K^-1].

    T_corona_min : ndarray, shape (nt,)
        Lower coronal temperature used by EBTEL++ [K].

    T_corona_max : ndarray, shape (nt,)
        Upper coronal temperature used by EBTEL++ [K].

    delta_temperature : ndarray, shape (nt,)
        Temperature interval used to normalize the coronal DEM [K].
    """

    temperature = np.asarray(temperature, dtype=float)
    density = np.asarray(density, dtype=float)
    logT_mid = np.asarray(logT_mid, dtype=float)

    if temperature.shape != density.shape:
        raise ValueError(
            "temperature and density must have the same shape."
        )

    if temperature.ndim != 1:
        raise ValueError(
            "temperature and density should be 1-D arrays."
        )

    # ---------------------------------------------------------
    # DEM temperature grid
    # ---------------------------------------------------------

    T_dem = 10.0**logT_mid

    # ---------------------------------------------------------
    # EBTEL++ coronal temperature limits
    #
    # dem.cpp:
    #
    # Tmax = max(Te / c2, 1.1e4)
    #
    # Tmin = max(
    #     Te * (2 - 1/c2),
    #     1.0e4
    # )
    # ---------------------------------------------------------

    T_corona_max = np.maximum(
        temperature / c2,
        1.1e4,
    )

    T_corona_min = np.maximum(
        temperature * (2.0 - 1.0 / c2),
        1.0e4,
    )

    # ---------------------------------------------------------
    # EBTEL++ normalization interval
    #
    # The +/-0.5/100 terms come directly from the DEM
    # temperature-grid treatment in dem.cpp.
    # ---------------------------------------------------------

    delta_temperature = (
        10.0**(0.5 / 100.0) * T_corona_max
        -
        10.0**(-0.5 / 100.0) * T_corona_min
    )

    # ---------------------------------------------------------
    # Coronal DEM level
    #
    # dem.cpp:
    #
    # DEM_corona = 2 * n^2 * L / delta_temperature
    # ---------------------------------------------------------

    dem_level = (
        density**2
        * loop_length
        / delta_temperature
    )

    # ---------------------------------------------------------
    # Coronal DDM
    # DDM_corona = 2 * n * L / delta_temperature
    # ---------------------------------------------------------
    ddm_level = (
        density
        * loop_length
        / delta_temperature
    )

    # ---------------------------------------------------------
    # Create output array
    # ---------------------------------------------------------

    dem_corona = np.zeros(
        (temperature.size, logT_mid.size),
        dtype=float,
    )

    ddm_corona = np.zeros(
        (temperature.size, logT_mid.size),
        dtype=float,
    )

    # ---------------------------------------------------------
    # DEM is constant between Tmin and Tmax
    # ---------------------------------------------------------

    mask = (
        (T_dem[None, :] >= T_corona_min[:, None])
        &
        (T_dem[None, :] <= T_corona_max[:, None])
    )

    dem_corona[mask] = np.broadcast_to(
        dem_level[:, None],
        dem_corona.shape,
    )[mask]

    ddm_corona[mask] = np.broadcast_to(
        ddm_level[:, None],
        ddm_corona.shape,
    )[mask]

    return dem_corona,ddm_corona #,T_corona_min,T_corona_max,delta_temperature,
   
def calculate_ddm(
    dem_corona,
    dem_tr,
    density,
    electron_temperature,
    ion_temperature,
    electron_pressure,
    loop_half_length,
    helium_to_hydrogen_ratio=0.075,
    surface_gravity=1.0,
):

    """
    Calculate coronal and transition-region Differential Density Measure
    (DDM) from EBTEL++ DEM outputs.

    This follows the prescription used in Jim Klimchuk's ebtel2dd.pro
    and the EBTEL++ DEM convention:

        DDM(t, T) = DEM(t, T) / n(t, T)

    Parameters
    ----------
    dem_corona : array_like, shape (nt+1, nT)
        EBTEL++ coronal DEM array.

        First row must contain the temperature grid [K].
        Remaining rows contain DEM(t,T) [cm^-5 K^-1].

    dem_tr : array_like, shape (nt+1, nT)
        EBTEL++ transition-region DEM array.

        First row must contain the temperature grid [K].
        Remaining rows contain DEM(t,T) [cm^-5 K^-1].

    density : array_like, shape (nt,)
        EBTEL++ coronal electron density [cm^-3].

    electron_temperature : array_like, shape (nt,)
        Mean coronal electron temperature [K].

    ion_temperature : array_like, shape (nt,)
        Mean coronal ion temperature [K].

    electron_pressure : array_like, shape (nt,)
        Electron pressure [erg cm^-3].

    loop_half_length : float
        Loop length used in the gravitational correction [cm].

        This should have the same meaning as ``loop_length`` in the
        original ebtel2dd.pro / EBTEL++ calculation.

    helium_to_hydrogen_ratio : float, optional
        Helium-to-hydrogen abundance ratio.
        Default = 0.075.

    surface_gravity : float, optional
        Surface gravity relative to the solar surface gravity.
        Default = 1.0.

    Returns
    -------
    ddm_corona : ndarray, shape (nt+1, nT)
        Coronal DDM.

        First row contains temperature [K].
        Remaining rows contain DDM [cm^-2 K^-1].

    ddm_tr : ndarray, shape (nt+1, nT)
        Transition-region DDM.

        First row contains temperature [K].
        Remaining rows contain DDM [cm^-2 K^-1].

    n_tr : ndarray, shape (nt, nT)
        Transition-region electron density used in the calculation
        [cm^-3].

    scale_height : ndarray, shape (nt,)
        Gravitational pressure scale height [cm].
    """
    # ---------------------------------------------------------
    # Convert inputs to numpy arrays
    # ---------------------------------------------------------

    dem_corona = np.asarray(dem_corona, dtype=float)
    dem_tr = np.asarray(dem_tr, dtype=float)

    density = np.asarray(density, dtype=float)
    electron_temperature = np.asarray(
        electron_temperature,
        dtype=float,
    )
    ion_temperature = np.asarray(
        ion_temperature,
        dtype=float,
    )
    electron_pressure = np.asarray(
        electron_pressure,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Physical constants, cgs
    # ---------------------------------------------------------

    k_B = 1.380649e-16       # Boltzmann's constant [erg K^-1]
    m_p = 1.672621923e-24    # Proton mass [g] 
    g_s = 2.74e4             # Surface gravity of the Sun [cm/s^2]

    # ---------------------------------------------------------
    # Temperature grid
    # ---------------------------------------------------------

    temp = dem_tr[0, :]

    # ---------------------------------------------------------
    # Helium corrections (# Calculate k_B and m_p corrections based on Helium abundance)
    # ---------------------------------------------------------

    he_h = helium_to_hydrogen_ratio

    z_avg = (
        (1.0 + 2.0 * he_h)
        / (1.0 + he_h)
    )

    #k_B_correct = k_B / z_avg  # ebtellplusplus version
    # This is the correction used in ebtel2.pro / ebtel2dd.pro
    k_B_correct = (k_B* 0.5* (1.0 + 1.0 / z_avg))

    ion_mass_correction = (
        (1.0 + 4.0 * he_h)
        / (2.0 + 3.0 * he_h)
        * (1.0 + z_avg) / z_avg)

    m_p_correct = (m_p* ion_mass_correction)

    # =========================================================
    # CORONAL DDM
    # =========================================================
    #
    # DDM_corona = DEM_corona / n_corona
    #
    # ---------------------------------------------------------

    ddm_corona = np.zeros_like(dem_corona)

    # Divide every temperature bin by coronal density
    ddm_corona[0:, :] = (
        dem_corona[0:, :]
        / density[0:, None]
    )

    # =========================================================
    # TRANSITION-REGION DDM
    # =========================================================

    # ---------------------------------------------------------
    # Gravitational pressure scale height
    # Calculate transition region gravitational scale height with temperature
    #
    # H =
    #
    #   k_B * (Te + Ti / z_avg)
    #   ------------------------
    #       m_p_correct * g
    #
    # ---------------------------------------------------------

    scale_height = (
        k_B
        * (electron_temperature + ion_temperature / z_avg)
        / (
            m_p_correct
            * surface_gravity
            * g_s
        )
    )

    # ---------------------------------------------------------
    # Local transition-region density
    #
    # n_TR(T) =
    #
    #      p_e
    #  -------------
    #   k_B,corr T
    #
    # × exp[
    #
    #      2 L sin(pi/5)
    #      ---------------
    #          pi H
    #
    # ]
    #
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # # Compute transition region DDM
    #
    # DDM_TR(T) = DEM_TR(T) / n_TR(T)
    #
    # ---------------------------------------------------------

    p_e_array = np.broadcast_to(electron_pressure[:,None], dem_tr[0:,:].shape)
    temp_array = np.broadcast_to(electron_temperature[:,None], dem_tr[0::].shape)
    scale_height_array = np.broadcast_to(scale_height[:,None], dem_tr[0::].shape)
    n_tr = ((p_e_array / k_B_correct / temp_array)
            * np.exp(2 * loop_half_length * np.sin(np.pi/5) / scale_height_array / np.pi))
    ddm_tr = np.zeros(dem_tr.shape)
    ddm_tr[0:,:] = dem_tr[0:,:] / n_tr

    return ddm_corona,ddm_tr,n_tr,scale_height


NL, NQ = L_2d.shape
NT = 80 #number of logT bin

dem_tr = np.zeros([NL,NQ, NT])
dem_cor = np.zeros([NL,NQ, NT])
ddm_tr = np.zeros([NL,NQ, NT])
ddm_cor = np.zeros([NL,NQ, NT])
lrun = L_2d / 2 #loop half length
qrun = Q_2d
trun = np.zeros([NL,NQ]) ##maximum electron temperature during the run in K

for l in range(NL):
    for q in range(NQ):
        if Status_2d[l,q] == 0:
            print('Start: (l,q)',l,q)
            ebtel_file = glob.glob(os.path.join(ebtel_dir,'LInd'+format('%0.6d'%l_ind_2d[l,q])+'_'+'MuInd'+format('%0.6d'%mu_ind_2d[l,q])+'_Lhalf*'+'_mu*'+'_power_law.pkl'))[0]

            data = ar.util.load_obj(ebtel_file[0:-4])

            logt = np.log10(data['dem_temperature'].value)
            dem_cor_ = data['dem_corona'].value
            dem_tr_ = data['dem_tr'].value

            time_ = data['time'].value
            electron_temperature = data['electron_temperature'].value
            ion_temperature = data['ion_temperature'].value
            electron_pressure = data['electron_pressure'].value
            density = data['density'].value

            ind = time_ > 5000
            dem_cor_ = dem_cor_[ind]
            dem_tr_ = dem_tr_[ind]

            time_ = time_[ind]
            electron_temperature = electron_temperature[ind]
            ion_temperature = ion_temperature[ind]
            electron_pressure = electron_pressure[ind]
            density = density[ind]

            delta_t = np.gradient(time_)
            #dem_avg_total = np.average(dem_cor+dem_tr,axis=0,weights=delta_t)
            dem_avg_tr = np.average(dem_tr_,axis=0,weights=delta_t)
            dem_avg_corona = np.average(dem_cor_,axis=0,weights=delta_t)

            TR_L = 0.15*(L_2d[l,q])
            coronal_L = 0.85*(L_2d[l,q])

            '''            
            #Independent calculation from tem and density array
            dem,ddm=ebtel_coronal_dem_python(
                electron_temperature,
                density,
                logt,
                loop_length = 1,#coronal_L, #Provide loop_length = 1 to match the unit of Eq 2,3 of Gelu et al 2023.
                c2=0.9,)
            dem = np.average(dem,axis=0,weights=delta_t) 
            ddm = np.average(ddm,axis=0,weights=delta_t)
            dem_cor[l,q,:] = dem
            dem_tr[l,q,:] = dem_avg_tr
            ddm_cor[l,q,:] = ddm
            '''
            loop_half_length = L_2d[l,q]/2
            ddm_corona_,ddm_tr_,n_tr,scale_height = calculate_ddm(
                dem_cor_,
                dem_tr_,
                density,
                electron_temperature,
                ion_temperature,
                electron_pressure,
                loop_half_length*0.85, #loop half-length in corona [cm]
                helium_to_hydrogen_ratio=0.075,
                surface_gravity=1.0,
            )

            ddm_cor_ave = np.average(ddm_corona_, axis=0, weights=delta_t) #in unit of cm^-2 K^-1
            ddm_tr_ave = np.average(ddm_tr_,axis=0,weights=delta_t) #Thus was estimated using total loop length instead of loop_half_length
          
            dem_avg_corona = dem_avg_corona/(coronal_L) # Average DEM in the unit of cm^-6 K^-1
            ddm_cor_ave = ddm_cor_ave / (coronal_L) # Average DDM in the unit of cm^-3 K^-1

            #Note: In GX simulator TR is considerd as a single voxcel layer at the bottom. Thus its the total TR emission instead of perunit length. However, as EBTEL calculate it for the entire loop length only one footpoint need to be store for GX-simulator.
            dem_avg_tr = dem_avg_tr /2 # Average DEM over the two footpoints
            ddm_tr_ave = ddm_tr_ave / 2 # Average DDM over the two footpoints


            '''
            #plt.plot(logt,dem_avg_corona)
            #plt.plot(logt,dem,ls='--')
            #plt.twinx()
            #plt.plot(logt,ddm,ls='-',color='r')
            #plt.twinx()
            #plt.plot(logt,ddm_tr_ave,ls='-')
            #plt.plot(logt,ddm_cor_ave,ls='--')

            plt.plot(logt,dem_avg_tr,ls='-')
            plt.plot(logt,dem_avg_corona,ls='--')
            plt.yscale('log')

            plt.twinx()
            plt.plot(logt,ddm_tr_ave,ls='-',color='r')
            plt.plot(logt,ddm_cor_ave,ls='--',color='r')
           
            plt.yscale('log')
            plt.show()
            ''' 

            dem_cor[l,q,:] = dem_avg_corona 
            dem_tr[l,q,:] = dem_avg_tr
            ddm_cor[l,q,:] = ddm_cor_ave 
            ddm_tr[l,q,:] = ddm_tr_ave 
            trun[l,q] = electron_temperature.max()

with h5py.File(os.path.join(OutDir,OutFile), "w") as f:
    f.create_dataset("LOGTDEM", data=np.asarray(logt, dtype=np.float32))
    f.create_dataset("LRUN", data=np.asarray(lrun, dtype=np.float32))
    f.create_dataset("QRUN", data=np.asarray(qrun, dtype=np.float32))
    f.create_dataset("TRUN", data=np.asarray(trun, dtype=np.float32))

    f.create_dataset("DEM_COR_RUN", data=np.asarray(dem_cor, dtype=np.float32))
    f.create_dataset("DEM_TR_RUN", data=np.asarray(dem_tr, dtype=np.float32))
    f.create_dataset("DDM_COR_RUN", data=np.asarray(ddm_cor, dtype=np.float32))
    f.create_dataset("DDM_TR_RUN", data=np.asarray(ddm_tr, dtype=np.float32))

print(os.path.join(OutDir,OutFile))

 
