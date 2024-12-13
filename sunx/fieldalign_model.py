import configparser
from sunx.util import *
import numpy as np
import os,sys,glob,time
import pickle 
from scipy import interpolate
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage.interpolation import shift
import multiprocessing as mul
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from sunx.util_ebtel import run_ebtel, read_xml

#import warnings
#warnings.simplefilter('ignore')
#import logging, sys
#logging.disable(sys.maxsize)

mul.set_start_method('fork',force=True)#to resolve the issue with mac and multiprocessing

class fieldalign_model(object):
    def __init__(self,configfile=None):
        if os.path.isfile(configfile) is False:
            raise Exception("%% simar_error : Congig-File not exist- "+configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config = config


    def iteration_over_loops_ebtel(self, l_ind):
        L_half=self.L_half; B_avg = self.B_avg; 
        c = self.c ; Vs=self.Vs; tau_half= self.tau_half; sub_dir = self.sub_dir; SimulationTime=self.SimulationTime; BKG_T=self.BKG_T; Fp=self.Fp; B_Coronal_base=self.B_Coronal_base
        L_ind_no=self.L_ind_no; EBTEL_dir=self.EBTEL_dir; EBTEL_ConfigFle = self.EBTEL_ConfigFle; Out_phys = self.Out_phys

        OutFile_name = 'EBTEL_Lhalf_'+format('%d'%L_half[l_ind])+'_Lind_'+ind_6(L_ind_no[l_ind])
        q0max = (c*B_avg[l_ind])**2 / (tau_half*8.0*np.pi) #(erg/cm3/s) maximum heating rate that a loop can afforts
        q0min = self.q0min_fract*q0max #Consider as 1% of q0max

        Store_outputs = True

        if Fp is None: Fp = Estimate_PoyntingFlux(L_half = L_half[l_ind],Vs=Vs, c = c,Fp = Fp,B_avg =  B_avg[l_ind],B_Coronal_base=B_Coronal_base[l_ind])

        #Estimate the Peak_time, Peak_heat
        Peak_time, Peak_heat = nanoflareprof(qbkg=0.0, tau=tau_half, dur=SimulationTime, q0min=q0min, q0max=q0max,
                          Fp = Fp,  L_half = L_half[l_ind], PrintOut = False,HeatingFunction = False, seed = None)
        ''' 
        E_ = Peak_heat*tau_half*3.14*2.0*L_half[l_ind]*Fixed_radious*Fixed_radious*1.0e24
        ind = np.where(E_ > 1.0e24)
        Peak_time = Peak_time[ind]
        Peak_heat = Peak_heat[ind]
        '''
        #Peak_time = Peak_time[Peak_heat > 1.0e-4]
        #Peak_heat = Peak_heat[Peak_heat > 1.0e-4]

        ### Configure and run ebtel++
        top_dir = EBTEL_dir
        base = read_xml(EBTEL_ConfigFle)
        base['calculate_dem'] = True
        base['heating']['partition'] = 0.5 #0.0 # 0-> pure ion heating, 1-> pure e heating
        base['dem']['temperature']['bins'] = self.logTnbins
        base['dem']['temperature']['log_min'] = self.logTmin
        base['dem']['temperature']['log_max'] = self.logTmax
        base['use_adaptive_solver'] = True
        base['adaptive_solver_safety'] = 0.5
        base['total_time'] = SimulationTime #+BKG_heating_time
        base['loop_length'] = L_half[l_ind]*1.0e8 #Loop loop half-length in cm
        base['heating']['background'] = H_back(L_half[l_ind]*1.0e8,BKG_T) #Set the background heating to mentain a temperature of ~0.5 Mk 

        #base['use_flux_limiting'] = True
        #base['saturation_limit'] = 0.5 #Flux limiter, f in section 2.1 of Barnes et al. (2016)

        events = []
        for event_ind in range(len(Peak_heat)):
            Rise_Start = Peak_time[event_ind] - tau_half #+BKG_heating_time
            Rise_End = Peak_time[event_ind] #+BKG_heating_time
            Decay_Start = Rise_End
            Decay_End = Rise_End + tau_half
            events.append({'event':
                        {'rise_start':Rise_Start,
                        'rise_end':Rise_End,
                        'decay_start':Decay_Start,
                        'decay_end':Decay_End,
                        'magnitude': Peak_heat[event_ind]}})
        base['heating']['events'] = events

        results = run_ebtel(base, top_dir)
        #EBTEL outputs
        #ind1 = 0 #store only the run of last 2h of data
        #ind2 = -1
        results['dem_tr'] = results['dem_tr']
        results['dem_corona'] = results['dem_corona']
        results['time'] = results['time']
        results['heat'] = results['heat']
        results['electron_temperature'] = results['electron_temperature']
        results['dem_total'] = results['dem_tr'] + results['dem_corona']
        if Out_phys is False:
            results['ion_temperature'] = 0 #results['ion_temperature'][ind1:ind2]
            results['velocity'] = 0
            results['ion_pressure'] = 0
            results['electron_pressure'] = 0
        results['Loop_half_length'] = L_half[l_ind]
        #results['Loop_half_length_projected'] = L_half_projected[l_ind]
        results['loop_index'] = L_ind_no[l_ind]
        results['peak_heating_rate'] = Peak_heat
        results['peak_heating_time'] = Peak_time
        results['tau_half'] = tau_half
        if Fp is None : results['F_p(erg/cm2/s)'] = (c/(4.0*3.14))*Vs*1.0e5*(B_avg[l_ind]*B_Coronal_base[l_ind])
        else: results['F_p(erg/cm2/s)'] = Fp
        results['B_avg'] = B_avg[l_ind]
        results['B_Coronal_base'] = B_Coronal_base[l_ind]
        #if Store_outputs is True and chr_delay > 50.0:
        if Store_outputs is True :
            # Store the outputs of each-loop
            save_obj(results, os.path.join(sub_dir,OutFile_name))
        return None

    def run_Ebtel(self,L_half,B_avg,B_Coronal_base,L_ind_no,  V_h, tan_theta, OutDir, SimulationTime = 1000, tau_half = 50, q0min_fract = 0.01, BKG_T = 5.0e5, Fp = None,Out_phys = True, N_loops = None, NumCore = 1,EBTEL_dir=None, EBTEL_ConfigFle = None, logTbinSize = 0.05, logTmax = 7.5, logTmin = 5.5):

        start_time = time.time()

        L_half = np.array(L_half)
        B_avg = np.array(B_avg)
        B_Coronal_base = np.array(B_Coronal_base)
        L_ind_no = np.array(L_ind_no)

        c = tan_theta
        Vs = V_h
        self.logTnbins = (logTmax-logTmin)//logTbinSize

        self.L_half = L_half; self.B_avg = B_avg; 
        self.logTbinSize = logTbinSize; self.logTmax = logTmax; self.logTmin = logTmin
        self.c = c ; self.Vs = Vs; self.tau_half= tau_half; 
        self.SimulationTime=SimulationTime; self.BKG_T=BKG_T; 
        self.Fp=Fp; self.B_Coronal_base=B_Coronal_base
        self.L_ind_no=L_ind_no; self.EBTEL_dir=EBTEL_dir; self.EBTEL_ConfigFle = EBTEL_ConfigFle; self.Out_phys = Out_phys
        self.q0min_fract = q0min_fract

        if EBTEL_dir is None: raise Exception("%% simar_error : Define correct EBTEL directory!")
        if EBTEL_ConfigFle is None: Exception("%% simar_error : Define correct EBTEL config file!")

        ##Mkdir for each c and vs combination
        if Fp is None: sub_dir = os.path.join(OutDir,'MultiFreq_C'+format('%0.2f'%c)+'_Vh'+format('%0.2f'%Vs))
        else: sub_dir = os.path.join(OutDir,'MultiFreq_C'+format('%0.2f'%c)+'_Fp'+format('%0.2f'%Fp))
        if os.path.isdir(sub_dir) is False: os.mkdir(sub_dir)
        self.sub_dir = sub_dir

        if Fp is not None: B_Coronal_base = [0] #If Fp is given B_Coronal_base is not needed

        pool = mul.Pool(NumCore)
        R = pool.map(self.iteration_over_loops_ebtel, range(len(L_half)))
        pool.close()
        print("END Calculation--- %s seconds ---" % (time.time() - start_time),'\n')
        return None

    def get_DEM_TimeAvg(self,L_array,Tstart,Tstop,hydrad_results,dem_log10T_low=5.8,dem_log10T_high=7.0,dem_deltalog10T=0.1):

        #Return the DEM for a given grids (L_array) for the simulation time range [Tstart,Tstop]
        #L_array --> grid of loop deight in Mm for which DEM is to be estimated
        #Tstart --> start time (in numpy unit) for which DEM is to be estimated
        #Tstop --> stop time (in numpy unit) for which DEM is to be estimated
        #dem_log10T_low --> lower grid of logT for which DEM is to be estimated
        #dem_log10T_high --> higher grid
        #dem_deltalog10T --> log10T grid spacing

        Tstart = Tstart*u.s
        Tstop = Tstop*u.s

        time_index = np.where(((hydrad_results.time >= Tstart) & (hydrad_results.time <=Tstop)))[0]
        time = hydrad_results.time[time_index]

        n = hydrad_results.to_constant_grid('electron_density',grid=L_array*u.Mm)[time_index,:].value #n[times,grids]
        T = hydrad_results.to_constant_grid('electron_temperature',grid=L_array*u.Mm)[time_index,:].value
        log10T = np.log10(T)

        dL = (shift(L_array, -1, cval=0.0) - shift(L_array, 1, cval=0.0)) * 0.5
        nlength = len(L_array)
        dL[0] = L_array[1] - L_array[0]
        dL[nlength-1] = (L_array[nlength-1]-L_array[nlength-2])

        nTbins = round((dem_log10T_high-dem_log10T_low)/dem_deltalog10T)
        DEM = np.zeros([nTbins,nlength])
        dem_T_grids = 10**np.arange(start=dem_log10T_low,stop=(dem_log10T_high-dem_deltalog10T),step=dem_deltalog10T)
        log10 = np.log(10)
        for i in range(nlength):
            T_bins = np.rint((log10T[:,i] - dem_log10T_low) / dem_deltalog10T).astype(int) #grod of T-bins in time-space
            ind_bad = np.where(np.logical_or(T_bins < 0, T_bins >= nTbins))[0]
            ind_good = np.where(((T_bins >= 0) & (T_bins < nTbins)))[0]
            n[ind_bad,i] = 0 #set the density 0 for the outside dem temperature range

            if len(ind_good) > 0:
                n2_grid = np.zeros([len(n[:,i]),nTbins])
                for j in range(len(T_bins[ind_good])):
                    n2_grid[:,T_bins[ind_good[j]]] += n[:,i]**2
                n2_grid_avg = np.average(n2_grid,axis=0,weights=np.gradient(time)) #Array[len(dem_T_grids)]
                DEM[T_bins[ind_good],i] = (n2_grid_avg[T_bins[ind_good]] / (dem_T_grids[T_bins[ind_good]]*log10)) * (dL[i]/dem_deltalog10T) #unit of 1.0e8 cm^-5

        return DEM*1.0e8 #DEM[T-bins,length] #unit (cm^-5 K^{-1})

    def DEMmap(self, Simulation_folder = '', HMI_image = '', AIA_image= '', Tstart=8000, Tstop=9000, log_min = 5.6,log_max = 7.0, logt_bins = 0.1, N_loops=300,Chromospheric_height = 0.5 ,Loop_Files_dir='./',OutDir='./',name_str = '_',Extrp_voxel_size = [0.3625,0.3625,0.3625],NumCores=0,Store_outputs = False):
        
        '''
        Purpose: Make the DEM array from the output of Hydrad for each loops of an AR

        Biswajit, 02-June-2023
                  09-Oct-2023
        '''
       
        if os.path.isdir(Simulation_folder) == False: raise Exception("%% simar_error: {Simulation_folder} does not exist: "+Simulation_folder)
        if os.path.isfile(HMI_image) == False: raise Exception("%% simar_error: {HMI_image} does not exist: "+HMI_image)
        if os.path.isfile(AIA_image) == False: raise Exception("%% simar_error: {AIA_image} does not exist: "+AIA_image)

        #Read HMI cucout image to plot LOS magnetogram and make the array dimention for loop structure
        aprep_hmi = sunpy.map.Map(HMI_image)
        aprep = sunpy.map.Map(AIA_image) 
        hmi_scale_factor = (aprep_hmi.scale.axis1*u.pix).to('arcsec',equivalencies=sunpy.coordinates.utils.solar_angle_equivalency(aprep_hmi.observer_coordinate)) / u.pix
        aia_scale_factor = (aprep.scale.axis1*u.pix).to('arcsec',equivalencies=sunpy.coordinates.utils.solar_angle_equivalency(aprep.observer_coordinate)) / u.pix
        out_shape = tuple(np.array(np.array(aprep.data.shape)*(aia_scale_factor.value / hmi_scale_factor.value)))
        aprep = aprep.resample((out_shape[1],out_shape[0]) * u.pix) #Reform into HMI pixcel size
        out_shape = aprep.data.shape
        bot_left = aprep.world_to_pixel(aprep.bottom_left_coord)
        bot_left = [int(bot_left.x.value),int(bot_left.y.value)]
        top_right = aprep.world_to_pixel(aprep.top_right_coord)
        top_right = [int(top_right.x.value),int(top_right.y.value)]
        logT = np.arange(start=log_min,stop=log_max,step=logt_bins)
        EM_map = np.zeros([out_shape[0],out_shape[1],len(logT)])#DEM map
        Non_zero_array = EM_map[:,:,0]*0.0 #Location of non-zero pixcels in EM map

        ##Read Simulation outputs:
        Hydrad_OutFiles = glob.glob(os.path.join(os.path.join(Simulation_folder),'*.pkl'))
        Hydrad_OutFiles = sorted(Hydrad_OutFiles)
        strand_no = len(Hydrad_OutFiles)
        if strand_no < N_loops:
            print('%% SunX messege: N_loops exceding the total number of loops, set N_loops with the total number of the loops')
            N_loops = strand_no
        Loop_half_length_all = []
        x_ind_all = []#index of x-pixcels where loops coordinate coincides
        y_ind_all = []

        if NumCores == 0: #No parallel run
            for i in range(N_loops):
                results = load_obj(Hydrad_OutFiles[i][0:-4])
                hydrad_res = results['results']
                           
                loop_dir = [os.path.join(Loop_Files_dir,'LoopInd_'+results['loop_index']+'_HMIcut_newOBS_XBP001_Extrapolated.dat')]
                fieldline_coords = LoopCoordinate2map(HMI_image,loop_dir,Extrp_voxel_size=Extrp_voxel_size)
                #Map the EM in AIA pixcels:
                x = fieldline_coords[0].x.value #in Mm
                y = fieldline_coords[0].y.value #in Mm
                z = fieldline_coords[0].z.value #in Mm
                x_ind = aprep.world_to_pixel(fieldline_coords[0]).x.value #in Pix
                y_ind = aprep.world_to_pixel(fieldline_coords[0]).y.value #in Pix
                #x=x[1::] ; y=y[1::] ; z=z[1::] #exclude 1st point which is same as second point
                dx = (shift(x, -1, cval=0.0) - shift(x, 1, cval=0.0)) * 0.5
                nx = len(x) ; dx[0] = x[1] - x[0] ; dx[nx-1] = (x[nx-1]-x[nx-2])
                dy = (shift(y, -1, cval=0.0) - shift(y, 1, cval=0.0)) * 0.5
                ny = len(y) ; dy[0] = y[1] - y[0] ; dy[ny-1] = (y[ny-1]-y[ny-2])
                dz = (shift(z, -1, cval=0.0) - shift(z, 1, cval=0.0)) * 0.5
                nz = len(z) ; dz[0] = z[1] - z[0] ; dz[nz-1] = (z[nz-1]-z[nz-2])
                delta_length = np.sqrt(dx**2 + dy**2 + dz**2)

                L_array = [] #Length grids for which we need to calculate DEM
                dll = 0
                for ll in range(len(delta_length)):
                    dll += delta_length[ll]
                    L_array += [dll]
                L_array = np.array(L_array)
                ind_coronal = np.where(((L_array > Chromospheric_height) & (L_array < ((hydrad_res.loop_length.value/1.0e8)-Chromospheric_height))))[0] #Consider only coronal part of the loop
                L_array = L_array[ind_coronal]
                x_ind = x_ind[ind_coronal] ; y_ind = y_ind[ind_coronal] 
                ind__ = np.intersect1d(np.where((x_ind>=bot_left[0]) & (x_ind <= top_right[0])), np.where((y_ind >= bot_left[1]) & (y_ind <= top_right[1]))) #Remove the points outside AIA FOV
                x_ind = np.round(x_ind[ind__]).astype('int'); y_ind=np.round(y_ind[ind__]).astype('int')
                L_array = L_array[ind__]
                #For XBP-1 paper, the DEM is averaged for t = 8000 to 9000
                DEM_i = self.get_DEM_TimeAvg(L_array,Tstart,Tstop,hydrad_res,dem_log10T_low=log_min,dem_log10T_high=log_max+logt_bins,dem_deltalog10T=logt_bins) ###DEM[T-bins,length] 
                ##Fill the DEM_map
                for jj in range(len(x_ind)):
                    EM_map[y_ind[jj],x_ind[jj],:] += DEM_i[:,jj]
                    Non_zero_array[y_ind[jj],x_ind[jj]] += sum(DEM_i[:,jj]) #location of the loop grids in the DEM array

                loop_half_length = hydrad_res.loop_length.value/2.0e8 #Loop loop half-length in Mm
                Loop_half_length_all += [loop_half_length] #Loop loop half-length in Mm
        else:
            print('%%Error Parallel processing is not implemented yet.')
            sys.exit()
        '''
            def loop_over_length(i):
                global Hydrad_OutFiles, Loop_Files_dir, HMI_image, loop_dir, Extrp_voxel_size, Tstart, Tstop, log_min, log_max, logt_bins, logt_bins, EM_map, Non_zero_array, Loop_half_length_all

                results = load_obj(Hydrad_OutFiles[i][0:-4])
                hydrad_res = results['results']

                loop_dir = [os.path.join(Loop_Files_dir,'LoopInd_'+results['loop_index']+'_HMIcut_newOBS_XBP001_Extrapolated.dat')]
                fieldline_coords = LoopCoordinate2map(HMI_image,loop_dir,Extrp_voxel_size=Extrp_voxel_size)
                #Map the EM in AIA pixcels:
                x = fieldline_coords[0].x.value #in Mm
                y = fieldline_coords[0].y.value #in Mm
                z = fieldline_coords[0].z.value #in Mm
                x_ind = aprep.world_to_pixel(fieldline_coords[0]).x.value #in Pix
                y_ind = aprep.world_to_pixel(fieldline_coords[0]).y.value #in Pix
                #x=x[1::] ; y=y[1::] ; z=z[1::] #exclude 1st point which is same as second point
                dx = (shift(x, -1, cval=0.0) - shift(x, 1, cval=0.0)) * 0.5
                nx = len(x) ; dx[0] = x[1] - x[0] ; dx[nx-1] = (x[nx-1]-x[nx-2])
                dy = (shift(y, -1, cval=0.0) - shift(y, 1, cval=0.0)) * 0.5
                ny = len(y) ; dy[0] = y[1] - y[0] ; dy[ny-1] = (y[ny-1]-y[ny-2])
                dz = (shift(z, -1, cval=0.0) - shift(z, 1, cval=0.0)) * 0.5
                nz = len(z) ; dz[0] = z[1] - z[0] ; dz[nz-1] = (z[nz-1]-z[nz-2])
                delta_length = np.sqrt(dx**2 + dy**2 + dz**2)

                L_array = [] #Length grids for which we need to calculate DEM
                dll = 0
                for ll in range(len(delta_length)):
                    dll += delta_length[ll]
                    L_array += [dll]
                L_array = np.array(L_array)
                ind_coronal = np.where(((L_array > Chromospheric_height) & (L_array < ((hydrad_res.loop_length.value/1.0e8)-Chromospheric_height))))[0] #Consider only coronal part of the loop
                L_array = L_array[ind_coronal]
                x_ind = x_ind[ind_coronal] ; y_ind = y_ind[ind_coronal]
                ind__ = np.intersect1d(np.where((x_ind>=bot_left[0]) & (x_ind <= top_right[0])), np.where((y_ind >= bot_left[1]) & (y_ind <= top_right[1]))) #Remove the points outside AIA FOV
                x_ind = np.round(x_ind[ind__]).astype('int'); y_ind=np.round(y_ind[ind__]).astype('int')
                L_array = L_array[ind__]
                #For XBP-1 paper, the DEM is averaged for t = 8000 to 9000
                DEM_i = self.get_DEM_TimeAvg(L_array,Tstart,Tstop,hydrad_res,dem_log10T_low=log_min,dem_log10T_high=log_max+logt_bins,dem_deltalog10T=logt_bins) ###DEM[T-bins,length] 
                ##Fill the DEM_map
                for jj in range(len(x_ind)):
                    EM_map[y_ind[jj],x_ind[jj],:] += DEM_i[:,jj]
                    Non_zero_array[y_ind[jj],x_ind[jj]] += sum(DEM_i[:,jj]) #location of the loop grids in the DEM array

                loop_half_length = hydrad_res.loop_length.value/2.0e8 #Loop loop half-length in Mm
                Loop_half_length_all += [loop_half_length] #Loop loop half-length in Mm
                return None
            #if __name__ == '__main__': 
            pool = mul.Pool(NumCores)
            R = pool.map(loop_over_length, range(N_loops))
            pool.close()
        '''
        #save the EM map
        EM_map[np.isnan(EM_map[:,:])] = 0
        Non_zero_pixcels = np.where(Non_zero_array > 0)#indices of non-zero pixcels
        Model_data = {'Simulated DEM-map': 'Values'}
        Model_data['Loop_half_length_all'] = Loop_half_length_all
        Model_data['DEM_Map'] = EM_map #DEM map [x,y,logT] 
        Model_data['Non_zero_pixcels'] = Non_zero_pixcels #location of non-zero pixcels in EM_map 
        Model_data['logT'] = logT
        Model_data['WCS_map'] = aprep
        name = 'DEM_Map_HMIres_'+format('%d'%N_loops)+'_'+name_str#'Twait_'+Tw

        if Store_outputs is True: save_obj(Model_data, os.path.join(OutDir,name))
        return EM_map

    def DEMmap_ebtel(self, Simulation_folder, HMI_image, ExtPo_loop_par_file, Tstart=8000, Tstop=9000,dT = 10, N_loops=300,Chromospheric_height = 3, OutDir='./',name_str = 'ebtel',Extrp_voxel_size = [0.3625,0.3625,0.3625],DEM_type = 'tot'):

        #Read HMI cucout image to plot LOS magnetogram and make the array dimention for loop structure

        aprep_hmi = sunpy.map.Map(HMI_image)
        hmi_image = aprep_hmi.data
        hmi_image_shape = hmi_image.shape
        #EM_map = np.zeros([hmi_image_shape[0],hmi_image_shape[1],len(logT)])#DEM map
        Non_zero_array = np.zeros([hmi_image_shape[0],hmi_image_shape[1]])  #Location of non-zero pixcels in EM map
        
        ##Read EBTEL outputs:
        EBTEL_OutFiles = glob.glob(os.path.join(Simulation_folder,'*.pkl'))
        EBTEL_OutFiles = sorted(EBTEL_OutFiles)
        strand_no = len(EBTEL_OutFiles)
        if N_loops is None: N_loops = strand_no
        elif strand_no < N_loops:
            print('%% N_loops exceding the total number of loops, set N_loops with the total number of the loops')
            N_loops = strand_no

        Time_grid = np.arange(Tstart,Tstop,dT) 
        n_Time_grid = len(Time_grid)
        dem_temperature = 0
        coronal_emission_test=0
        Net_Temp = 0
        Net_Density = 0
        Net_Heat = 0
        Loop_half_length_all = []
        Em_weighted_T = np.zeros(n_Time_grid)
        Avg_coronalEM = np.zeros(n_Time_grid)
        Density_weighted_T = 0

        x_ind_all = []#index of x-pixcels where loops coordinate coincides
        y_ind_all = []
        hdul = fits.open(ExtPo_loop_par_file)
        for i in range(N_loops):
            results = load_obj(EBTEL_OutFiles[i][0:-4])
            logT = results['dem_temperature']
            if i == 0:
                EM_map = np.zeros([hmi_image_shape[0],hmi_image_shape[1],len(logT)])#DEM map
                dem_total =  np.zeros([len(logT),strand_no])
                tmp_dem_corona = np.zeros((n_Time_grid,len(logT)))
                #dem_tr = np.zeros([n_Time_grid,len(logT)])#0
                #dem_corona = np.zeros([n_Time_grid,len(logT)])#0   

            ##Read the loop indices in x,y plane:
            x = Extrp_voxel_size[0]*hdul['L'+format('%d'%results['loop_index'])].data['y'] #X and Y are inherently interchanged. X, Y are in pixcel index. Thus multiplied by pixel length of 0.3625 Mm for HMI.
            y = Extrp_voxel_size[1]*hdul['L'+format('%d'%results['loop_index'])].data['x']
            z = Extrp_voxel_size[2]*hdul['L'+format('%d'%results['loop_index'])].data['z']

            dx = (shift(x, -1, cval=0.0) - shift(x, 1, cval=0.0)) * 0.5
            nx = len(x)
            dx[0] = x[1] - x[0]
            dx[nx-1] = (x[nx-1]-x[nx-2])

            dy = (shift(y, -1, cval=0.0) - shift(y, 1, cval=0.0)) * 0.5
            ny = len(y)
            dy[0] = y[1] - y[0]
            dy[ny-1] = (y[ny-1]-y[ny-2])

            dz = (shift(z, -1, cval=0.0) - shift(z, 1, cval=0.0)) * 0.5
            nz = len(z)
            dz[0] = z[1] - z[0]
            dz[nz-1] = (z[nz-1]-z[nz-2])

            delta_length = np.sqrt(dx**2 + dy**2 + dz**2)
            L_array = [] #Length grids for which we need to calculate DEM
            dll = 0
            for ll in range(len(delta_length)):
                dll += delta_length[ll]
                L_array += [dll]
            L_array = np.array(L_array)

            ind_coronal = np.where(((L_array > Chromospheric_height) & (L_array < ((2*results['Loop_half_length'])-Chromospheric_height))))[0] #Consider only coronal part of the loop
            L_array = L_array[ind_coronal]
            x = x[ind_coronal]
            y = y[ind_coronal]
            z = z[ind_coronal]


            #L_array = np.array(L_array)
            #ind_coronal = np.where(((L_array > Chromospheric_height) & (L_array < ((hydrad_res.loop_length.value/1.0e8)-Chromospheric_height))))[0] #Consider only coronal part of the loop
            #L_array = L_array[ind_coronal]
            #x_ind = x_ind[ind_coronal] ; y_ind = y_ind[ind_coronal]
            #ind__ = np.intersect1d(np.where((x_ind>=bot_left[0]) & (x_ind <= top_right[0])), np.where((y_ind >= bot_left[1]) & (y_ind <= top_right[1]))) #Remove the points outside AIA FOV
            #x_ind = np.round(x_ind[ind__]).astype('int'); y_ind=np.round(y_ind[ind__]).astype('int')
            #L_array = L_array[ind__]


            x_ind = hdul['L'+format('%d'%results['loop_index'])].data['xp'][ind_coronal] #Projected coordinate associated with x
            y_ind = hdul['L'+format('%d'%results['loop_index'])].data['yp'][ind_coronal]

            #Remove the portion of the loops outside the final HMI cutout
            ind1 = np.where((x_ind >= 0) & ((x_ind < EM_map.shape[1])))
            ind2 = np.where((y_ind >= 0) & ((y_ind < EM_map.shape[0])))
            ind = np.intersect1d(ind1,ind2)
            x_ind = x_ind[ind]
            y_ind = y_ind[ind]
            L_array = L_array[ind]

            x_ind_all += [x_ind]
            y_ind_all += [y_ind]

            dem_temperature = results['dem_temperature']
            #dem_total = dem_total + results['dem_total'] #Array[time,temperature]
            #dem_tr = dem_tr + results['dem_tr'] 
            #dem_corona = dem_corona + results['dem_corona']
            dem_total1 = np.zeros([n_Time_grid,len(logT)])
            for tem_dem_ind in range(len(logT)):
                '''
                DEM_interpol_func_tr = interpolate.interp1d(results['time'], results['dem_tr'][:,tem_dem_ind],kind='linear') 
                DEM_interpol_func_corona = interpolate.interp1d(results['time'], results['dem_corona'][:,tem_dem_ind],kind='linear')
                
                DEM_interpol_tr = DEM_interpol_func_tr(Time_grid)
                DEM_interpol_corona = DEM_interpol_func_corona(Time_grid)
                dem_tr[:,tem_dem_ind] += DEM_interpol_tr
                dem_corona[:,tem_dem_ind] += DEM_interpol_corona
                '''
                dem_cor_tr = results['dem_corona']+(results['dem_tr']/1)
                if DEM_type == 'tot':DEM_interpol_func_total = interpolate.interp1d(results['time'], results['dem_total'][:,tem_dem_ind],kind='linear')
                elif DEM_type == 'cor':DEM_interpol_func_total = interpolate.interp1d(results['time'], results['dem_corona'][:,tem_dem_ind],kind='linear')
                elif DEM_type == 'TR':DEM_interpol_func_total = interpolate.interp1d(results['time'], results['dem_tr'][:,tem_dem_ind],kind='linear')
                DEM_interpol_total = DEM_interpol_func_total(Time_grid)
                dem_total1[:,tem_dem_ind] += DEM_interpol_total
            
            dem_total[:,i] = np.average(dem_total1,axis=0,weights=np.gradient(Time_grid))
            
            ##Fill the DEM_map
            for jj in range(len(x_ind)):
                EM_map[y_ind[jj],x_ind[jj],:] += dem_total[:,i]
                Non_zero_array[y_ind[jj],x_ind[jj]] += sum(dem_total[:,i]) #location of the loop grids in the DEM array
            loop_half_length = results['Loop_half_length']#results['L_half'] #Loop loop half-length in Mm
            Loop_half_length_all += [loop_half_length] #Loop loop half-length in Mm

        #save the EM map
        EM_map[np.isnan(EM_map)] = 0
        Non_zero_pixcels = np.where(Non_zero_array > 0)#indices of non-zero pixcels

        Model_data = {'foo': 'bar'}
        Model_data['Loop_half_length_all'] = Loop_half_length_all
        Model_data['DEM_Map'] = EM_map #DEM map [x,y,logT] 
        Model_data['Non_zero_pixcels'] = Non_zero_pixcels #location of non-zero pixcels in EM_map 
        Model_data['logT'] = np.log10(logT)
        name = 'DEM_Map_HMIres_'+format('%d'%N_loops)+'_'+name_str#'Twait_'+Tw
        save_obj(Model_data, os.path.join(OutDir,name))
        return EM_map


    def demMap2Image(self,tresp,tresp_logT,DEM_logT,DEM_Map,Non_zero_pixcels):
        '''
        inputs:
            tresp -> 1-D response array 
            tresp_logT -> response logT grids
            DEM_Map -> 3D array, dimension = [x,y,logT]
            DEM_logT -> DEM logT grids
            Non_zero_pixcels -> 2-column array, representing the indices of non-zero pixcels to which the counts is to be 
                              calculated, to reduce the computation time.
        outputs: Model_image -> model image in the resolution of DEM_Map
        '''
        DEM_logT = DEM_logT[1::]
        DEM_Map = DEM_Map[:,:,1::]
        tresp_interpolation = interpolate.interp1d(tresp_logT , tresp)
        G_T = tresp_interpolation(DEM_logT)
        dT = (shift(DEM_logT, -1, cval=0.0) - shift(DEM_logT, 1, cval=0.0)) * 0.5
        ntemps = len(DEM_logT)
        dT[0] = DEM_logT[1] - DEM_logT[0]
        dT[ntemps-1] = (DEM_logT[ntemps-1]-DEM_logT[ntemps-2])
    
        #indices of non-zero pixcels
        ind_xx = Non_zero_pixcels[0][:]
        ind_yy = Non_zero_pixcels[1][:]
    
        Model_image = DEM_Map[:,:,0]*0
        for i in range(len(ind_xx)):
            dem = DEM_Map[ind_xx[i],ind_yy[i],:]
            Model_image[ind_xx[i],ind_yy[i]] += sum(G_T * dem * (10**DEM_logT) *np.log(10.) * dT)
        return Model_image

    def DemMap2EMmap(self,DEM_logT,DEM_Map,Non_zero_pixcels):
        '''
        inputs:
            DEM_Map -> 3D array, dimension = [x,y,logT]
            DEM_logT -> DEM logT grids
            Non_zero_pixcels -> 2-column array, representing the indices of non-zero pixcels to which the counts is to be 
                              calculated, to reduce the computation time.
        outputs: Eff_Tmap --> EM-waighted temperature
        '''
        DEM_logT = DEM_logT[0::]
        DEM_Map = DEM_Map[:,:,0::]
        dT = (shift(DEM_logT, -1, cval=0.0) - shift(DEM_logT, 1, cval=0.0)) * 0.5
        ntemps = len(DEM_logT)
        dT[0] = DEM_logT[1] - DEM_logT[0]
        dT[ntemps-1] = (DEM_logT[ntemps-1]-DEM_logT[ntemps-2])

        #indices of non-zero pixcels
        ind_xx = Non_zero_pixcels[0][:]
        ind_yy = Non_zero_pixcels[1][:]

        Model_EM = DEM_Map[:,:,:]*0
        for i in range(len(ind_xx)):
            for j in range(len(DEM_logT)):
                dem = DEM_Map[ind_xx[i],ind_yy[i],j]
                Model_EM[ind_xx[i],ind_yy[i],j] += (dem * (10**DEM_logT[j]) *np.log(10.) * dT[j])

        return Model_EM


    def DEM2EffTemp(self,DEM_logT,DEM_Map,Non_zero_pixcels):
        '''
        Purpose: get the emission measure weighted temperature map.
        '''
        #DEM_logT = DEM_logT[1::]
        #DEM_Map = DEM_Map[:,:,1::]
        dT = (shift(DEM_logT, -1, cval=0.0) - shift(DEM_logT, 1, cval=0.0)) * 0.5
        ntemps = len(DEM_logT)
        dT[0] = DEM_logT[1] - DEM_logT[0]
        dT[ntemps-1] = (DEM_logT[ntemps-1]-DEM_logT[ntemps-2])
        #indices of non-zero pixcels
        ind_xx = Non_zero_pixcels[0][:]
        ind_yy = Non_zero_pixcels[1][:]
        Model_EM = self.DemMap2EMmap(DEM_logT,DEM_Map,Non_zero_pixcels)
        EffT = Model_EM[:,:,0]*0
        for i in range(len(ind_xx)):
            EM_i = Model_EM[ind_xx[i],ind_yy[i],:]
            EffT[ind_xx[i],ind_yy[i]] = sum(EM_i*(10**DEM_logT)) / sum(EM_i)
        return EffT
    
    def classify_heating_ferq_ebtel(self,EBTEL_results_dir,Tstart = None, Tstop=None, StoreOutputs = False, OutDir = None, OutFileName = None, N_loops=None, min_dT = 0.5, BKG_T = 0.5):
        '''
         Purpose: Classify the events for each loops from EBTEL average temperature profile and store outputs in a fits file.
             EBTEL_results_dir -> directory of the EBTEL outputs for all the loops
             Tstart -> Start time of the simulation to be considered.
             Tstop -> Stop time of the simulation outputs to be considered
             N_loops -> Number of loops to be considered from 'LoopParFile'. If none all the loops will be considered.
             min_dT -> minimum temperature (MK) difference between temperature peek and dip, above which a event would be considered.
             BKG_T -> Loop background temperature. If after an event no event will be there then loop will sattle with this temperature
        '''
        #L_half = Full_length_Mm/2

        EBTEL_OutFiles = glob.glob(os.path.join(EBTEL_results_dir,'*.pkl'))
        EBTEL_OutFiles = sorted(EBTEL_OutFiles)

        if N_loops > len(EBTEL_OutFiles) :
            N_loops = len(EBTEL_OutFiles)
            print('%SunX_massege : Exceeding maximum available loops. Set to maximum number of loops ->'+format('%d'%N_loops))
        if Tstart is None: Tstart = 0

        min_dT = min_dT*1.0e6
        BKG_T = BKG_T * 1.0e6
        HF_Avg = np.zeros([3,N_loops]) #['energy_rate','delay_time']
        LF_Avg = np.zeros([3,N_loops])
        IF_Avg = np.zeros([3,N_loops])
        loop_INDX = []
        L_halfs = []
        for i in range(N_loops):
            results = load_obj(EBTEL_OutFiles[i][0:-4])
            loop_INDX += [results['loop_index']]
            L_halfs += [results['Loop_half_length']]  
            time = results['time']
            if Tstop is None: Tstop = time[-1]
            ind = np.where((time>Tstart)&(time<Tstop))
            ind2 = np.where((results['peak_heating_time']>Tstart)&(results['peak_heating_time']<Tstop))
            time = time[ind]
            peak_heating_time = results['peak_heating_time'][ind2]
            peak_heating_rate = results['peak_heating_rate'][ind2]
        
            Avg_temp = results['electron_temperature'][ind] #in K
            Avg_density = results['density'][ind] #cm^-3
        
            ind = np.where((Avg_temp != np.nan)&(Avg_temp>0))
            Avg_temp = Avg_temp[ind]
            Avg_density = Avg_density[ind]
            time = time[ind]
        
            if len(Avg_temp) > 1:
                
                #Find the temperature peaks:
                peaks_ind, _ = find_peaks(Avg_temp, distance=1)  # Adjust distance as needed
                peak_T = Avg_temp[peaks_ind]
                peak_n = Avg_density[peaks_ind]
                peak_times = time[peaks_ind]

                good_peak_ind = np.where(peak_T > BKG_T) #Consider all the peaks which are above BKG T
                peak_T = peak_T[good_peak_ind]
                peak_n = peak_n[good_peak_ind]
                peak_times = peak_times[good_peak_ind]
                peaks_ind = peaks_ind[good_peak_ind]

                #Consider the loops having atleast two events:
                if len(peak_T) > 1:
                    #Find the temperature dips :
                    dip_times = [] # Finding dip times of the temperature minimum after each temperature peaks
                    dip_T = []
                    for ii in range(len(peaks_ind) - 1):
                        start_idx = peaks_ind[ii]
                        end_idx = peaks_ind[ii + 1]
                        valley_idx = np.argmin(Avg_temp[start_idx:end_idx]) + start_idx
                        dip_times.append(time[valley_idx])
                        dip_T.append(Avg_temp[valley_idx])

                    dip_times = np.array(dip_times)
                    dip_T = np.array(dip_T)
                    delta_T = peak_T[0:-1] - dip_T
                    if len(peak_heating_rate) < len(dip_T)+1 :
                        ind_good = np.where(delta_T > min_dT)
                        dip_T = dip_T[ind_good]
                        dip_times = dip_times[ind_good]
                        peak_times = np.concatenate((peak_times[ind_good],[peak_times[-1]])) #Remove some discrepency in the peek estimations
                        peak_T = np.concatenate((peak_T[ind_good],[peak_T[-1]]))
                   
                    LF_ind = np.where(dip_T <= BKG_T)
                    LF_ = list(dip_T[LF_ind])
                    #print(EBTEL_OutFiles[i][0:-4],results['loop_index'],peak_heating_rate,LF_ind)
                    peak_heating_rate_LF_ = peak_heating_rate[LF_ind]
                    HF_ind = np.where(dip_T > BKG_T)
                    if len(HF_ind[0]) >0: 
                        HF = peak_heating_rate[HF_ind]; 
                        HF_dt = peak_times[tuple(np.array(HF_ind)+1)] - peak_times[HF_ind]
                        HF_Avg[0,i] = len(HF); HF_Avg[1,i] = np.average(HF) ; HF_Avg[2,i] = np.average(HF_dt)
                    if len(LF_) > 0:
                        t_cool__ = dip_times[LF_ind] - peak_times[LF_ind] #cooling time
                        dt_peek = peak_times[tuple(np.array(LF_ind)+1)] - peak_times[LF_ind] #Repetation time
                        LF = [] ; IF = [] ; IF_dt = []; LF_dt = []
                        for kk in range(len(t_cool__)):
                            if dt_peek[kk] > 2*t_cool__[kk]: 
                                LF += [peak_heating_rate_LF_[kk]]
                                LF_dt += [dt_peek[kk]]
                            else: 
                                IF += [peak_heating_rate_LF_[kk]]
                                IF_dt += [dt_peek[kk]]
                        if len(LF) > 0 :LF_Avg[0,i] = len(LF); LF_Avg[1,i] = np.average(np.array(LF)) ; LF_Avg[2,i] = np.average(LF_dt)
                        if len(IF) > 0 :IF_Avg[0,i] = len(IF);IF_Avg[1,i] = np.average(IF) ; IF_Avg[2,i] = np.average(IF_dt)
                    '''
                    print('LF: ',LF_Avg[0,i],LF_Avg[1,i],LF_Avg[2,i])
                    print('IF: ',IF_Avg[0,i],IF_Avg[1,i],IF_Avg[2,i])
                    print('HF: ',HF_Avg[0,i],HF_Avg[1,i],HF_Avg[2,i])
                    print('\n')
                    '''
                    ''' 
                    #plt.plot(time,Avg_density)
                    ##plt.plot(peak_dens_times, peak_n, "x", label='Peaks')
                    #ax = plt.twinx()
                    plt.plot(time,Avg_temp,color='r')
                    plt.plot([time[0],time[-1]],[BKG_T]*2)
                    plt.plot(peak_times, peak_T, "x", label='Peaks',color='r')
                    plt.plot(dip_times,dip_T,'*b')
                    ## Plotting dip times
                    ##plt.plot(dip_times, [Avg_temp[np.argmin(Avg_temp[start_idx:end_idx]) + start_idx] for start_idx, _ in zip(peaks_ind[:-1], peaks_ind[1:])], "o", label='Dips')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Avg. T (K)')
                    plt.show()
                    '''
        if StoreOutputs is True:
            if OutDir is None: print('%% SunX error: Define a valid Output directory')
            if OutFileName is None: OutFileName = 'test_event_classifications'
            results = {}
            results['loop_INDX'] = loop_INDX
            results['L_half'] = L_halfs 
            results['HF'] ={}
            results['HF']['numbers'] = HF_Avg[0,:]
            results['HF']['avg_energy'] = HF_Avg[1,:]
            results['HF']['delay_time'] = HF_Avg[2,:]
            results['HF']['units'] = 'avg_energy: ergs/cm3/s, delay_time: seconds'
            results['LF'] ={}
            results['LF']['numbers'] =    LF_Avg[0,:]
            results['LF']['avg_energy'] = LF_Avg[1,:]
            results['LF']['delay_time'] = LF_Avg[2,:]
            results['LF']['units'] = 'avg_energy: ergs/cm3/s, delay_time: seconds'
            results['IF'] ={}
            results['IF']['numbers'] =    IF_Avg[0,:]
            results['IF']['avg_energy'] = IF_Avg[1,:]
            results['IF']['delay_time'] = IF_Avg[2,:]
            results['IF']['units'] = 'avg_energy: ergs/cm3/s, delay_time: seconds'
            save_obj(results,os.path.join(OutDir,OutFileName))
        print(i)
        return loop_INDX, L_halfs, HF_Avg,IF_Avg,LF_Avg
    
    def classify_heating_ferq_ebtel_method2(self,EBTEL_results_dir,Tstart = None, Tstop=None, StoreOutputs = False, OutDir = None, OutFileName = None, N_loops=None, dt_half = 50,dt_tol = 25): #min_dT = 0.5, BKG_T = 0.5):
        '''
         **This function is used in 2024 paper
         Purpose: Classify the events for each loops from EBTEL average temperature profile and store outputs in a fits file.
             EBTEL_results_dir -> directory of the EBTEL outputs for all the loops
             Tstart -> Start time of the simulation to be considered.
             Tstop -> Stop time of the simulation outputs to be considered
             N_loops -> Number of loops to be considered from 'LoopParFile'. If none all the loops will be considered.
             dt_half -> half duration of input heating rate profile.
             dt_tol -> T_peak is alwayse after the peah of heating rate. This is the additional time from the peak-heating-time to estimate the peak temperature.
          
             ###min_dT -> minimum temperature (MK) difference between temperature peek and dip, above which a event would be considered.
             ###BKG_T -> Loop background temperature. If after an event no event will be there then loop will sattle with this temperature
        '''
        #L_half = Full_length_Mm/2

        EBTEL_OutFiles = glob.glob(os.path.join(EBTEL_results_dir,'*.pkl'))
        EBTEL_OutFiles = sorted(EBTEL_OutFiles)
        if N_loops > len(EBTEL_OutFiles) :
            N_loops = len(EBTEL_OutFiles)
            print('%SunX_massege : Exceeding maximum available loops. Set to maximum number of loops ->'+format('%d'%N_loops))
        if Tstart is None: Tstart = 0

        #min_dT = min_dT*1.0e6
        #BKG_T = BKG_T * 1.0e6
        HF_Avg = np.zeros([4,N_loops]) #['energy_rate','delay_time']
        LF_Avg = np.zeros([4,N_loops])
        IF_Avg = np.zeros([4,N_loops])
        loop_INDX = []
        L_halfs = []
        Eff_simulation_time = []
        for i in range(N_loops):
            results = load_obj(EBTEL_OutFiles[i][0:-4])
            loop_INDX += [results['loop_index']]
            L_half_ = results['Loop_half_length']*1.0e8
            L_halfs += [results['Loop_half_length']]
            time = results['time']
            if Tstop is None: Tstop = time[-1]
            Actual_simulation_time = Tstop-Tstart
            ind = np.where((time>Tstart)&(time<Tstop))
            ind2 = np.where((results['peak_heating_time']>Tstart)&(results['peak_heating_time']<Tstop))
            time = time[ind]
            peak_heating_time = results['peak_heating_time'][ind2]
            peak_heating_rate = results['peak_heating_rate'][ind2]

            Avg_temp = results['electron_temperature'][ind] #in K
            Avg_density = results['density'][ind] #cm^-3
            
            ind = np.where((Avg_temp != np.nan)&(Avg_temp>0))
            Avg_temp = Avg_temp[ind]
            Avg_density = Avg_density[ind]
            time = time[ind]

            if len(Avg_temp) > 1:
                Avg_temp_intpFunc = interpolate.interp1d(time , Avg_temp)
                '''
                #Find the temperature peaks:
                peaks_ind, _ = find_peaks(Avg_temp, distance=1)  # Adjust distance as needed
                peak_T = Avg_temp[peaks_ind]
                peak_n = Avg_density[peaks_ind]
                peak_times = time[peaks_ind]

                good_peak_ind = np.where(peak_T > BKG_T) #Consider all the peaks which are above BKG T
                peak_T = peak_T[good_peak_ind]
                peak_n = peak_n[good_peak_ind]
                peak_times = peak_times[good_peak_ind]
                peaks_ind = peaks_ind[good_peak_ind]

                #Consider the loops having atleast two events:
                if len(peak_T) > 1:
                
                    #Find the temperature dips :
                    dip_times = [] # Finding dip times of the temperature minimum after each temperature peaks
                    dip_T = []
                    time_cool = []
                    for ii in range(len(peaks_ind) - 1):
                        start_idx = peaks_ind[ii]
                        end_idx = peaks_ind[ii + 1]
                        valley_idx = np.argmin(Avg_temp[start_idx:end_idx]) + start_idx
                        dip_times.append(time[valley_idx])
                        dip_T.append(Avg_temp[valley_idx])
                        
                    dip_times = np.array(dip_times)
                    dip_T = np.array(dip_T)
                    delta_T = peak_T[0:-1] - dip_T
                    
                    if len(peak_heating_rate) < len(dip_T)+1 :
                        ind_good = np.where(delta_T > min_dT)
                        dip_T = dip_T[ind_good]
                        dip_times = dip_times[ind_good]
                        peak_times = np.concatenate((peak_times[ind_good],[peak_times[-1]])) #Remove some discrepency in the peek estimations
                        peak_T = np.concatenate((peak_T[ind_good],[peak_T[-1]]))
                    
                    plt.plot(time,Avg_temp,color='r')
                    plt.plot([time[0],time[-1]],[BKG_T]*2)
                    plt.plot(peak_times, peak_T, "x", label='Peaks',color='r')
                    plt.plot(dip_times,dip_T,'*b')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Avg. T (K)')
                    #Find the cooling time of the loops by fitting the temperature decay with an exponential function
                     
                    for iii in range(len(dip_times)):
                        
                        ind_ = np.where((time >= peak_times[iii]) & (time < dip_times[iii]))
                        x = time[ind_] 
                        y = Avg_temp[ind_]
                        x_data_norm = (x - x.min()) / (x.max() - x.min()) #Normalize x data to avoid overflow issues
                        params, covariance = curve_fit(exponential_func, x_data_norm, y)
                        #params, covariance = curve_fit(lambda x, b: exponential_func(x, b, peak_T[iii]), x, y)
                        b,a = params
                        y_fitted = exponential_func(x_data_norm, b, a)
                        plt.plot(x,y_fitted,'m-')
                        t_cool = ((x.max() - x.min())/b) + x.min()
                        print(peak_T[iii]/1.0e6,a/1.0e6,t_cool)
                '''
                #plt.plot(time,Avg_temp,color='r')
                #plt.xlabel('Time (s)')
                #plt.ylabel('Avg. T (K)')
                #if len(peak_times) == len(peak_heating_time): 
                HF = []; HF_dt = []; LF = [] ; IF = [] ; IF_dt = []; LF_dt = []
                Simulation_time = 0
                for iii in range(len(peak_heating_time)-1):
                    time_rep = peak_heating_time[iii+1]-dt_half
                    if time_rep < time[0]: time_rep = peak_heating_time[iii+1]
                    time_max = peak_heating_time[iii]+dt_tol
                    if time_max > time[-1]: time_max = peak_heating_time[iii]
                    dt = peak_heating_time[iii+1] - peak_heating_time[iii]
                    T_rep =  Avg_temp_intpFunc(time_rep)#temperature @ time of next event start
                    T_max = Avg_temp_intpFunc(time_max)
                    Simulation_time += np.sum(dt)
                    if T_rep > 0.61*T_max : 
                        HF += [peak_heating_rate[iii]]; HF_dt += [dt]
                        #plt.plot(time_max,T_max,'o',color='m') #HF
                    elif T_rep < 0.14*T_max : 
                        LF += [peak_heating_rate[iii]]; LF_dt += [dt]
                        #plt.plot(time_max,T_max,'o',color='g') #LF
                    else:
                        IF += [peak_heating_rate[iii]]; IF_dt += [dt]  
                        #plt.plot(time_max,T_max,'o',color='k') #IF
                    #plt.plot(time_rep,T_rep,'^b')
                    #elif T_rep <= 0.61*peak_T[iii] and T_rep >= 0.14*peak_T[iii]: plt.plot(peak_times[iii],peak_T[iii],'o',color='k') #IF
                #if len(HF) > 0 :HF_Avg[0,i] = len(HF); HF_Avg[1,i] = np.average(np.array(HF)) ; HF_Avg[2,i] = np.average(HF_dt)
                #if len(LF) > 0 :LF_Avg[0,i] = len(LF); LF_Avg[1,i] = np.average(np.array(LF)) ; LF_Avg[2,i] = np.average(LF_dt)
                #if len(IF) > 0 :IF_Avg[0,i] = len(IF);IF_Avg[1,i] = np.average(IF) ; IF_Avg[2,i] = np.average(IF_dt)
                Eff_simulation_time += [Simulation_time]
                if len(HF) > 0 :HF_Avg[0,i] = len(HF); HF_Avg[1,i] = np.sum(np.array(HF))*dt_half ; HF_Avg[2,i] = np.average(HF_dt) ;HF_Avg[3,i] = np.sum(HF_dt)
                if len(LF) > 0 :LF_Avg[0,i] = len(LF); LF_Avg[1,i] = np.sum(np.array(LF))*dt_half ; LF_Avg[2,i] = np.average(LF_dt) ;LF_Avg[3,i] = np.sum(LF_dt)
                if len(IF) > 0 :IF_Avg[0,i] = len(IF);IF_Avg[1,i] = np.sum(np.array(IF))*dt_half ; IF_Avg[2,i] = np.average(IF_dt) ;IF_Avg[3,i] = np.sum(IF_dt)
                #else: print('Skiping loop No:',loop_INDX[i],len(peak_times),len(peak_heating_time))
                #plt.show()
                #print('LF: ',LF_Avg[0,i],LF_Avg[1,i],LF_Avg[2,i])
                #print('IF: ',IF_Avg[0,i],IF_Avg[1,i],IF_Avg[2,i])
                #print('HF: ',HF_Avg[0,i],HF_Avg[1,i],HF_Avg[2,i])
                #print('\n')
        if StoreOutputs is True:
            if OutDir is None: print('%% SunX error: Define a valid Output directory')
            if OutFileName is None: OutFileName = 'test_event_classifications'
            results = {}
            results['loop_INDX'] = loop_INDX
            results['L_half'] = L_halfs
            results['eff_simulation_time'] = Eff_simulation_time
            results['actual_simulation_time'] = Actual_simulation_time
            results['HF'] ={}
            results['HF']['numbers'] = HF_Avg[0,:]
            results['HF']['avg_energy'] = HF_Avg[1,:]
            results['HF']['delay_time'] = HF_Avg[2,:]
            results['HF']['fract_time'] = HF_Avg[3,:]
            results['HF']['units'] = 'avg_energy: ergs/cm3, delay_time: seconds'
            results['LF'] ={}
            results['LF']['numbers'] =    LF_Avg[0,:]
            results['LF']['avg_energy'] = LF_Avg[1,:]
            results['LF']['delay_time'] = LF_Avg[2,:]
            results['LF']['fract_time'] = LF_Avg[3,:]
            results['LF']['units'] = 'avg_energy: ergs/cm3, delay_time: seconds'
            results['IF'] ={}
            results['IF']['numbers'] =    IF_Avg[0,:]
            results['IF']['avg_energy'] = IF_Avg[1,:]
            results['IF']['delay_time'] = IF_Avg[2,:]
            results['IF']['fract_time'] = IF_Avg[3,:]
            results['IF']['units'] = 'avg_energy: ergs/cm3, delay_time: seconds'
            save_obj(results,os.path.join(OutDir,OutFileName))
        print(i)
        return loop_INDX, L_halfs, HF_Avg,IF_Avg,LF_Avg
        


    '''
    def PredXRT(self,IDL_trespFile,passbands = None):

        #load XRT TStore_outputsresp:
        xrt_tresp = io.readsav(IDL_trespFile)
        filters = np.array(xrt_tresp['filters'])
        filters[0]=filters[0].decode('utf-8')
        filters[1]=filters[1].decode('utf-8')
        filters[2]=filters[2].decode('utf-8')
        filters[3]=filters[3].decode('utf-8')
        filters[4]=filters[4].decode('utf-8')
        filters[5]=filters[5].decode('utf-8')
        units = xrt_tresp['units'].decode('utf-8')
        xrt_tresp_logT = xrt_tresp['logt']

    filters = filters[0:3] #consider only 'Al-mesh', 'Al-poly', 'Be-thin'
    xrt_tresp['tr'] = xrt_tresp['tr'][0:3]

    if Store_outputs is True:
        results_obs['instruments']['XRT']['Tresp_log10T'] = xrt_tresp_logT
        results_obs['instruments']['XRT']['Tresp'] = xrt_tresp['tr']
        results_obs['instruments']['XRT']['Tresp_unit'] = 'DN cm5 s^{-1} px^{-1}'
    fig_comp,ax_comp = plt.subplots()

    XRT_pred_counts = []
    xrt_ch_all = []
    for fil_xrt in range(len(filters)): #loop over different-XRT filters
        print(filters[fil_xrt])

        #Read observed_image
        obs_image = Observed_image_directory+'XRT/XRTcut_FOV1_XBP001_'+filters[fil_xrt]+'.fits'
        map_xrt = sunpy.map.Map(obs_image)
        map_xrt.data[map_xrt.data < 0] = 0
        #remove some portion of the FOV to integrate the counts
        bottom_left = map_hmi_modeled.bottom_left_coord
        top_right = map_hmi_modeled.top_right_coord
        sub_map_xrt = map_xrt.submap(bottom_left=bottom_left, top_right=top_right) #XRT lev-1 data are in the unit of DN/sec/px
        xrt_exposure = sub_map_xrt.exposure_time.value
        ###

        tresp = xrt_tresp['tr'][fil_xrt]
        tresp = tresp* ((Au2Km/725)**2) #unit of DN cm^5 s^-1 Sr^-1

        #create DEM map to Image
        Model_image = demMap2Image(tresp,xrt_tresp_logT,DEM_logT,DEM_Map,Non_zero_pixcels)
        #Model_image = rotate(Model_image,angle=15,reshape=False)
        #Model_image = Model_image[indx:indx+dim[0],indy:indy+dim[1]]

        map_model_dummy.data[:,:] = Model_image
        Rebin_image = np.asarray(Img.fromarray(map_model_dummy.data).reduce(2)) #Rebin to original XRT resolution (1")
        Rebin_image = ndimage.gaussian_filter(Rebin_image, xrt_sigma) #Convolve the PSF
        Rebin_image = np.asarray(Img.fromarray(Rebin_image).reduce(2))#Rebin to XRT Synoptic images

        Rebin_image = Rebin_image * (2*725 / Au2Km)**2 # DN/s #2 is multiplied as each px is of 2"
        map_model_dummy.data[:,:] = map_model_dummy.data[:,:]*0 #re-initialize to zero
        if i ==0 :
            XRT_obs_counts += [np.average(sub_map_xrt.data)] #DN/s/px
            XRT_obs_exposure += [xrt_exposure]
        if Plot is True:
            if fil_xrt == 0:
                fig, axs = plt.subplots(1, 3,subplot_kw={'projection': map_xrt},figsize=(6, 3))
                AllAxes = fig.get_axes()
                if i == 0:
                    fig_obs, axs_obs = plt.subplots(1, 3,subplot_kw={'projection': map_xrt},figsize=(6, 3))
                    AllAxes_obs = fig_obs.get_axes()
            XRT_pred_counts += [np.average(Rebin_image)]

            pred_pp = AllAxes[fil_xrt].imshow(Rebin_image,origin='lower',interpolation='gaussian',cmap=map_xrt.cmap,norm=colors.PowerNorm(gamma=0.5))#,cmap="hot")#,vmin=v_min_xrt,vmax=v_max_xrt)
            plt.colorbar(pred_pp, ax=AllAxes[fil_xrt])
            AllAxes[fil_xrt].set_axis_off()#switch off axes
            if i==0:
                obs_pp = map_xrt.plot(axes=AllAxes_obs[fil_xrt],title=filters[fil_xrt],vmin=map_xrt.min(),vmax=map_xrt.max(),interpolation='gaussian')
                #obs_pp = AllAxes_obs[fil_xrt].imshow(sub_map_xrt.data,origin='lower',interpolation='gaussian',cmap=map_xrt.cmap,norm=colors.PowerNorm(gamma=0.5)) #interpolation='gaussian'
                plt.colorbar(obs_pp, ax=AllAxes_obs[fil_xrt])
                AllAxes_obs[fil_xrt].set_axis_off()
                #Draw a box for which AIA counts are considered
                coords = SkyCoord(Tx=(map_hmi_modeled.bottom_left_coord.Tx.value, map_hmi_modeled.top_right_coord.Tx.value)*u.arcsec,Ty=(map_hmi_modeled.bottom_left_coord.Ty.value, map_hmi_modeled.top_right_coord.Ty.value)*u.arcsec,frame=map_xrt.coordinate_frame)
                map_xrt.draw_quadrangle(coords,axes=AllAxes_obs[fil_xrt],edgecolor="blue",linestyle="-",linewidth=2,label='')
        if Store_outputs is True:
            if i ==0: results_obs['instruments']['XRT']['ObsIMG'][filters[fil_xrt]] = sub_map_xrt
            results['instruments']['XRT']['PredIMG'][filters[fil_xrt]] = Rebin_image
        xrt_ch_all += [filters[fil_xrt]]
    ##=================== END XRT ==========================

    def PredAIA(self,):

    def PredMaGIXS(self,):
    '''

    '''
    def demMap2Image(self,tresp,tresp_logT,DEM_logT,DEM_Map,Non_zero_pixcels,resolution = 0, ImgShape = None):
        #inputs:
        #    tresp -> 1-D response array 
        #    tresp_logT -> response logT grids
        #    DEM_Map -> 3D array, dimension = [x,y,logT]
        #    DEM_logT -> DEM logT grids
        #    Non_zero_pixcels -> 2-column array, representing the indices of non-zero pixcels to which the counts is to be 
        #                      calculated, to reduce the computation time.
        #outputs: Model_image -> model image in the resolution of DEM_Map
        
        DEM_logT = DEM_logT[1::]
        DEM_Map = DEM_Map[:,:,1::]
        tresp_interpolation = interpolate.interp1d(tresp_logT , tresp)
        G_T = tresp_interpolation(DEM_logT)
        dT = (shift(DEM_logT, -1, cval=0.0) - shift(DEM_logT, 1, cval=0.0)) * 0.5
        ntemps = len(DEM_logT)
        dT[0] = DEM_logT[1] - DEM_logT[0]
        dT[ntemps-1] = (DEM_logT[ntemps-1]-DEM_logT[ntemps-2])

        #indices of non-zero pixcels
        ind_xx = Non_zero_pixcels[0][:]
        ind_yy = Non_zero_pixcels[1][:]

        Model_image = DEM_Map[:,:,0]*0
        for i in range(len(ind_xx)):
            dem = DEM_Map[ind_xx[i],ind_yy[i],:]
            Model_image[ind_xx[i],ind_yy[i]] += sum(G_T * dem * (10**DEM_logT) *np.log(10.) * dT)
        #Rebin with given array:
        return Model_image
    '''
    '''
    def DEM2Image(self,DEMmap_file = '', resolution = 0, OutDir = None, tresp = None, tresp_logT = None, ImgShape = None):
        #
        #Purpose: Generate the XRT, AIA, and MaGIXS images from the EM array given by 'HydradOut2DEM_map.py'
        #
        #Outputs: Save the observed images in proper units and also save the predicted level-0 images in python .pkl file
        #
        #Note: Care should be taken while dealing with AIA images (except 94A channel), where the 
        #      pixcels values are forced to be integer in the observed data and hence in predicted 
        #      data. Thus uncertainty appear for the pixcels with a fraction of values.
        #
        #Biswajit, 09-June-2023
        #
        if os.path.isfile(DEMmap_file) is False: raise Exception("%% simar_error : DEM map file does not exist: "+DEMmap_file)
        if tresp is None: 
            raise Exception("%% simar_error : Define {tresp}") 
        elif np.array(tresp).ndim != 1 :raise Exception("%% simar_error : {TempResp} should be an 1D array")
        if tresp_logT is None: 
            raise Exception("%% simar_error : Define {tresp_logT}")
        elif np.array(tresp_logT).ndim != 1 :raise Exception("%% simar_error : {tresp_logT} should be an 1D array")

        DEM_map_data = load_obj(DEMmap_file[0:-4])
        DEM_logT = DEM_map_data['logT']
        DEM_Map  = DEM_map_data['DEM_Map']
        DEM_Map[np.isnan(DEM_Map)] = 0
    '''

'''
import fieldalign_model as sp
m=sp.fieldalign_model('/Users/bmondal/BM_Works/softwares/simar/examples/config.dat')
EM_map = m.DEMmap(Simulation_folder = '/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/Hydrad_outputs/MultiFreq_C0.20_Vs1.50/MultiFreq_0000/',HMI_image = '/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/HMI/HMIcut_newOBS_XBP001.fits',AIA_image='/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/AIA/AIAcut_FOV1_XBP001_211.fits',Chromospheric_height = 5)


'''
