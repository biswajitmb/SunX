import configparser
from sunx.util import *
import numpy as np
from scipy.io import readsav
from scipy import interpolate
import matplotlib.pyplot as plt
from pydrad.parse import Strand
from scipy.ndimage.interpolation import shift
import os, glob
from fiasco import Element

class nei(object):
    def __init__(self,configfile=None):
        if os.path.isfile(configfile) is False:
            raise Exception("%% SunX_error : Congig-File not exist- "+configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config = config
        self.HydradDir = config['NEI']['HydradDir']
        self.results_hydrad = Strand(self.HydradDir)
        OutDir = os.path.join(self.HydradDir,'NEI')
        #if os.path.isdir(OutDir) is False: os.mkdir(OutDir)  
        self.OutDir = OutDir
        #IDL_DataDir = os.path.join(self.HydradDir,'NEI')
        #self.IDL_DataDir = os.path.join(IDL_DataDir,'chianti')
        EQ_IF_logT, ioneq = read_EQ_file(config['NEI']['EquilIonFraction_file']) #ioneq[logT, element, chrg_state]
        print('%% SunX messege: Using Equilibrium Ion Fraction File - '+config['NEI']['EquilIonFraction_file'])
        self.EQ_IF_logT = EQ_IF_logT
        self.ioneq = ioneq #ioneq[logT, element, chrg_state]

    def AvgHydrad_TemRho(self,ChromosHeight=5,tstart=None,tstop=None):
        '''
            Purpose: Return the length-averaged T, density from Hydrad runs as a function of time
        '''
        results_hydrad = self.results_hydrad
        Avg_density =[]; Avg_T =[]
        times = results_hydrad.time.value
        if tstart == None: tstart=times[0]
        if tstop == None: tstop = times[-1]
        indt = np.where(((times >= tstart) & (times <= tstop)))[0]
        times = times[indt]
        results_hydrad = results_hydrad[indt]
        n_times = len(times)
        Loop_Length = results_hydrad.loop_length.value/1.0e8
        for tim in range(n_times):
            grid_centers = results_hydrad[tim].__dict__['_grid_centers']/1.0e8
            ind = np.where(((grid_centers > ChromosHeight) & (grid_centers<(Loop_Length-ChromosHeight))))[0]
            grid_centers = grid_centers[ind]
            ndl = len(grid_centers)
            dL = (shift(grid_centers, -1, cval=0.0) - shift(grid_centers, 1, cval=0.0)) * 0.5
            dL[0] = grid_centers[1] - grid_centers[0]
            dL[ndl-1] = (grid_centers[ndl-1] - grid_centers[ndl-2])
            Temp = results_hydrad[tim].electron_temperature.value[ind]
            Rho = results_hydrad[tim].electron_density.value[ind]
            L = sum(dL)
            Avg_T += [sum(Temp * dL)/L]
            Avg_density += [sum(Rho * dL)/L]
        return times,np.array(Avg_T),np.array(Avg_density)

    def AvgHydradNEI_frac(self,elements_Z,Charge_state=None, tstart=None,tstop=None, ChromosHeight=5,info=True):
        '''
            Purpose: Return the length-averaged T, density, equilibrium ion-fraction and NEI ion-fraction from Hydrad-NEI runs as a function of time
        '''
        results_hydrad = self.results_hydrad
        times = results_hydrad.time.value
        Loop_Length = results_hydrad.loop_length.value/1.0e8
        if tstart == None: tstart=times[0]
        if tstop == None: tstop = times[-1]
        indt = np.where(((times >= tstart) & (times <= tstop)))[0]
        times = times[indt]
        results_hydrad = results_hydrad[indt]
        n_times = len(times)
        ele_lon = element(elements_Z)[4]
        ele = element(elements_Z)[3]
        key = '_population_fraction_'+ele_lon

        EQ_IF_logT = self.EQ_IF_logT
        ioneq = self.ioneq 

        Avg_T = []; Avg_density=[]
        for tim in range(n_times):
            NEI_IF = results_hydrad[tim].__dict__[key] #[length_array,charge_states]
            grid_centers = results_hydrad[tim].__dict__['_grid_centers']/1.0e8
            ind = np.where(((grid_centers > ChromosHeight) & (grid_centers<(Loop_Length-ChromosHeight))))[0]
            grid_centers = grid_centers[ind]
            Temp = results_hydrad[tim].electron_temperature.value[ind]
            Rho = results_hydrad[tim].electron_density.value[ind]
            ndl = len(grid_centers)
            dL = (shift(grid_centers, -1, cval=0.0) - shift(grid_centers, 1, cval=0.0)) * 0.5
            dL[0] = grid_centers[1] - grid_centers[0]
            dL[ndl-1] = (grid_centers[ndl-1] - grid_centers[ndl-2])
            NEI_IF = NEI_IF[ind,:]
            dl_tot = sum(dL)
            Avg_T += [sum(Temp * dL)/dl_tot]
            Avg_density += [sum(Rho * dL)/dl_tot]
            if tim == 0:
                if Charge_state is None: Charge_state = np.arange(len(NEI_IF[0,:]))+1
                else:Charge_state = np.array(Charge_state)
                NEI_IonFrac = np.zeros([n_times, len(Charge_state)]) #[time,chargestate]
                EQ_IonFrac = np.zeros([n_times, len(Charge_state)]) #[time,chargestate]
            for chs in range(len(Charge_state)):
                NEI_IonFrac[tim,chs] = sum(NEI_IF[:,Charge_state[chs]-1]*dL)/dl_tot
                intp_func_IF0 = interpolate.interp1d(EQ_IF_logT,ioneq[:,elements_Z-1,Charge_state[chs]-1],kind='cubic')
                EQ_IonFrac[tim,chs] = intp_func_IF0(np.log10(Avg_T[tim]))
        return times,np.array(Avg_T),np.array(Avg_density),NEI_IonFrac, EQ_IonFrac,Charge_state

    def get_lc_spec(self,DataDir,nei_file = 'nei_outputs',chianti_outdir = None):

        '''
        nei_file -> Output of AvgHydradNEI_frac()
        DataDir -> directory contaning the unity-ion-fraction file and  chianti/IonSpec_ioneq_unity/IonSpec_ioneq_unity_fe.idl file
        '''
        #if DataDir is None: DataDir = self.OutDir
        #Read the NEI evolution file:
        neidata = load_obj(os.path.join(DataDir,nei_file))
        elements_Z = neidata['elements']['Z']
        element_chr = neidata['elements']['Charge_state']

        nele = len(elements_Z)

        for i in range(nele):
            UnityEQ_spec_file = 'IonSpec_ioneq_unity_'+element(elements_Z[i])[3]+'.idl'
            if chianti_outdir is None: UnityEQ_spec_file = os.path.join(DataDir,'chianti','IonSpec_ioneq_unity',UnityEQ_spec_file)
            else: UnityEQ_spec_file = os.path.join(chianti_outdir,UnityEQ_spec_file)
            data_unit = readsav(os.path.join(UnityEQ_spec_file))
            time = data_unit['time']
            ind_sort = np.argsort(time)
            time = time[ind_sort]
            temperatures = data_unit['temperature'][ind_sort]
            n_time = len(time)
            spec = data_unit['allspec'][:,ind_sort,:] #[chrgstate,time,energy]
            ion_list = data_unit['ion_list'].astype(str)
            for ii in range(len(ion_list)): #Remove the flag 'd' for satelite lines
                if ion_list[ii][-1] == 'd': ion_list[ii] = ion_list[ii][0:-1]
            n_chrs = len(element_chr[i])
            if i == 0:
                n_energy = len(data_unit['allspec'][0,0,:])
                NonEQ_spec = np.zeros([nele,31,n_energy]) #elements,chrg_states,energies
                EQ_spec = np.zeros([nele,31,n_energy])
                NonEQ_I = np.zeros([nele,31,n_time]) #elements,chrg_states,time
                EQ_I = np.zeros([nele,31,n_time])
            NEI_IF = neidata['IonFraction']['NEI'][element(elements_Z[i])[3]]#[time,charge_states]
            EQ_IF = neidata['IonFraction']['EQ'][element(elements_Z[i])[3]]
            for j in range(n_chrs):
                ind = np.where(ion_list == element(elements_Z[i])[3]+'_'+format('%d'%element_chr[i][j]))[0]
                if len(ind) > 0: #Consider only valid charge-states 
                    for mi in ind :
                        EQ_spec[i,j,:] += np.dot(EQ_IF[:,j],spec[mi,:,:])/n_time #[chrgstate,time,energy]
                        NonEQ_spec[i,j,:] += np.dot(NEI_IF[:,j],spec[mi,:,:]) / n_time

                        for t in range(n_time): #LC
                            EQ_I[i,j,t] += EQ_IF[t,j]*sum(spec[mi,t,:])
                            NonEQ_I[i,j,t] += NEI_IF[t,j]*sum(spec[mi,t,:])

        return time,temperatures,data_unit['allene'],EQ_I,NonEQ_I,EQ_spec,NonEQ_spec,elements_Z,element_chr

    def fiascoAvgNEI(self,elements_Z,fname='fiasconei',ChromosHeight=5,tstart=None,tstop=None,HydradDir=None):
        '''
            Estimate the evolution of length-average NEI fraction of the loop using hydrad-outputs (without NEI)
            NEI evolutions will be stored in python files inside hydrad_outdir/NEI
        ''' 
        if HydradDir is None: results_hydrad = self.results_hydrad ; OutDir = os.path.join(self.HydradDir,'NEI')
        else: results_hydrad = Strand(HydradDir) ; OutDir = os.path.join(HydradDir,'NEI')
        if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
        times = results_hydrad.time
        Loop_Length = results_hydrad.loop_length.value/1.0e8
        tstart = tstart*u.s ; tstop = tstop*u.s
        if tstart == None: tstart=times[0]
        if tstop == None: tstop = times[-1]
        indt = np.where(((times >= tstart) & (times <= tstop)))[0]
        times = times[indt]
        results_hydrad = results_hydrad[indt]
        n_times = len(times)

        elements_Z = np.array(elements_Z)
        nelem = len(elements_Z)

        #EQ_IF_logT = self.EQ_IF_logT * u.K
        #ioneq = self.ioneq #ioneq[logT, element, chrg_state]

        Avg_T = []; Avg_density=[]
        for tim in range(n_times):
            grid_centers = results_hydrad[tim].grid_centers.value/1.0e8
            ind = np.where(((grid_centers > ChromosHeight) & (grid_centers<(Loop_Length-ChromosHeight))))[0]
            grid_centers = grid_centers[ind]
            Temp = results_hydrad[tim].electron_temperature.value[ind]
            Rho = results_hydrad[tim].electron_density.value[ind]
            ndl = len(grid_centers)
            dL = (shift(grid_centers, -1, cval=0.0) - shift(grid_centers, 1, cval=0.0)) * 0.5
            dL[0] = grid_centers[1] - grid_centers[0]
            dL[ndl-1] = (grid_centers[ndl-1] - grid_centers[ndl-2])
            dl_tot = sum(dL)
            Avg_T += [sum(Temp * dL)/dl_tot]
            Avg_density += [sum(Rho * dL)/dl_tot]

        Avg_T = np.array(Avg_T) ; Avg_density = np.array(Avg_density)
        temperature = Avg_T * u.K
        density = Avg_density * u.cm**-3
        
        #Save the IFs in python file for later use:
        results = {'NEI-results': 'Hydrad'}
        results['elements'] = {}
        results['IonFraction'] = {}
        results['IonFraction']['EQ'] = {}
        results['IonFraction']['NEI'] = {}
       
        All_chrg_states = [] 
        temperature_array = np.logspace(4, 8, 1000) * u.K
        for e in range(nelem):
            # Equilibrim IF
            elem = element(elements_Z[e])[4]
            elem = Element(elem, temperature_array)
            func_interp = interpolate.interp1d(elem.temperature.to_value('K'), elem.equilibrium_ionization.value,axis=0, kind='cubic', fill_value='extrapolate')
            elem_ieq = u.Quantity(func_interp(temperature.to_value('K')))

            #NEI calculation
            elem_nei = np.zeros(times.shape + (elem.atomic_number + 1,)) #elem_nei[time,charge_states]
            elem_nei[0, :] = elem_ieq[0,:]
            func_interp = interpolate.interp1d(elem.temperature.to_value('K'), elem._rate_matrix.value,axis=0, kind='cubic', fill_value='extrapolate')
            elem_rate_matrix = func_interp(temperature.to_value('K')) * elem._rate_matrix.unit

            identity = u.Quantity(np.eye(elem.atomic_number + 1))
            for i in range(1, n_times):
                dt = times[i] - times[i-1]
                term1 = identity - density[i] * dt/2. * elem_rate_matrix[i, ...]
                term2 = identity + density[i-1] * dt/2. * elem_rate_matrix[i-1, ...]
                elem_nei[i, :] = np.linalg.inv(term1) @ term2 @ elem_nei[i-1, :]
                elem_nei[i, :] = np.fabs(elem_nei[i, :])
                elem_nei[i, :] /= elem_nei[i, :].sum()
            
            elem_nei = u.Quantity(elem_nei)#elem_nei[time,charge_states]

            results['IonFraction']['EQ'][element(elements_Z[e])[3]] = elem_ieq.value
            results['IonFraction']['NEI'][element(elements_Z[e])[3]] = elem_nei.value
            All_chrg_states += [np.arange(elem.atomic_number + 1)+1]

        results['time'] = times.value
        results['temperature'] = Avg_T
        results['density'] = Avg_density
        results['elements']['Z'] = elements_Z
        results['elements']['Charge_state'] = All_chrg_states #All_chrg_states
        save_obj(results, os.path.join(OutDir,fname))

        return None



