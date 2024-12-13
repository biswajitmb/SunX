'''
Example 1 : It Simulate an AR 12758 on 2020-03-15T03:00

-- Biswajit. Dec.13.2024
'''


import sunx as ar
import sunpy.map
import os, glob
from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
from PIL import Image as Img
from scipy import ndimage
import matplotlib.colors as colors




###----- Prepare observed data ------
###----------------------------------

# Call SunX.fieldextrap class:
m = ar.fieldextrap(configfile='config.dat')

# Download HMI and AIA data 
#(Some time this does not work because of conectivity issue. Better to download the AIA and HMI data from JSOC and process them seperately)
#m.get_data(data_dir='./sample_data') 

# Process downloaded data using SunPy
#m.data_processing(AIA_Image='aia.lev1_euv_12s.2020-03-08T163006Z.193.image_lev1.fits',HMI_Mag='hmi.M_45s.20200308_163000_TAI.2.magnetogram.fits',aia_filt=193,AIA_dir=os.path.join('sample_data','AIA'),HMI_dir=os.path.join('sample_data','HMI'),remove_level1_data = False)

'''
# Select the region to simulate

Select the FOV center by clicking right-mouse botton on AIA full-disk image. Close the window at the end.
    This will store the cutout image and mahnetogram of that region inside 'cutout' directory
'''
m.select_region(OutFile = None,choose_FOV_center = True, FOV_xy_width = [150,150])

'''
#check the cutout images:
sunpy.map.Map('sample_data/AIA/cutouts/AIAcut_lev1.5_region_001.fits').peek()
sunpy.map.Map('sample_data/HMI/cutouts/HMIcut_lev1.5_region_001.fits').peek()
'''

'''
# Reproject the HMI mag to its FOV center direction
'''

AIAcut_image = os.path.join('sample_data','AIA','cutouts','AIAcut_lev1.5_region_001.fits')
HMI_Mag = 'HMIcut_lev1.5_region_001.fits'
OutFile = os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_01_newOBS.fits')
m.reproject_HMI_newOBS(NewFOV_xy_width = [250,250], FOV_centre_from_AIA=True,AIAcut_image=AIAcut_image,OutFile = OutFile,HMI_Mag=HMI_Mag,HMI_dir =os.path.join('sample_data','HMI','cutouts'))

'''
# Plot the reprojected HMI magnetogram:
sunpy.map.Map(os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_01_newOBS.fits')).peek()
'''

###----------------------------------
###----------------------------------


###-----   LFF Extrapolation   ------
###----------------------------------
'''
Run the extrapolation module to extrapolate LOS HMI magnetogram
'''
alpha = 0.0 # Appropiate alpha value can be decided for which traced loops match with AIA images
hmap_reproj = sunpy.map.Map(os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_01_newOBS.fits')) 
Bz = hmap_reproj.data
dx = hmap_reproj.scale.axis1.value ; dy = hmap_reproj.scale.axis2.value ; dz = dx
dr = [dx,dy,dz] #voxel length in arcsec for each directions
z_pix_hight = 500 #in pixcel unit of HMI
Bcube = m.gx_bz2lff(Bz, Nz=z_pix_hight, dr=dr, alpha1=alpha,
    OutFileName = 'LFF_Bcube_alpha_'+str(alpha), 
    OutDir=os.path.join('sample_data','HMI')
    )

###----------------------------------
###----------------------------------

###- Trace Field lines from Extrapolated magnetogram ------
###----------------------------------

'''
Using YT, trace the field lines. It will take some time.
    It will show HMI magnetogram.
    Using right-click, select the bottom left and top-roght location defining the region of seed points.

'''
Bcube_file = os.path.join('sample_data','HMI','LFF_Bcube_alpha_0.0.fits')
with fits.open(Bcube_file) as hdul:
    Bcube = np.nan_to_num(hdul[0].data,nan=0)
m.trace_fields(
    Bcube,
    min_footpoint_z=5,
    B_pos_thresold=20,
    B_neg_thresold=-20,
    OutFileName = 'LFF_loops_xyzB_0.0',
    OutDir = os.path.join('sample_data','HMI'),
    HMI_Mag = os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_01_newOBS.fits'),
    Seed_Zheight=0,
    Select_seed_region=True,
    N_lines=500, #Number of output fieldlines 
    dN=1000,     #Number of iteration in each YT streams.
    ) 

#Project the loops to Earth LOS:
m.Project_loops_to_OBS(
    HMI_Mag = os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_01_newOBS.fits'),
    HMI_Mag_orig = os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_001.fits'), 
    ExtPo_loop_par_file = os.path.join('sample_data','HMI','LFF_loops_xyzB_0.0.fits'),
    OutFile=None, 
    Plot=False, 
    AIA_Image = None, 
    PlotAIA=False)

'''
#Plot the field-lines on top of HMI and AIA images
m.plot_loop(
    HMI_Mag = os.path.join('sample_data','HMI','cutouts','HMIcut_lev1.5_region_001.fits'), 
    ExtPo_loop_par_file = os.path.join('sample_data','HMI','LFF_loops_xyzB_0.0.fits'),
    AIA_Image = os.path.join('sample_data','AIA','cutouts','AIAcut_lev1.5_region_001.fits'),
    alpha=0.01,
    N_loops=None,
    Proj_corr = True,
    StoreOutputs=True)
'''

#Calculate loop averaged parameters:
ExtPo_loop_par_file = os.path.join('sample_data','HMI','LFF_loops_xyzB_0.0.fits')
da = m.get_loop_parameters(
    N_loops=None, 
    ExtPo_loop_par_file = os.path.join('sample_data','HMI','LFF_loops_xyzB_0.0.fits'), 
    define_Bbase = 1, 
    Min_Z = 4,
    OutDir = os.path.join('sample_data','HMI'), 
    OutFileName = 'LFF_loops_xyzB_alpha_-0.001_LoopAvgPar'
    )
'''
## plot the loop parameters
#consider the loops whose any one footpoitns |B| > 30 G
B_thrs = 30
ind1 = np.where((np.abs(da[2])>B_thrs) | (np.abs(da[3])>B_thrs))[0]
#Plot B vs L
plt.plot(np.array(da[1])[ind1],np.array(da[6])[ind1],'*');plt.yscale('log');plt.xscale('log')
#plot L dist
plt.hist(np.array(da[1])[ind1],bins=20)
plt.show()
'''
###----------------------------------
###----------------------------------

###----------- Run EBTEL++ ----------
###----------------------------------

sim = ar.fieldalign_model(configfile='config.dat')

EBTEL_source_dir = '/Users/bmondal/BM_Works/softwares/ebtelPlusPlus'
EBTEL_ConfigFle = '/Users/bmondal/BM_Works/softwares/ebtelPlusPlus/config/ebtel.example.cfg.xml'

Loop_par_file = os.path.join('sample_data','HMI','LFF_loops_xyzB_alpha_0.0_LoopAvgPar.dat')
loop_ind, Full_length_Mm, LOS_B_left_foot, LOS_B_right_foot, ABS_B_left_foot, ABS_B_right_foot, Avg_B, ABS_B_left_defined_base, ABS_B_right_defined_base, Avg_B_above_base, Loop_height = np.loadtxt(Loop_par_file,unpack=True)

L_half = Full_length_Mm/2
B_Coronal_base = (ABS_B_left_defined_base+ABS_B_right_defined_base)/2
OutDir = os.path.join('EBTEL_results')
if os.path.isdir(OutDir) is False: os.mkdir(OutDir)

B_thrs = 20 ##consider the loops whose any one footpoitns |B| > 30 G
ind1 = np.where((np.abs(LOS_B_right_foot)>B_thrs) | (np.abs(LOS_B_right_foot)>B_thrs))[0]
L_half = L_half[ind1]
B_Avg = Avg_B[ind1]
B_Coronal_base = B_Coronal_base[ind1]
L_ind_no = loop_ind[ind1]

Fp = 1.0e6
tan_theta = 0.22
V_h = 1.0 #Does not have any use if Fp is not None.

sim.run_Ebtel(
    L_half,
    B_Avg,
    B_Coronal_base,
    L_ind_no,  
    V_h, 
    tan_theta, 
    OutDir, 
    SimulationTime = 10000, 
    tau_half = 50, 
    q0min_fract = q0min_fract, 
    BKG_T = 5.0e5, 
    Fp = Fp,
    Out_phys = True, 
    N_loops = None, 
    NumCore = 1,
    EBTEL_dir = EBTEL_dir, 
    EBTEL_ConfigFle = EBTEL_ConfigFle, 
    logTbinSize = 0.05, 
    logTmax = 7.5, 
    logTmin = 5.5
    )

###----------------------------------
###----------------------------------

###-- Create EM map from EBTEL outputs ---
###----------------------------------

### Create EM map from EBTEL outputs ###
name_str = 'C'+format('%0.2f'%tan_theta)+'_Fp'+format('%0.2f'%(Fp))
Simulation_folder = os.path.join('EBTEL_results','MultiFreq_'+name_str)
OutDir = os.path.join('EBTEL_results')
if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
HMI_Mag =  os.path.join('sample_data','HMI','HMI_lev1.5_LOSmag_20200315T030000.fits') #in Sun-Earth LOS
ExtPo_loop_par_file = os.path.join('sample_data','HMI','LFF_loops_xyzB_0.0.fits')

DEM_map = sim.DEMmap_ebtel(
    Simulation_folder, 
    HMI_Mag, 
    ExtPo_loop_par_file, 
    Tstart=4500, 
    Tstop=9500,
    dT = 10, 
    N_loops=2500,
    Chromospheric_height = 1, 
    OutDir=OutDir,
    name_str = name_str,
    Extrp_voxel_size = [0.3625,0.3625,0.3625],
    DEM_type = 'tot'
    )

'''
#Plot EMs at T = 1 MK and T = 3 MK
DEM_map = ar.load_obj(os.path.join('EBTEL_results','DEM_Map_HMIres_2500_C0.22_Fp1000000.00'))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(DEM_map['DEM_Map'][:,:,10],origin='lower', cmap='hot')
ax[0].set_title('T = '+format('%0.1f'%(10**DEM_map['logT'][10]/1.0e6))+' MK')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')

ax[1].imshow(DEM_map['DEM_Map'][:,:,19],origin='lower', cmap='hot')
ax[1].set_title('T = '+format('%0.1f'%(10**DEM_map['logT'][19]/1.0e6))+' MK')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')

plt.savefig('EBTEL_results/DEM.png')

plt.show()

'''

###----------------------------------
###----------------------------------

###-- Create Synthetic images and spectra ---
###----------------------------------
'''
TBD
'''
###----------------------------------
###----------------------------------



