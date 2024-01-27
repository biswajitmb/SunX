import simar as sp
import matplotlib.pyplot as plt
import glob

import warnings
warnings.simplefilter('ignore')
import logging, sys
logging.disable(sys.maxsize)

m = sp.fieldalign_model('/Users/bmondal/BM_Works/softwares/simar/examples/config.dat')

Simulation_dir = '/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/Hydrad_outputs/MultiFreq_*/MultiFreq_0000/'
HMI_image = '/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/HMI/HMIcut_newOBS_XBP001.fits'
AIA_image='/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/AIA/Final_CutOut/AIAcut_211.fits' #required to set the output dimension and projection
Loop_Files_dir='/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/Extrapolated_loops/Traced_loops_PosNeg_SeedMaxZ050.0/newOBS/XBP001/'
OutDir = '/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/Hydrad_outputs/MultiFreq_EM_maps/Final_DEMs/'

allfiles = glob.glob(Simulation_dir)

for i in range(len(allfiles)):
    Simulation_folder = allfiles[i]
    name_str = Simulation_folder.split('/')[-3][-12::]
    EM_map = m.DEMmap(Simulation_folder = Simulation_folder, HMI_image = HMI_image, AIA_image = AIA_image, Loop_Files_dir = Loop_Files_dir,Chromospheric_height = 5,Store_outputs = True,OutDir=OutDir,name_str=name_str,Tstart=8000, Tstop=9000)

    #plt.imshow(EM_map[:,:,5],origin='lower',vmax=1.0e2);plt.show()



