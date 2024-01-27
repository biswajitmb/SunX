import numpy as np
import matplotlib.pyplot as plt
import simar as fld
from sunpy.map 
from util import *
import os

data_dir = 'sample_data/'
HMI_magnetogram = data_dir+'HMIcut_newOBS_XBP001.fits'

m = fld.fieldextrap(configfile='config.dat')

hmi = sunpy.map.Map(HMI_magnetogram)
bz0 = hmi.data


if os.isfile(data_dir+'ExtpBcube_HMIcut_newOBS_XBP001.pkl') is False:
    bcube = m.gx_bz2lff(bz0, Nz=100, dr=[0.5,0.5,0.5], alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False)
    save_obj(bcube,data_dir+'ExtpBcube_HMIcut_newOBS_XBP001')
else: bcube = load_obj(bcube,data_dir+'ExtpBcube_HMIcut_newOBS_XBP001')


