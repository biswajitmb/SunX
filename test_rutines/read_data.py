import configparser
from arsim_util import *
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.time import Time
import numpy as np
import os,sys

config = configparser.ConfigParser()
config.read('config.dat')

datetime = config['SelectEvent']['event_dateTime']

#date = datetime[0:-9]
#time_mid = datetime[-8::]

data_dir = config['SelectEvent']['data_dir']

#Search if HMI data pre-exist
hmi_dir = data_dir+'HMI/'

#Search if AIA data pre-exist
aia_dir = data_dir+'AIA/'

#If not search from Fido

start_time = Time(datetime, scale='utc', format='isot')
results = Fido.search(a.Time(start_time - 15*u.s, start_time + 15*u.s), a.Instrument.aia | a.Instrument.hmi)
aia, hmi = results 
#results.show("Start Time", "Instrument", "Physobs", "Wavelength") 
hmi_los = hmi[hmi["Physobs"] == "LOS_magnetic_field"]
aia_wl = aia[aia["Wavelength"][:, 0] == int(config['SelectEvent']['aia_filter'])* u.AA]  

aia_times = Time(aia_wl['Start Time'].value) 
hmi_times = Time(hmi_los['Start Time'].value)

time_req = Time(datetime)

aia_final_file = aia_wl[aia_wl["Start Time"] == nearest_time(aia_times,time_req)]
hmi_final_file = hmi_los[hmi_los["Start Time"] == nearest_time(hmi_times,time_req)]

#downloaded_files = Fido.fetch(results, path=os.path.join(data_dir,'{file}')) 
#downloaded_files
