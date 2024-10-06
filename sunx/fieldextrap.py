'''
Purpose:
    Extrapolate the selected field from phorospheric magnetogram.
Biswajit, Aug.28.2023

'''

import configparser
from sunx.util import *
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.time import Time
import numpy as np
import os,sys,glob
import sunpy.map
from aiapy.calibrate import register, update_pointing,fix_observer_location
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.io import fits
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
from astropy.table import Table
import ast
import multiprocessing as mulp
from functools import partial
from sunpy.coordinates import get_body_heliographic_stonyhurst,frames
import yt
from yt.visualization.api import Streamlines

class fieldextrap(object): 
    def __init__(self,configfile=None):
        if os.path.isfile(configfile) is False:
            raise Exception("%% SunX_error : Congig-File not exist- "+configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config = config
        #try:
        NumCore=int(self.config['SelectEvent']['NumCore'])
        if NumCore > 0:
            self.cpu = mulp.cpu_count()
            if NumCore > self.cpu:
                print("%% SunX_message: NumCore is exciding the maximum value. Set NumCore = "+str(cpu))
                NumCore = self.cpu 
            #print("%% SunX_status: Parallel processing is using with no.cores = "+str(NumCore))
            #self.NumCore = NumCore
        elif NumCore == 0: print("%% SunX_massege: Using single process!") 
        else: print("%% SunX_Error: < NumCore > should be an integer, like 0,1,2,3... (0 indicated no parallel process)")
        #except:self.NumCore = 0
        self.NumCore = NumCore
        self.arcs2Mm = 0.725

    def get_data(self,data_dir=None,Datetime = None, aia_filt = None):
        '''
        Purpose: Download the HMI (LOS-mag) and AIA (given channel) data for the closest epoch of the time in configfile.
        '''
        config = self.config
        if data_dir is None: data_dir = config['SelectEvent']['data_dir']
        if os.path.isdir(data_dir) is False:raise Exception("%% SunX_error : Directory not exist- "+data_dir)

        hmi_dir = os.path.join(data_dir,'HMI')
        aia_dir = os.path.join(data_dir,'AIA')
        if os.path.isdir(hmi_dir) is False: os.mkdir(hmi_dir)
        if os.path.isdir(aia_dir) is False: os.mkdir(aia_dir)
        if Datetime is None: Datetime = config['SelectEvent']['event_dateTime']
        if aia_filt is None: aia_filt = config['SelectEvent']['aia_filter']
 
        start_time = Time(Datetime, scale='utc', format='isot')

        hmi = Fido.search(a.Time(start_time - 15*u.s, start_time + 15*u.s), a.Instrument.hmi)
        hmi_los = hmi['vso'][hmi['vso']["Physobs"] == "LOS_magnetic_field"]
        hmi_times = Time(hmi_los['Start Time'].value)
        hmi_final_file = hmi_los[hmi_los["Start Time"] == nearest_time(hmi_times,start_time)]
        downloaded_files = Fido.fetch(hmi_final_file, path=os.path.join(data_dir,'HMI'))
        t = str(hmi_final_file["Start Time"].value[0]).split(' ')
        t1 = t[1].split(':')
        mm = ind_2(int(np.round(float(t1[1])+(float(t1[2])/60.0))))
        #hfname = 'hmi_m_45s_'+t[0][0:4]+'_'+t[0][5:7]+'_'+t[0][8:10]+'_'+t1[0]+'_'+mm+'_'+'45_tai_magnetogram.fits'
        hfname = glob.glob(os.path.join(hmi_dir,'hmi_m_45s_'+t[0][0:4]+'_'+t[0][5:7]+'_'+t[0][8:10]+'_'+t1[0]+'*_tai_magnetogram.fits'))[0]
        hfname = os.path.basename(hfname)

        #if os.path.isfile(os.path.join(aia_dir,aia_fname)) is False:
        aia = Fido.search(a.Time(start_time - 15*u.s, start_time + 15*u.s), a.Instrument.aia)
        aia_wl = aia['vso'][aia['vso']["Wavelength"][:, 0] == int(aia_filt)* u.AA]
        aia_times = Time(aia_wl['Start Time'].value)
        aia_final_file = aia_wl[aia_wl["Start Time"] == nearest_time(aia_times,start_time)]
        #print(aia_final_file)
        downloaded_files = Fido.fetch(aia_final_file, path=os.path.join(data_dir,'AIA'))
        t = str(aia_final_file["Start Time"].value[0]).split(' ')
        t1 = t[1].split(':')
        #afname = glob.glob(os.path.join(aia_dir,'aia_lev1_'+ind_2(aia_filt)+'a'+t[0][0:4]+'_'+t[0][5:7]+'_'+t[0][8:10]+'t'+t1[0]+'_'+t1[1]+'_'+t1[2].split('.')[0]+'_*_image_lev1.fits'))[0]
        afname = glob.glob(os.path.join(aia_dir,'aia_lev1_*'+t[0][0:4]+'_'+t[0][5:7]+'_'+t[0][8:10]+'t'+t1[0]+'_'+t1[1]+'_'+t1[2].split('.')[0]+'_*_image_lev1.fits'))[0]
        afname = os.path.basename(afname)

        self.hfname = hfname
        self.afname = afname

    def data_processing(self,AIA_Image=None, HMI_Mag = None,aia_filt=None, AIA_dir = None, HMI_dir = None, remove_level1_data = True,Datetime=None):
        '''
        Purpose: Process the downloaded level-1 data to higher level (e.g., 1.5)
        '''
        if Datetime is None: Datetime = self.config['SelectEvent']['event_dateTime']
        if aia_filt is None: aia_filt = self.config['SelectEvent']['aia_filter']
        if remove_level1_data is None: self.config['SelectEvent']['remove_level1_data']

        datetim = datetime.strptime(Datetime,'%Y-%m-%dT%H:%M:%S')
        if AIA_dir is None: adir = os.path.join(self.config['SelectEvent']['data_dir'],'AIA')
        else: adir = AIA_dir
        if AIA_Image is None: AIA_Image = self.afname
        if HMI_Mag is None: HMI_Mag = self.hfname
        self.afname = AIA_Image ; self.hfname = HMI_Mag
        m = sunpy.map.Map(os.path.join(adir,AIA_Image))
        m_updated_pointing = update_pointing(m)
        m_observer_fixed = fix_observer_location(m_updated_pointing)
        m_registered = register(m_updated_pointing)
        #m_normalized = normalize_exposure(m_registered) #not part of level 1.5 “prep” data. It will convert the data in unit DN/px/ s.
        m_registered.save(os.path.join(adir,'AIA_lev1.5_'+ind_4(aia_filt)+'_'+format('%0.4d'%datetim.year)+format('%0.2d'%datetim.month)+format('%0.2d'%datetim.day)+'T'+format('%0.2d'%datetim.hour)+format('%0.2d'%datetim.minute)+format('%0.2d'%datetim.second)+'.fits'),overwrite='True')

        if HMI_dir is None: hdir = os.path.join(self.config['SelectEvent']['data_dir'],'HMI')
        else: hdir = HMI_dir
        h = sunpy.map.Map(os.path.join(hdir,HMI_Mag))
        prep_hmi = h.rotate(order=3)
        prep_hmi.save(os.path.join(hdir,'HMI_lev1.5_'+'LOSmag'+'_'+format('%0.4d'%datetim.year)+format('%0.2d'%datetim.month)+format('%0.2d'%datetim.day)+'T'+format('%0.2d'%datetim.hour)+format('%0.2d'%datetim.minute)+format('%0.2d'%datetim.second)+'.fits'),overwrite='True')
        if remove_level1_data is True:
            os.remove(os.path.join(adir,self.afname))
            os.remove(os.path.join(hdir,self.hfname)) 

    def reproject_HMI_newOBS(self,NewFOV_xy_width = [200,200],HMI_Mag = None,New_Obs_dist = None,Plot = False, FOV_centre = None,FOV_centre_from_AIA = True, FOV=None,AIAcut_image = None, OutFile=None, HMI_dir = None):
        '''
            Reproject the HMI magnetogram to different observer
            FOV_centre_from_AIA -> If true the center and FOV will be selected from provided AIA cutout image
            FOV_centre -> Center of the region in original HMI image. (Not requier if FOV_centre_from_AIA = True)
            NewFOV_xy_width = FOV from the center of projected HMi image added to the new image in [x, y] in arcsec. 
                              (Not required if FOV_centre_from_AIA = True, then this will be determined from given AIA image)
            New_Obs_dist -> Observer distance in astropy distance coordinate e.g., 1.43457186e+11*u.m
            AIAcut_image ->
        '''
        if HMI_Mag is None: HMI_Mag = self.hfname
        if HMI_dir is None: hdir = os.path.join(self.config['SelectEvent']['data_dir'],'HMI')
        else: hdir = HMI_dir
        hmap = sunpy.map.Map(os.path.join(hdir,HMI_Mag))
        if OutFile is None: 
            OutDir = os.path.join(hdir,'cutouts')
            if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
            OutFile = os.path.join(OutDir,'HMIcut_lev1.5_region_01_newOBS.fits')
        if FOV_centre_from_AIA is False:
            if FOV_centre is None: 
                if FOV is None: FOV = np.array(ast.literal_eval(self.config['SelectEvent']['FOV'])) 
                else: FOV = np.array(FOV)
                if FOV.shape != (2,2): Exception("%% SunX_error : Incorrect FOV. It should be an 2D array [[xy_bot_left],[xy_top_right]]")
                FOV_centre = np.array([(FOV[0][0]+FOV[1][0])/2 , (FOV[0][1]+FOV[1][1])/2])
        else: 
            if AIAcut_image is None: 
                AIA_cut_dir = os.path.join(self.config['SelectEvent']['data_dir'],'AIA','cutouts')
                AIAcut_image = os.path.join(AIA_cut_dir,'AIAcut_lev1.5_region_01.fits')
            amap = sunpy.map.Map(AIAcut_image)
            #FOV = [[amap.bottom_left_coord.Tx.value,amap.bottom_left_coord.Ty.value],[amap.top_right_coord.Tx.value,amap.top_right_coord.Ty.value]]
            #FOV_centre = np.array([(FOV[0][0]+FOV[1][0])/2 , (FOV[0][1]+FOV[1][1])/2])
            FOV_centre = np.array([amap.center.Tx.value,amap.center.Ty.value])
            bottom_left = amap.bottom_left_coord ; top_right = amap.top_right_coord
            #NewFOV_xy_width = np.array([top_right.Tx.value - bottom_left.Tx.value , top_right.Ty.value - bottom_left.Ty.value])
        stonyhurst_frame = frames.HeliographicStonyhurst(obstime=hmap.date)
        pointinhmap = SkyCoord(FOV_centre[0]*u.arcsec, FOV_centre[1]*u.arcsec,frame=hmap.coordinate_frame)
        Observer_lon_lat = pointinhmap.transform_to(stonyhurst_frame)
        if New_Obs_dist is None: New_Obs_dist = hmap.dsun
        Observer_lon_lat = SkyCoord(Observer_lon_lat.lon,Observer_lon_lat.lat,New_Obs_dist,frame='heliographic_stonyhurst', obstime=hmap.date)
        coordinate = hmap.reference_coordinate.replicate(rsun=hmap.reference_coordinate.rsun,observer = Observer_lon_lat) 
        out_shape = hmap.data.shape

        out_header = sunpy.map.make_fitswcs_header(
            out_shape,
            coordinate,
            scale=u.Quantity(hmap.scale),
            instrument="HMI",
            observatory="SDO",
            #wavelength=map_aia.wavelength
        )
        outmap = hmap.reproject_to(out_header)
        #if FOV_centre_from_AIA is False:
        NewFOV_xy_width_half = np.array(NewFOV_xy_width)/2
        new_center = [outmap.center.Tx.value,outmap.center.Ty.value]
        bottom_left = SkyCoord((new_center[0]-NewFOV_xy_width_half[0])*u.arcsec,(new_center[1]-NewFOV_xy_width_half[1])*u.arcsec, frame=outmap.coordinate_frame)
        top_right = SkyCoord((new_center[0]+NewFOV_xy_width_half[0])*u.arcsec,(new_center[1]+NewFOV_xy_width_half[1])*u.arcsec, frame=outmap.coordinate_frame)
        ###
        

        outmap_crop = outmap.submap(bottom_left, top_right=top_right)
        outmap_crop.save(os.path.join(OutFile),overwrite='True')

        if Plot is True:
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection=hmap)
            hmap.plot_settings['norm'].vmin = -500
            hmap.plot_settings['norm'].vmax = 500
            hmap.plot(axes=ax1)
            #hmap.draw_grid(axes=ax1, color='blue')
            ax2 = fig.add_subplot(122, projection=outmap)
            outmap_crop.plot(axes=ax2, title='New Observer')
            outmap_crop.plot_settings['norm'].vmin = -500
            outmap_crop.plot_settings['norm'].vmax = 500
            ax2.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
            ax2.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
            plt.show()
        return outmap_crop

    def Project_loops_to_OBS(self,HMI_Mag = None,HMI_Mag_orig = None, ExtPo_loop_par_file = None,OutFile=None,N_loops=None, Plot=False, AIA_Image = None, PlotAIA=False,lw=2,alpha = 0.5, color = 'C0', StoreOutput = True,hmivmin = -500, hmivmax = 500):
        '''
        Purpose: Project the extrapolated loops towards original HMI observer LOS. (It can be implemented for any LOS in future).

        '''        
        if HMI_Mag is None: HMI_Mag = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','cutouts','HMIcut_lev1.5_region_01_NewOBS.fits')
        if HMI_Mag_orig is None: HMI_Mag_orig = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','cutouts','HMIcut_lev1.5_region_01.fits')
        if ExtPo_loop_par_file is None: ExtPo_loop_par_file = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','PF','PF_loops_xyzB.fits')
        if OutFile is None: OutFile = ExtPo_loop_par_file
        hmap = sunpy.map.Map(HMI_Mag)
        hmap_orig = sunpy.map.Map(HMI_Mag_orig)
        bot_left = hmap_orig.bottom_left_coord
        top_right = hmap_orig.top_right_coord

        if Plot is True:
            if PlotAIA is True:
                if AIA_Image is None: AIA_Image = os.path.join(self.config['SelectEvent']['data_dir'],'AIA','cutouts','AIAcut_lev1.5_region_01.fits')
                amap = sunpy.map.Map(AIA_Image)
                fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': amap})
                AllAxes=plt.gcf().get_axes()
                AllAxes[0].set_title('HMI overlaid on AIA')
                O = AllAxes[0].imshow(amap.data,origin='lower',interpolation='nearest',cmap=amap.cmap,norm=colors.PowerNorm(gamma=0.5))
                #Overplot HMI contures
                levels = [30,50, 100, 150, 300, 500, 1000] * u.Gauss
                levels = np.concatenate((-1 * levels[::-1], levels))
                bounds = AllAxes[0].axis()
                cset = hmap_orig.draw_contours(levels, axes=AllAxes[0], cmap='seismic', alpha=0.5)
                AllAxes[0].axis(bounds)
            else:
                fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': hmap_orig})
                AllAxes=plt.gcf().get_axes()
                AllAxes[0].set_title('HMI overlaid on AIA')
                hmap_orig.plot_settings['norm'].vmin = hmivmin
                hmap_orig.plot_settings['norm'].vmax = hmivmax
                hmap_orig.plot(axes = AllAxes[0])

        with fits.open(ExtPo_loop_par_file) as hdul:
            nloops = len(hdul)
            if N_loops is not None:
                if N_loops > nloops:
                     print('%SunX_message: N_loops is exeeding maximum available loops. Set to maximum value of ->'+format('%d'%nloops))
                else: nloops = N_loops
            for l in range(2,nloops):
                data = hdul[l].data
                data_WCS = self.loop_WCS(hmap,[data['y'],data['x'],data['z']]) #WCS coordinates in extrapolated HMI_mag LOS
                data_WCS = data_WCS.transform_to(hmap_orig.coordinate_frame) #Original HMI_mag LOS
                #pixx = np.full((len(data_WCS.Tx.value)),np.nan) 
                #pixy = np.full((len(data_WCS.Ty.value)),np.nan)
                flag = np.zeros(len(data_WCS.Tx.value))#,dtype=int)
                ind1 = np.where((data_WCS.Tx.value >= bot_left.Tx.value) & (data_WCS.Tx.value <= top_right.Tx.value))
                ind2 = np.where((data_WCS.Ty.value >= bot_left.Ty.value) & (data_WCS.Ty.value <= top_right.Ty.value))
                ind = np.intersect1d(ind1,ind2)
                flag[ind] = 1 #flag = 1 for the loops coordinates inside FOV
                #if len(ind) > 3: #Mark the inside loops with a minimum grid points 3.
                #data_WCS = data_WCS[ind] #Remove the loops coordinate/loops outside Original HMI_mag
                #if len(data_WCS.Tx.value) > 3: #Consider the loops with a minimum 3 grids
                pix = hmap_orig.world_to_pixel(data_WCS)
                pixx = pix.x.value
                pixy = pix.y.value
                error_stat = 0
                if 'xp' in data.columns.names: 
                    #data['xp'] = pixx
                    #data['yp'] = pixx
                    #data['flag'] = flag
                    error_stat = 1
                    print("%% SunX_error : Column 'xp' exist. Update table data functionality will to be available in future!!")
                    break 
                else:
                    c10 = fits.Column(name='xp', array=pixx, format='K');
                    c20 = fits.Column(name='yp', array=pixy, format='K');
                    c30 = fits.Column(name='flag', array=flag, format='K');
                    new_cols = data.columns + fits.ColDefs([c10,c20,c30])
                    #new_table_hdu = fits.TableHDU.from_columns(data.columns+[c1,c2,c3],name='L'+format('%d'%l_ind))
                    new_table_hdu = fits.TableHDU.from_columns(new_cols,name=hdul[l].name)
                    hdul[l] = new_table_hdu
                if Plot is True: AllAxes[0].plot_coord(data_WCS, color=color, lw=lw, alpha=alpha)
            if StoreOutput is True and error_stat != 1: hdul.writeto(OutFile,overwrite=True)
            if Plot is True: plt.show()
        

    def get_all_AIA_cutouts(self,AIA_filters = [94,131,335,304,171,211,193],FOV=None):
        '''
        Purpose: Download all AIA channels data for the closest epoch of the time in configfile, processed them and make cutout for the selected region.
        '''
        if FOV is None:
            FOV = np.array(ast.literal_eval(self.config['SelectEvent']['FOV']))
        if FOV.shape != (2,2): Exception("%% SunX_error : Incorrect FOV. It should be an 2D array [[xy_bot_left],[xy_top_right]]")
        config = self.config
        data_dir = config['SelectEvent']['data_dir']
        if os.path.isdir(data_dir) is False:raise Exception("%% SunX_error : Directory not exist- "+data_dir)

        aia_dir = os.path.join(data_dir,'AIA','AllChnsCutouts')
        if os.path.isdir(aia_dir) is False: os.mkdir(aia_dir)

        datetim = datetime.strptime(self.config['SelectEvent']['event_dateTime'],'%Y-%m-%dT%H:%M:%S')
        start_time = Time(config['SelectEvent']['event_dateTime'], scale='utc', format='isot')
        aia = Fido.search(a.Time(start_time - 15*u.s, start_time + 15*u.s), a.Instrument.aia)
        for i in range(len(AIA_filters)):
            aia_wl = aia['vso'][aia['vso']["Wavelength"][:, 0] == AIA_filters[i]* u.AA]
            aia_times = Time(aia_wl['Start Time'].value)
            aia_final_file = aia_wl[aia_wl["Start Time"] == nearest_time(aia_times,start_time)]
            downloaded_files = Fido.fetch(aia_final_file, path=aia_dir)
            t = str(aia_final_file["Start Time"].value[0]).split(' ')
            t1 = t[1].split(':')
            afname = glob.glob(os.path.join(aia_dir,'aia_lev1_'+str(AIA_filters[i])+'a_'+t[0][0:4]+'_'+t[0][5:7]+'_'+t[0][8:10]+'t'+t1[0]+'_'+t1[1]+'_'+t1[2].split('.')[0]+'_*_image_lev1.fits'))[0]
            #afname = os.path.basename(afname)

            #convert lavel-1.5
            m = sunpy.map.Map(afname)
            m_updated_pointing = update_pointing(m)
            m_observer_fixed = fix_observer_location(m_updated_pointing)
            m_registered = register(m_updated_pointing)
            #m_normalized = normalize_exposure(m_registered) #not part of level 1.5 “prep” data. It will convert the data in unit DN/px/ s.
            os.remove(afname)
            afname = os.path.join(aia_dir,'AIA_lev1.5_'+ind_4(AIA_filters[i])+'_'+format('%0.4d'%datetim.year)+format('%0.2d'%datetim.month)+format('%0.2d'%datetim.day)+'T'+format('%0.2d'%datetim.hour)+format('%0.2d'%datetim.minute)+format('%0.2d'%datetim.second)+'.fits')
            m_registered.save(afname,overwrite='True')
            #Get cutout
            Make_AIAXRTcutout(instrument = 'AIA', DataDir=aia_dir,image=os.path.basename(afname),left_bot_arcsec=FOV[0],right_top_arcsec=FOV[1],OutDir=aia_dir,Plot=False)
            os.remove(afname)

    def select_region(self,FOV=None,OutFile = None,choose_FOV_center = False, FOV_xy_width = [200,200]):
        '''
        Purpose: Select the regions(s) from the full disk AIA and HMI data to produced cutouts of these regions.
        '''
        
        if FOV is None:
            FOV = np.array(ast.literal_eval(self.config['SelectEvent']['FOV'])) 
        if choose_FOV_center is True: FOV = None
         
        datetim = datetime.strptime(self.config['SelectEvent']['event_dateTime'],'%Y-%m-%dT%H:%M:%S')
        adir = os.path.join(self.config['SelectEvent']['data_dir'],'AIA')       
        afile = os.path.join(adir,'AIA_lev1.5_'+ind_4(self.config['SelectEvent']['aia_filter'])+'_'+format('%0.4d'%datetim.year)+format('%0.2d'%datetim.month)+format('%0.2d'%datetim.day)+'T'+format('%0.2d'%datetim.hour)+format('%0.2d'%datetim.minute)+format('%0.2d'%datetim.second)+'.fits') 
        if os.path.isfile(afile) is False: raise Exception("%% SunX_error : AIA level-1.5 data not exist (run get_data and data_processing) - "+afile)
        hdir = os.path.join(self.config['SelectEvent']['data_dir'],'HMI')
        hfile = os.path.join(hdir,'HMI_lev1.5_LOSmag'+'_'+format('%0.4d'%datetim.year)+format('%0.2d'%datetim.month)+format('%0.2d'%datetim.day)+'T'+format('%0.2d'%datetim.hour)+format('%0.2d'%datetim.minute)+format('%0.2d'%datetim.second)+'.fits')
        if os.path.isfile(hfile) is False:raise Exception("%% SunX_error : HMI level-1.5 data not exist (run get_data and data_processing) - "+hfile)
        a = sunpy.map.Map(afile)
        h = sunpy.map.Map(hfile)

        hdir_ = os.path.join(hdir,'cutouts')
        if os.path.isdir(hdir_) is False: os.mkdir(hdir_)
        if OutFile is None: OutFile = os.path.join(hdir_,'HMIcut_lev1.5_region_'+ind_3(1)+'.fits')

        fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': a})
        AllAxes=plt.gcf().get_axes()

        h.plot_settings['cmap'] = "hmimag"
        h.plot_settings['norm'] = plt.Normalize(-1500, 1500)

        a.plot(axes=AllAxes[0],clip_interval=(1, 99.9)*u.percent)
        #h.plot(axes=AllAxes[0],autoalign=True, alpha=0.3) 
        #AllAxes[0].set_title('HMI overlaid on AIA')
        AllAxes[0].set_title('Select FOV and close the window')

        out_file = open(os.path.join(adir,'SelectedRegion.dat'),'w')
        out_file.write('#select_region()\n')
        out_file.write('#region_no, Selection_x0(arcsec), Selection_x1, Selection_y0,Selection_y1, Non-zero pixcels, Area (cm2)\n')
        if FOV is not None:
            FOV = np.array(FOV)
            print("%% SunX_message :Creating cutouts for FOV -> [["+format('%0.1f'%FOV[0][0])+','+format('%0.1f'%FOV[0][1])+'], ['+format('%0.1f'%FOV[1][0])+','+format('%0.1f'%FOV[1][1])+']]')
            if FOV.shape == (2,2):
                bottom_left = SkyCoord(FOV[0][0]*u.arcsec,FOV[0][1]*u.arcsec, frame = a.coordinate_frame)
                top_right =   SkyCoord(FOV[1][0]*u.arcsec,FOV[1][1]*u.arcsec, frame = a.coordinate_frame)
                sub_a = a.submap(bottom_left=bottom_left, top_right=top_right)
                adir_ = os.path.join(adir,'cutouts')
                if os.path.isdir(adir_) is False: os.mkdir(adir_)
                sub_a.save(os.path.join(adir_,'AIAcut_lev1.5_region_'+ind_3(1)+'.fits'),overwrite=True)

                bottom_left = SkyCoord(FOV[0][0]*u.arcsec,FOV[0][1]*u.arcsec, frame = h.coordinate_frame)
                top_right =   SkyCoord(FOV[1][0]*u.arcsec,FOV[1][1]*u.arcsec, frame = h.coordinate_frame)
                sub_h = h.submap(bottom_left=bottom_left, top_right=top_right)
                #hdir_ = os.path.join(hdir,'cutouts')
                #if os.path.isdir(hdir_) is False: os.mkdir(hdir_)
                #if OutFile is None: OutFile = os.path.join(hdir_,'HMIcut_lev1.5_region_'+ind_3(1)+'.fits')
                sub_h.save(OutFile,overwrite=True)
                out_file.write('%d\t'%1)
                out_file.write('%d\t'%FOV[0][0])
                out_file.write('%d\t'%FOV[1][0])
                out_file.write('%d\t'%FOV[0][1])
                out_file.write('%d\t'%FOV[1][1])
                n_pix = len(np.where(sub_a.data > 0)[0])
                area = n_pix * 0.6 * 0.6 * (725*1.0e5)**2
                out_file.write('%d\t'%n_pix)
                out_file.write('%e\n'%area)

                print("%% SunX_message: Cutouts are saved.")
            else: Exception("%% SunX_error : Incorrect FOV. It should be an 2D array [[xy_bot_left],[xy_top_right]]")
        else: cid = fig[0].canvas.mpl_connect('button_press_event', lambda event: select_ar_onclick(event, fig,AllAxes,out_file,a,h,outfile_dir=self.config['SelectEvent']['data_dir'],choose_FOV_center = choose_FOV_center, FOV_xy_width =FOV_xy_width))
        plt.show()
        out_file.close()
        plt.close('all')
        print(OutFile) 
        self.hfname_cutout = os.path.basename(OutFile)
        self.hfname = os.path.basename(hfile)
        self.afname = os.path.basename(afile)
    def j_b_lff(self,bz0, z, alpha1=0.0,seehafer=False,sub_b00=False, sub_plan=False):
    
        '''
        Python implementation of J.R.Costa j_b_lff.pro extrapolation routine, which is available within SSW.
        -- Biswajit, Aug.30.2023
        '''
    
        nx1, ny1 = bz0.shape
        nz = len(z)
        b00=0.0
        if sub_b00 and sub_plan:
            raise Exception("%% j_b_lff_error : You cannot set both keyword: sub_b00 and sub_plan together!!'")
        if sub_b00:
            b00 = np.mean(bz0)
        if sub_plan:
            x = np.arange(0, len(bz0[:,0,0]))
            y = np.arange(0, len(bz0[0,:,0]))
            X, Y = np.meshgrid(x, y)
            Z = bz0
            rbf = Rbf(X, Y, Z, function="quintic")
            b00=rbf(X, Y)
        if seehafer:
            nx = 2 * nx1
            ny = 2 * ny1
            bz0e = np.zeros((nx, ny))
            bz0e[:nx1, :ny1] = bz0 - b00
            bz0e[nx1:, :ny1] = - np.rot90(bz0,1).T
            bz0e[:nx1, ny1:] = - np.rot90(bz0.T,1)
            bz0e[nx1:, ny1:] = - np.rot90(bz0e[:nx1, ny1:], 1).T
        else:
            nx = nx1 ; ny= ny1
            bz0e = bz0-b00
        kx = 2 * np.pi * np.concatenate([np.arange(nx // 2+1,dtype=np.float32),np.flip(-1-np.arange(nx-nx//2-1,dtype=np.float32))])/nx
        ky = 2 * np.pi * np.concatenate([np.arange(ny//2+1,dtype=np.float32),np.flip(-1-np.arange(ny-ny//2-1,dtype=np.float32))])/ny
        if abs(alpha1) >= 1.0:
            print('The magnitude of alpha is too big! ')
            print('|alpha| should be less than 1.')
            return None
        alpha=alpha1
        print('alpha=',-alpha)
        #kx=np.outer(np.ones([1,ny],dtype=int),kx)
        #ky=np.outer(ky,np.ones([nx],dtype=int))
        kx=np.outer(kx,np.ones([ny],dtype=int))    
        ky=np.outer(np.ones([1,nx],dtype=int),ky)

        fbz0 = np.fft.fftn(bz0e, norm='forward')
        kz=np.sqrt(np.maximum(kx**2 + ky**2 - alpha**2, 0))  # Positive solutions, see Nakagawa e Raadu p. 132
        ex__ = kz**2 + alpha**2
        #ex__[ex__ < kx[0,1]**2] = kx[0,1]**2
        ex__[ex__ < kx[1,0]**2] = kx[1,0]**2
        argx = fbz0 * (-1j) * (kx * kz - alpha * ky) / ex__
        ex__ = kz**2 + alpha**2
        #ex__[ex__ < ky[1,0]**2] = ky[1,0]**2
        ex__[ex__ < ky[0,1]**2] = ky[0,1]**2
        argy = fbz0 * (-1j) * (ky * kz + alpha * kx) / ex__
        bx = np.zeros((nx1, ny1, nz))
        by = np.zeros((nx1, ny1, nz))
        bz = np.zeros((nx1, ny1, nz))
        for j in range(nz):
            bx[:, :, j] = np.real(np.fft.ifftn(argx * np.exp(-kz * z[j]),norm='forward')[:nx1, :ny1])
            by[:, :, j] = np.real(np.fft.ifftn(argy * np.exp(-kz * z[j]),norm='forward')[:nx1, :ny1])
            bz[:, :, j] = np.real(np.fft.ifftn(fbz0 * np.exp(-kz * z[j]),norm='forward')[:nx1, :ny1]) + b00
        return bx, by, bz
    
    def gx_bz2lff(self,Bz, Nz=None, dr=None, alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False, OutFileName = None, OutDir = None):
        '''
        Python implementation of gx_bz2lff.pro routine available in SSW to run j_b_lff.pro
        
        Inputs:
            Bz -> 2D array of LOS magnetogram
            Nz -> z_pix_hight in pixcel unit
            dr -> [dx,dy,dz] #voxel length in arcsec for each directions
            alpha1 = alpha for llf extrapolation.
            OutFileName -> if None then the output B-cube will be store in a name 'PF_Bcube.fits'.
            OutDir -> directory where the OutFile will store. If None, default directory is data/HMI/PF 

        #Example to test the python implementation of extrapolation routine
        from scipy.io import readsav
        import subprocess
        import package as fld
        
        m = fld.fieldextrap(configfile='config.dat')
        
        ## create LOZ magnetogram
        x=np.linspace(-10,10, num=100)
        y=np.linspace(-10,10, num=100)
        x, y = np.meshgrid(x, y)
        bz0 = 50*(np.exp(-0.1*(x-5)**2-0.1*(y-5)**2) - np.exp(-0.1*(x+5)**2-0.1*(y+5)**2))
        bcube = m.gx_bz2lff(bz0, Nz=50, dr=[0.5,0.5,0.5], alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection="3d")
        ax2.plot_surface(x, y, bcube[:,:,30,2])
        
        #plot the idl outputs obtained by runing 'run_idl_jb_lff.pro' (run it from idl)
        #idl_command="ssw -e "+"'"+"run_idl_jb_lff,x="+str(list(x))+",y="+list(y)+",bz0="+list(bz0)+"'"
        #spectr = subprocess.check_output([idl_command],shell=True)
        sav_data = readsav('test_idl_out.sav')#read idl output
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(sav_data['x'], sav_data['y'], sav_data['bcube'][2,30,:,:])
        plt.show()

        -- Biswajit, Aug.30.2023
        '''
    
        sz = Bz.shape
        Nx,Ny = sz[0],sz[1]
        N = [Nx,Ny]
        alpha1= -alpha1 #To match with the IDL results. 
        if len(sz) >= 4 : Bz = np.reshape(Bz[:,:,:,1],(Nx,Ny,sz[3]))
        if len(sz) >= 3 : Bz = np.reshape(Bz[:,:,sz[2]],(Nx,Ny))
        if Nz is None: Nz = min(Nx, Ny) #;Nz specifies the height of the extrapolation in pixel unit
        N.extend([Nz])
        if dr is None: dr = [1.0, 1.0, 1.0] #;dr represents voxel size in each direction (arcsec) 
        z = np.arange(Nz) * dr[2] / dr[0]
        bxout, byout, bzout = self.j_b_lff(Bz, z, alpha1=alpha1, seehafer=seehafer, sub_b00=sub_b00, sub_plan=sub_plan)
        Bcube = np.zeros((Nx, Ny, Nz, 3))
        Bcube[:, :, :, 0] = bxout
        Bcube[:, :, :, 1] = byout
        Bcube[:, :, :, 2] = bzout
        if OutFileName is None: OutFileName = 'PF_Bcube' 
        if OutDir is None: OutDir = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','PF')
        if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
        hdul = fits.HDUList()
        hdul.append(fits.ImageHDU(Bcube,name='Bcube'))
        hdul[0].header.append(('dr', str(dr)))
        hdul[0].header.append(('Nz',Nz))
        hdul[0].header.append(('alpha',alpha1))
        hdul[0].header.append(('seehafer',seehafer))
        hdul[0].header.append(('sub_b00',sub_b00))
        hdul.writeto(os.path.join(OutDir,OutFileName+'.fits'),overwrite=True)
        #Note: difference between IDL and This version: Bx and By are interchanged
        return Bcube
    '''
    def field_interp(self,Vx, Vy, Vz, x, y, z, coord):
        ndim = len(Vx.shape)
        q  = np.zeros(ndim,dtype=float)
        xp = coord[0]
        yp = coord[1]
        if ndim == 3:zp = coord[2]
        else: zp = 0.0
        #; ----------------------------------------
        #;  Locate i0 and j0, indexes of
        #;  cell to which xp,yp belong to
        #; ----------------------------------------
    
        try:i0 = np.where(xp < x)[0][0]
        except:i0 = 0
        try:j0 = np.where(yp < y)[0][0]
        except:j0=0
        if ndim == 3:
            try:k0 = np.where(zp < z)[0][0]
            except:k0=0
        else : k0 = 0
        #Interpolate
        if ndim == 2:
            #; -- interpolate Vx --
    
            Vx_x = np.interp(xp,x,Vx[:,j0])
            Vx_y = np.interp(yp,Vx[i0,:])
            q[0] = Vx_x + Vx_y - Vx[i0,j0]
    
            #; -- interpolate Vy --
    
            Vy_x = np.interp(xp,x,Vy[:,j0])
            Vy_y = np.interp(yp,y,Vy[i0,:])
    
            q[1] = Vy_x + Vy_y - Vy[i0,j0]
    
        if ndim == 3:
            #; -- interpolate Vx --
             
            Vx_x = np.interp(xp,x,Vx[:,j0,k0])
            Vx_y = np.interp(yp,y,Vx[i0,:,k0])
            Vx_z = np.interp(zp,z,Vx[i0,j0,:])
            
            #
            #Vx_xf = interp1d(x,Vx[:,j0,k0],fill_value="extrapolate") ; Vx_x = Vx_xf(xp)
            #Vx_yf = interp1d(y,Vx[i0,:,k0],fill_value="extrapolate") ; Vx_y = Vx_yf(yp)
            #Vx_zf = interp1d(z,Vx[i0,j0,:],fill_value="extrapolate") ; Vx_z = Vx_zf(zp)
            #
            q[0] = Vx_x + Vx_y + Vx_z - 2.0*Vx[i0,j0,k0]
            #q[0] = interpn((x,y,z), Vx, np.array(coord),bounds_error=False,fill_value=0)

            #; -- interpolate Vy --
            
            Vy_x = np.interp(xp,x,Vy[:,j0,k0])
            Vy_y = np.interp(yp,y,Vy[i0,:,k0])
            Vy_z = np.interp(zp,z,Vy[i0,j0,:])
            
            #
            #Vy_xf = interp1d(x,Vy[:,j0,k0],fill_value="extrapolate") ; Vy_x = Vy_xf(xp)
            #Vy_yf = interp1d(y,Vy[i0,:,k0],fill_value="extrapolate") ; Vy_y = Vy_yf(yp)
            #Vy_zf = interp1d(z,Vy[i0,j0,:],fill_value="extrapolate") ; Vy_z = Vy_zf(zp)
            #
            q[1] = Vy_x + Vy_y + Vy_z - 2.0*Vy[i0,j0,k0]
            #q[1] = interpn((x,y,z), Vy, np.array(coord),bounds_error=False,fill_value=0)

            #; -- interpolate Vz --
            Vz_x = np.interp(xp,x,Vz[:,j0,k0])
            Vz_y = np.interp(yp,y,Vz[i0,:,k0])
            Vz_z = np.interp(zp,z,Vz[i0,j0,:])
            #
            #Vz_xf = interp1d(x,Vz[:,j0,k0],fill_value="extrapolate") ; Vz_x = Vz_xf(xp)
            #Vz_yf = interp1d(y,Vz[i0,:,k0],fill_value="extrapolate") ; Vz_y = Vz_yf(yp)
            #Vz_zf = interp1d(z,Vz[i0,j0,:],fill_value="extrapolate") ; Vz_z = Vz_xf(zp)
            #
            
            q[2] = Vz_x + Vz_y + Vz_z - 2.0*Vz[i0,j0,k0]
            #q[2] = interpn((x,y,z), Vz, np.array(coord),bounds_error=False,fill_value=0)
        return q
    
    def field_line(self,Vx, Vy, Vz, x, y, z, seed, method="RK2", maxstep=None,minstep=None, step=None, tol=None):
        #
        ##This is the python version of 'field_line.pro' originally 
        ##written by A. Mignone, which is available within PLUTO package.
    
        #PURPOSE: Given a 2 or 3D vector field (Vx, Vy) or (Vx, Vy, Vz) computes the 
        #      field line passing through the point (seed) [xseed, yseed, zseed].
        #      The line is computed by solving a system of ODE of the form
        #      
        #        dx/dt = Vx(x,y,z)
        #        dy/dt = Vy(x,y,z)
        #        dz/dt = Vz(x,y,z)
        #       
        #      Integration stops when either the domain boundaries are reached or 
        #      the max number of iteration is exceeded.
    
        #ARGUMENTS:
    
        #  Vx,Vy,Vz: 3D arrays giving the three vector components. In 2D, both Vz
        #            and z must be scalars and equal to 0.
    
        #  x,y,z:    1D coordinate arrays on top of which Vx, Vy and Vz are defined.
        #            In 2D, set z to be 0.0
    
        #  seed:     a 3-element array giving the point coordinates through which the
        #            field line goes. 
    
        #  pnt_list: on output, in contains 2D array giving the field line coordinates
        #            {x,y,z} = {pnt_list[0,*], pnt_list[1,*], pnt_list[2,*]} (in 3D) or
        #            {x,y }  = {pnt_list[0,*], pnt_list[1,*]} (in 2D)
    
        #KEYWORDS:
    
        #  step:   a scalar giving the initial step size. Default is (mean) grid spacing.
    
        #  method: a string giving the integration method. The possible choices are:
    
        #           "RK2"   explicit, fixed step, 2nd order Runge-Kutta methods.
        #           "RK4"   explicit, fixed step, 4th order Runge-Kutta methods.
        #           "BS23"  explicit, adaptive stepsize Runge-Kutta-Fehlberg of order 
        #                   3 with local truncation error based on a 2nd-order accurate
        #                   embedded solution.
        #           "CK45"  explicit, adaptive stepsize Cask-Karp of order 
        #                   5 with local truncation error based on a 4th-order accurate
        #                   embedded solution.
    
        #          The default is "RK2". Use an adaptive stepsize integrator
        #          when the field line does not close on itself inside the domain.
        #
    
        #  maxstep: a scalar giving the maximum allowed integration step.
        #           Default is 100*step.
    
        #  minstep: a scalar giving the minimum allowed integration step. 
        #           Default is 0.05*step.
    
        #  tol:   a scalar value giving the relative tolerance during adaptive stepsize
        #         integration. It is ignored for fixed step-size integration (such as RK2, RK4)
         
        ##EXAMPLE:
        #
        #from fieldextrap import*
        #from scipy.io import readsav
        #import package as fld
        #m = fld.fieldextrap(configfile='config.dat')
        #magnetogram = readsav('/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/HMI/HMIcut_newOBS_XBP001_Extrapolated.sav') #load the extrapolated B.
        #
        #bx1=np.transpose(magnetogram['Bx'], (1, 2, 0))
        #bx2=np.transpose(magnetogram['By'], (1, 2, 0))
        #bx3=np.transpose(magnetogram['Bz'], (1, 2, 0))
        #origin = 0
        #x1 = origin+np.arange(len(bx1[:,0,0]))#*0.36250001 #;in Mm (z-axis)
        #x2 = origin+np.arange(len(bx1[0,:,0]))#*0.36250001
        #x3 = np.arange(len(bx1[0,0,:]))#*0.36250001 #(x-axis)
        #x11_a = [225,225,10,50,100]
        #y11_a = [229,229,20,50,100]
        #for i in range(1):
        #    fig = plt.subplots(1, 1, figsize=(6, 6))
        #    AllAxes=plt.gcf().get_axes()
        #    plt.imshow(bx3[:,:,0],origin='lower')
        #    x11=x11_a[i]#229
        #    y11=y11_a[i]#225
        #    loc = np.array([x11,y11,0.0])# * 0.36250001
        #    pl = m.field_line(bx1,bx2,bx3, x1, x2, x3, seed=loc, method="RK2")
        #    px=pl[0,:]
        #    py=pl[1,:]
        #    pz=pl[2,:]
        #    bz_z = []
        #    igx=m.value_locate (x1, px)
        #    igy=m.value_locate (x2, py)
        #    igz=m.value_locate (x3, pz)
        #    bz_z = bx3[igx,igy,igz] 
        #    ind = np.where(pz > 0.0)[0]
        #    fig2 = plt.subplots(1, 1, figsize=(6, 6))
        #    plt.plot(bz_z[ind],pz[ind],'--')
        #    #plt.plot(pz[ind],'--');plt.plot(py[ind],'--');plt.plot(px[ind],'--')
        #    plt.show()

        #------------------------
        #--- Biswajit Aug-31-2023
        #------------------------
        #
        sz = np.array(Vx.shape)
        seed = np.array(seed)
        ndim = len(sz)
        nx = sz[0] ; ny = sz[1]
        if ndim ==2:
            nz   = 0
            norm = 1.0/np.sqrt(Vx**2 + Vy**2 + 1.e-18) #;  Normalization factor for vector field.
        elif ndim == 3:
            nz = sz[2]
            norm = 1.0/np.sqrt(Vx*Vx + Vy*Vy + Vz*Vz +1.e-18)
            Vz = Vz*norm
        Vx = Vx*norm #;  Normalize vector field to 1, Only direction can change.
        Vy = Vy*norm
    
        npt = np.zeros(ndim, dtype=int)
        dbeg = np.zeros(ndim, dtype=float)
        dend = np.zeros(ndim, dtype=float)
        L = np.zeros(ndim, dtype=float)
    
        #; ------------------------------------------
        #;  Get domain sizes. 
        #;  Take the initial and final coordinates
        #;  slightly larger to allow a seed to be 
        #;  specified on the boundary. 
        #; ------------------------------------------
    
    
        dbeg[0] = x[0]  - 0.51*(x[1] - x[0])  #Get domain sizes. Take the initial and final coordinates
                                              #slightly larger to allow a seed to be specified on the boundary.
        dend[0] = x[-1] + 0.51*(x[-1] - x[-2])
        L[0]    = dend[0] - dbeg[0]
    
        dbeg[1] = y[0]  - 0.51*(y[1] - y[0])
        dend[1] = y[-1] + 0.51*(y[-1] - y[-2])
        L[1]    = dend[1] - dbeg[1]
    
        if ndim == 3:
           dbeg[2] = z[0]    - 0.51*(z[1] - z[0])
           dend[2] = z[-1] + 0.51*(z[-1] - z[-2])
           L[2]    = dend[2] - dbeg[2]
    
        condt0 = np.less(seed,dend) == np.greater(seed,dbeg)
        if np.any(condt0 == False) == True: Exception("%% SunX_error : Given seed point falls outside grid range.") #Make sure initial seed point falls 
    
        max_steps = 16384
        max_fail = 1024
                                                                                               #inside the computational domain.
        xfwd = np.zeros([ndim, max_steps], dtype=float) #coordinates for forward  integration
        xbck = np.zeros([ndim, max_steps], dtype=float) #coordinates for backward integration
        xk0 = np.zeros(ndim, dtype=float)
        xk1 = np.zeros(ndim, dtype=float)
        xk2 = np.zeros(ndim, dtype=float)
        xk3 = np.zeros(ndim, dtype=float)
        xk4 = np.zeros(ndim, dtype=float)
        xk5 = np.zeros(ndim, dtype=float)
    
        xfwd[:,0] = seed[:] #Set initial conditions
        xbck[:,0] = seed[:]
        #;  Check keywords: step, method and tolerance
        if step is None: step = min((dend - dbeg)/sz)
        if tol is None: tol = 1.0e-6
        if maxstep is None: maxstep = 100*step
        if minstep is None: minstep = 0.05*step
        tol = tol*max(L) #tolerance factor should scale down actual to domain size.
    
        #; --------------------------------------------------------------------
        #;  Set the coefficients for adaptive step size integration:
        #;  Cash-Karp 45 (CK45) and Bogacki-Shampine 23 (BS23).
        #;  Taken from:
        #;  http://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        #; --------------------------------------------------------------------
    
        if method == "CK45":
            b1 = 37.0/378.0  ; b2 = 0.0 ; b3 = 250.0/621.0
            b4 = 125.0/594.0 ; b5 = 0.0 ; b6 = 512.0/1771.0
    
            bs1 = 2825.0/27648.0  ; bs2 = 0.0           ; bs3 = 18575.0/48384.0
            bs4 = 13525.0/55296.0 ; bs5 = 277.0/14336.0 ; bs6 = 0.25
    
            a21 = 0.2
            a31 = 3.0/40.0       ; a32 = 9.0/40.0
            a41 = 0.3            ; a42 = -0.9        ; a43 = 1.2
            a51 = -11.0/54.0     ; a52 = 2.5         ; a53 = -70.0/27.0    ; a54 = 35.0/27.0
            a61 = 1631.0/55296.0 ; a62 = 175.0/512.0 ; a63 = 575.0/13824.0 ; a64 = 44275.0/110592.0 ; a65 = 253.0/4096.0
    
        if method == "BS23":
            b1  = 2.0/9.0  ; b2  = 1.0/3.0 ; b3  = 4.0/9.0 ; b4  = 0.0
            bs1 = 7.0/24.0 ; bs2 = 1.0/4.0 ; bs3 = 1.0/3.0 ; bs4 = 1.0/8.0
    
            a21 = 0.5
            a31 = 0.0     ; a32 = 0.75
            a41 = 2.0/9.0 ; a42 = 1.0/3.0 ; a43 = 4.0/9.0
    
        for s in [-1,1]: #Integrate Backward (s=-1) and Forward (s=1)
            dh = s*step
            inside_domain = 1
            k             = 0
            kfail         = 0
            while (inside_domain == 1) and (k < max_steps):  # attempt to integrate from k to k+1.
                dh = s*min([abs(dh),maxstep]) #;  restrict dh to lie between minstep and maxstep
                dh = s*max([abs(dh),minstep])
                #if (abs(dh)/minstep <= 1.0):print("Minimum step reached")
                #if (abs(dh)/maxstep >= 1.0): print)"Maximum step reached")
    
                xk0 = xfwd[:,k] #; -- set initial condition 
    
                #; ----------------------------------------------------------
                #;   Explicit Runge-Kutta method with 2nd order accuracy.
                #;   Fixed step size. Requires 2 function evaluations.
                #; ----------------------------------------------------------
                if method == "RK2":
                    k = k+1
                    k1  = self.field_interp(Vx, Vy,Vz, x, y, z, xk0)
                    xk1 = xk0 + 0.5*dh*k1
                    k2 = self.field_interp(Vx, Vy, Vz, x, y, z, xk1)
                    xfwd[:,k] = xk0 + dh*k2
    
                #; ----------------------------------------------------------
                #;   Explicit Runge-Kutta method with 4th order accuracy.
                #;   Fixed step size. Requires 4 function evaluations.
                #; ----------------------------------------------------------
                if method == "RK4":
                    k = k+1
    
                    k1  = self.field_interp(Vx, Vy, Vz, x, y, z,xk0)
                    xk1 = xk0 + 0.5*dh*k1
    
                    k2  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + 0.5*dh*k2
    
                    k3 = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*k3
    
                    k4 = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xfwd[:,k] = xk0 + dh*(k1 + 2.0*(k2 + k3) + k4)/6.0
    
                #; ---------------------------------------------------------------
                #;  Explicit Runge-Kutta-Fehlberg pair (2,3) with adaptive 
                #;  step size. It is also known as Bogacki-Shampine and provide
                #;  third-order accuracy using a total of 4 function evaluations.
                #; ---------------------------------------------------------------
                
                if method == "BS23": #; -- use BS23
    
                    k1  = self.field_interp(Vx, Vy, Vz, x, y, z,xk0)
                    xk1 = xk0 + dh*a21*k1
    
                    k2  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*(a31*k1 + a32*k2)
    
                    k3  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*(a41*k1 + a42*k2 + a43*k3)
    
                    k4  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk3 = xk0 + dh*(b1*k1 + b2*k2 + b3*k3 + b4*k4)
    
                    xk2 = xk0 + dh*(bs1*k1 + bs2*k2 + bs3*k3 + bs4*k4)
    
                    #; ---- compute error ----
    
                    err = max(abs(xk3 - xk2))/tol
                    if (err < 1.0) or (abs(dh)/minstep < 1.0): #; -- accept step
                        k      = k + 1
                        err    = max([err,1.e-12])
                        dhnext = 0.9*abs(dh)*err**(-0.3333)
                        dhnext = min([dhnext,3.0*abs(dh)])
                        dh     = s*dhnext
                        xfwd[:,k] = xk3
                    else:
                        dh = 0.9*s*abs(dh)*err**(-0.5)
                        if (kfail > max_fail): raise Exception("%% SunX_error : Too many failures!")
    
                #; ---------------------------------------------------------------
                #;  Cash-Karp fifth-order method using a (4,5) pair.
                #;  Provides adaptive step-size control with monitoring of local 
                #;  truncation error. It requires 6 function evaluations.
                #; ---------------------------------------------------------------
                if method == "CK45": 
    
                    k1  = self.field_interp(Vx, Vy, Vz, x, y, z,xk0)
                    xk1 = xk0 + dh*a21*k1
    
                    k2  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*(a31*k1 + a32*k2)
    
                    k3  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*(a41*k1 + a42*k2 + a43*k3)
    
                    k4  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*(a51*k1 + a52*k2 + a53*k3 + a54*k4)
    
                    k5  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk1 = xk0 + dh*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
    
                    k6  = self.field_interp(Vx, Vy, Vz, x, y, z,xk1)
                    xk5 = xk0 + dh*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
    
                    xk4 = xk0 + dh*(bs1*k1 + bs2*k2 + bs3*k3 + bs4*k4 + bs5*k5 + bs6*k6)
    
                    #; ---- compute error ----
    
                    err = max(abs(xk5 - xk4))/tol
                    if (err < 1.0) or (abs(dh)/minstep < 1.0): #; -- accept step
                        k      = k + 1
                        err    = max([err,1.e-12])
                        dhnext = 0.9*abs(dh)*err**(-0.2)
                        dhnext = min([dhnext,3.0*abs(dh)])
                        dh     = s*dhnext
                        xfwd[:,k] = xk5
                    else:
                        dh = 0.9*s*abs(dh)*err**(-0.25)
                        if (kfail > max_fail):raise Exception("%% SunX_error : Too many failueres!")
                inside_domain = 1
                condt = np.greater(xfwd[:,k],dbeg[:]) == np.less(xfwd[:,k],dend[:])
                if np.any(condt == False) == True: inside_domain = 0 #Check whether we're still inside the domain.
            if s == -1:
                xbck  = xfwd
                k_bck = k
        k_fwd = k
        if k_fwd >= (max_steps-1): print("! Max number of iteration exceeded")
        if k_bck >= (max_steps-1): print("! Max number of iteration exceeded")
    
        print("Method: ",method,   "; Forward steps: "+format('%d'%k_fwd)+"; Bckward steps: "+format('%d'%k_bck))
        #; "; tol = "+format('%e'%tol))
    
        #; --------------------------------------------
        #;         return arrays
        #; --------------------------------------------
    
        #;xfield = [reverse(REFORM(xbck(0,0:k_bck))),REFORM(xfwd(0,0:k_fwd))]
        #;yfield = [reverse(REFORM(xbck(1,0:k_bck))),REFORM(xfwd(1,0:k_fwd))]
        #;zfield = [reverse(REFORM(xbck(2,0:k_bck))),REFORM(xfwd(2,0:k_fwd))]
    
        npt = k_bck + k_fwd + 2
    
        pnt_list = np.zeros([ndim, npt],dtype=float)
        for nd in range(ndim):
            pnt_list[nd, :] = np.concatenate((xbck[nd, 0:k_bck+1][::-1], xfwd[nd, 0:k_fwd+1]))
    
        return pnt_list
    
    def value_locate(self,xbins, x):
        # Find the indices where x falls into bins. This is a similar function as IDL value_locate().
        indices = np.searchsorted(xbins, x)
        return indices.astype(int)
    
    def trace_scan(self,l, args): #for parallel run
        Bx = args[0] ; By = args[1]; Bz = args[2]
        x1 = args[3]; x2=args[4]; x3=args[5]
        nonzero_px_loc = args[6]
        Seed_Zheight = args[7]
        loc = args[8]
        #for l in range(N_lines):
        if loc[0] is None: # seeds will be selected randomly from 'nonzero_px_loc' and 'Seed_Zheight'.
            #np.random.seed(seed)  
            ran_ind = np.random.randint(0, len(nonzero_px_loc[0]),size=1)[0]
            #np.random.seed(seed)
            seed_xyz = [nonzero_px_loc[0][ran_ind],nonzero_px_loc[1][ran_ind],np.random.randint(0,Seed_Zheight+1,size=1)[0]]
            #loc = [seed_xyz[0][l], seed_xyz[1][l], seed_xyz[2][l]]
            loc = [seed_xyz[0], seed_xyz[1], seed_xyz[2]]
        else: 
            seed_xyz = loc
        pl = self.field_line(Bx,By,Bz, x1,x2,x3, seed=loc, method="RK2")
        px=pl[0,:]
        py=pl[1,:]
        pz=pl[2,:]

        ind_ = np.where(pz > 0.0) #;Reject the lengths benith the photosphere
        px=px[ind_]
        py=py[ind_]
        pz=pz[ind_]

        #;Find the grid points of pl
        igx=self.value_locate (x1, px)
        igy=self.value_locate (x2, py)
        igz=self.value_locate (x3, pz)
        ind_ = np.where((igx < len(x1)) & (igy < len(x2)) &(igz < len(x3)) ) #Ignore the points outside
        igx = igx[ind_]; igy = igy[ind_]; igz = igz[ind_]
        status = 0
        nz = len(igz)
        if nz > 3 and igz[0] < 15 and igz[-1] <15: #consider the loops whiose both the footpoints are below 15 pixcels in z-direction. 
            #Store the loops those have more than 3 gridpoints and both the ends are 
            #within a height of two pixcels (.725 Mm, for HMI). This indicates we are 
            #considering only the close field lines.

            if igz[0] == igz[1] : igx = igx[1::] ; igy = igy[1::] ; igz = igz[1::] #exclude 1st point if same as second point
            if igz[-1] == igz[-2] : igx = igx[0:-1] ; igy = igy[0:-1] ; igz = igz[0:-1] #exclude last point if same as previous one

            #print('yes')
            #;locate the magneticfield on those points
            bx1_field=Bx[igx,igy,igz]
            bx2_field=By[igx,igy,igz]
            bx3_field=Bz[igx,igy,igz]
            status = 1
            return status, seed_xyz, igx, igy, igz, bx1_field,  bx2_field, bx3_field
        else:return 0, 0, 0, 0, 0, 0, 0, 0

    def multi_run_wrapper(self,l, args):
        return self.trace_scan(l, args)
    
    def trace_fields(self,Bcube,N_lines = 10,Seed_Zheight = 0, B_pos_thresold = 10, B_neg_thresold = -10,PlotAIA = False, AIA_Image = None, OutDir = None, OutFileName = None, Seeds_file = None,HMI_Mag = None, hmivmin=-500,hmivmax=500):
        
        #
        #Inputs: 
        #     Bcube -> 4D-array containing Bx[x,y,z], By[x,y,z], and Bz[x,y,x]
        #     N_lines -> Number of output fieldlines 
        #     Seed_Zheight -> Sheed_Zheight (in pixel unit) is the height of the seed selection volume. If 0 then seeds will be selected at the base.
        #     B_pos_thresold -> 
        #     B_neg_thresold -> 
        #     AIA_Image -> 
        #     OutDir -> 
        #     OutFileName -> 
        #     Seeds_file -> If given then the sheeds will be selected from a prestored file
        #     HMI_Mag -> 
        #     PlotAIA -> If true then only HMI contoure will overplot on HMI map for seed selections
        #
        
        data_dir = self.config['SelectEvent']['data_dir']
        if HMI_Mag is None: HMI_Mag = os.path.join(data_dir,'HMI','cutouts','HMIcut_lev1.5_region_01.fits')
        if PlotAIA is True and AIA_Image is None : AIA_Image = os.path.join(data_dir,'AIA','cutouts','AIAcut_lev1.5_region_01.fits')
        if OutDir is None: OutDir = os.path.join(data_dir,'HMI','PF'); 
        if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
        if  OutFileName is None: OutFileName = 'PF_loops_xyzB'

        hmap = sunpy.map.Map(HMI_Mag)

        if len(Bcube.shape) != 4: raise Exception("%% SunX_error: Incorrect dimension of Bcube!")
        Bx = Bcube[:,:,:,0]
        By = Bcube[:,:,:,1]
        Bz = Bcube[:,:,:,2]
        if Seed_Zheight > len(Bz[0,0,:])-1 : 
            print("%% SunX_massege : Seed_Zheight is more that the extrapolated volum. Set at Z-boundery!")
            Seed_Zheight = len(Bz[0,0,:])-1
        #origin = 0
        #arc2Mm = 0.725
        #dxx = hmap.scale.axis1.value*arc2Mm ; dyy = hmap.scale.axis2.value*arc2Mm 
        x1 = np.arange(len(Bz[:,0,0])) #in pixcel unit
        x2 = np.arange(len(Bz[0,:,0]))
        x3 = np.arange(len(Bz[0,0,:])) #same as x1

        
        if PlotAIA is True :
            amap = sunpy.map.Map(AIA_Image) 
            fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': amap})
            AllAxes[0].set_title('HMI overlaid on AIA')
            O = AllAxes[0].imshow(amap.data,origin='lower',interpolation='nearest',cmap=amap.cmap,norm=colors.PowerNorm(gamma=0.5))
            #Overplot HMI contures
            levels = [30,50, 100, 150, 300, 500, 1000] * u.Gauss
            levels = np.concatenate((-1 * levels[::-1], levels))
            bounds = AllAxes[0].axis()
            cset = hmap.draw_contours(levels, axes=AllAxes[0], cmap='seismic', alpha=0.5)
            AllAxes[0].axis(bounds)
        else: 
            fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': hmap})
            AllAxes=plt.gcf().get_axes()
            AllAxes[0].set_title('HMI overlaid on AIA')
            hmap.plot_settings['norm'].vmin = hmivmin
            hmap.plot_settings['norm'].vmax = hmivmax
            hmap.plot(axes = AllAxes[0])

        hdul = fits.HDUList()
        seed_xyz_all = np.zeros([N_lines,3])

        c1 = fits.Column(name='x', array=seed_xyz_all[:,0], format='K');
        c2 = fits.Column(name='y', array=seed_xyz_all[:,1], format='K');
        c3 = fits.Column(name='z', array=seed_xyz_all[:,2], format='K')
        hdul.append(fits.TableHDU.from_columns([c1, c2, c3],name='seeds_xyz'))
        hdul[0].header.append(('ref_image', os.path.basename(HMI_Mag)))
        hdul[0].header.append(('unit', 'pix'))
        #hdul[0].header.append(('dx', hmap.scale.axis1.value ,'arcsec/pix'))
        #hdul[0].header.append(('dy', hmap.scale.axis2.value ,'arcsec/pix'))
        #hdul[0].header.append(('dz', hmap.scale.axis1.value ,'arcsec/pix'))
        hdul[0].header.append(('dx', hmap.scale.axis1.to('arcsec/pix').value ,'arcsec/pix'))
        hdul[0].header.append(('dy', hmap.scale.axis2.to('arcsec/pix').value ,'arcsec/pix'))
        hdul[0].header.append(('dz', hmap.scale.axis1.to('arcsec/pix').value ,'arcsec/pix'))

        if Seeds_file is None:
            cid = fig[0].canvas.mpl_connect('button_press_event', lambda event: select_seed_regions(event, fig,AllAxes))
            plt.show()
            xi = xi_all[-2::] #obtained from util function
            yi = yi_all[-2::]
            bot = SkyCoord(xi[0]*u.arcsec,yi[0]*u.arcsec, frame = hmap.coordinate_frame)
            top = SkyCoord(xi[1]*u.arcsec,yi[1]*u.arcsec, frame = hmap.coordinate_frame)
            xi = [hmap.world_to_pixel(bot).x.value,hmap.world_to_pixel(top).x.value]
            yi = [hmap.world_to_pixel(bot).y.value,hmap.world_to_pixel(top).y.value]
            nonzero_px_loc = np.where(((hmap.data > B_pos_thresold) | (hmap.data < B_neg_thresold))) #[location of y, location of x]
            ind1 = np.where((nonzero_px_loc[1] >= xi[0]) & (nonzero_px_loc[1] <= xi[1]))
            ind2 = np.where((nonzero_px_loc[0] >= yi[0]) & (nonzero_px_loc[0] <= yi[1]))
            ind = list(np.intersect1d(ind1[0],ind2[0]))
            nonzero_px_loc = [nonzero_px_loc[0][ind],nonzero_px_loc[1][ind]]
            #ran_ind = list(np.random.randint(0, len(nonzero_px_loc[0]),size=N_lines))
            #seed_xyz = [nonzero_px_loc[0][ran_ind],nonzero_px_loc[1][ran_ind],np.random.randint(0,Seed_Zheight+1,size=N_lines)]
            loc = [None]*N_lines

            #Update some headers
            hdul[0].header.append(('bot_left_x', int(xi[0])))
            hdul[0].header.append(('bot_left_y', int(yi[0])))
            hdul[0].header.append(('top_right_x', int(xi[1])))
            hdul[0].header.append(('top_right_y', int(yi[1])))

        else: 
             nonzero_px_loc = None ; 
             Seed_Zheight = None 
             loc = np.loadtxt(Seeds_file,unpack=True) #To be read from an ASCII file
        if self.NumCore == 0: #Single core
            l_ind = 0       
            iteration = 0 
            while l_ind < N_lines:
                status, seed_xyz, igx, igy, igz, bx1_field,  bx2_field, bx3_field = self.trace_scan(l_ind, [Bx, By, Bz, x1, x2, x3, nonzero_px_loc,Seed_Zheight, loc])
                iteration += 1 
                if status == 1: 
                    seed_xyz_all[l_ind,:] = seed_xyz
                    #Writing to file
                    c1 = fits.Column(name='x', array=igx, format='K'); 
                    c2 = fits.Column(name='y', array=igy, format='K'); 
                    c3 = fits.Column(name='z', array=igz, format='K')
                    c4 = fits.Column(name='|Bx|',array=bx1_field,format='D')
                    c5 = fits.Column(name='|By|',array=bx2_field,format='D')
                    c6 = fits.Column(name='|Bz|',array=bx3_field,format='D')
                    hdul.append(fits.TableHDU.from_columns([c1, c2, c3,c4,c5,c6],name='L'+format('%d'%l_ind)))
                    l_ind += 1
                if l_ind > N_lines+5000 : 
                    print('SunX_message: Exceeding maximum iterations for number of loops. Do you want to continue (y/n)?')
                    inp = str(input())
                    if inp == 'y': l_ind = 0
                    else : break 
        else: #Multi processing #(Not working properly)
            print('Using Multi-process')
            l_ind = 0
            iteration = 0
            while l_ind < N_lines: #Note: If one process becomes slow, it will affect next iteration.
                iteration += 1
                pool = mulp.Pool(self.NumCore)
                l_ind = range(N_lines)
                resul = pool.map(partial(self.multi_run_wrapper, args=[Bx, By, Bz, x1, x2 ,x3, nonzero_px_loc, Seed_Zheight, loc]), l_ind)
                pool.close()
                print(resul[0])
                print(len(resul))
                status = resul[:][0] ; seed_xyz = resul[:][1] ; 
                igx = resul[:][2] ; igy = resul[:][3] ; igz = resul[:][4] ; 
                bx1_field = resul[:][5] ;  bx2_field = resul[:][6] ; bx3_field = resul[:][7] ;
                #Writing to file
                for ww in range(l_ind,l_ind+len(status)):
                    ind__ = ww - l_ind
                    if status[ind__] == 1 and l_ind <= N_lines:
                        seed_xyz_all[l_ind,:] = seed_xyz
                        c1 = fits.Column(name='x', array=igx[ind__], format='K');
                        c2 = fits.Column(name='y', array=igy[ind__], format='K');
                        c3 = fits.Column(name='z', array=igz[ind__], format='K')
                        c4 = fits.Column(name='|Bx|',array=bx1_field[ind__],format='D')
                        c5 = fits.Column(name='|By|',array=bx2_field[ind__],format='D')
                        c6 = fits.Column(name='|Bz|',array=bx3_field[ind__],format='D')
                        hdul.append(fits.TableHDU.from_columns([c1, c2, c3,c4,c5,c6],name='L'+format('%d'%l_ind)))
                        l_ind += 1
                if l_ind > N_lines+5000 :
                    print('SunX_message: Exceeding maximum iterations for number of loops. Do you want to continue (y/n)?')
                    inp = str(input())
                    if inp == 'y': l_ind = 0
                else : break 
        print('Total number of iterations : '+format('%d'%iteration))
        hdul[1].data['x'][:] = seed_xyz_all[:,0] #Update the seed points
        hdul[1].data['y'][:] = seed_xyz_all[:,1]
        hdul[1].data['z'][:] = seed_xyz_all[:,2]
 
        hdul.writeto(os.path.join(OutDir,OutFileName+'.fits'),overwrite=True)           
        return None
    '''

    def trace_fields(self,Bcube,N_lines = 10,Seed_Zheight = 0, min_footpoint_z = 15, B_pos_thresold = 10, B_neg_thresold = -10,PlotAIA = False, AIA_Image = None, OutDir = None, OutFileName = None, Seeds_file = None,HMI_Mag = None, hmivmin=-500,hmivmax=500,Select_seed_region = True,dN = 1000):
        
        #
        #Inputs: 
        #     Bcube -> 4D-array containing Bx[x,y,z], By[x,y,z], and Bz[x,y,x]
        #     N_lines -> Number of output fieldlines 
        #     Seed_Zheight -> Sheed_Zheight (in pixel unit) is the height of the seed selection volume. If 0 then seeds will be selected at the base.    
        #     min_footpoint_z -> All the traced fields foot point would below this height in pixcel unit. 
        #     B_pos_thresold -> 
        #     B_neg_thresold -> 
        #     AIA_Image -> 
        #     OutDir -> 
        #     OutFileName -> 
        #     Seeds_file -> If given then the sheeds will be selected from a prestored file
        #     HMI_Mag -> 
        #     PlotAIA -> If true then only HMI contoure will overplot on HMI map for seed selections
        #     Select_seed_region --> If True, then the seed-box from which the seeds will be selected randomly can be selected.
        #     #dN = 50 #Number of iteration in each YT streams.
        data_dir = self.config['SelectEvent']['data_dir']
        if HMI_Mag is None: HMI_Mag = os.path.join(data_dir,'HMI','cutouts','HMIcut_lev1.5_region_01.fits')
        hmap = sunpy.map.Map(HMI_Mag)
        if Select_seed_region is True:
            if PlotAIA is True and AIA_Image is None : AIA_Image = os.path.join(data_dir,'AIA','cutouts','AIAcut_lev1.5_region_01.fits')
        if OutDir is None: OutDir = os.path.join(data_dir,'HMI','PF'); 
        if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
        if  OutFileName is None: OutFileName = 'PF_loops_xyzB'

        if len(Bcube.shape) != 4: raise Exception("%% SunX_error: Incorrect dimension of Bcube!")

        #Bcube = np.nan_to_num(Bcube,nan=0) #remove the nan values to 0

        data = {}
        data['Bx'] = (Bcube[:,:,:,0],'gauss')
        data['By'] = (Bcube[:,:,:,1],'gauss')
        data['Bz'] = (Bcube[:,:,:,2],'gauss')

        dim_ = data["Bx"][0].shape
        pix_size = 1#0.3625 #in Mm
        dxx = dim_[0]*pix_size
        dyy = dim_[1]*pix_size
        dzz = dim_[2]*pix_size
        
        # Load the dataset in YT
        ds = yt.load_uniform_grid(
            data,
            data["Bx"][0].shape,
            magnetic_unit="gauss",
            length_unit = 'Mm',
            bbox=np.array([[0, dxx], [0, dyy], [0, dzz]]),
            periodicity=(True, True, True),
            dataset_name = 'field',
            axis_order = ("x","y","z"),
        )
        
        LOS_mag = np.array(ds.index.grids[0]['stream', 'Bz'][:,:,0])

        if Seed_Zheight > len(data['Bz'][0][0,0,:])-1 : 
            print("%% SunX_massege : Seed_Zheight is more that the extrapolated volum. Set at Z-boundery!")
            Seed_Zheight = len(Bz[0,0,:])-1
        
        if Select_seed_region is True :
            if PlotAIA is True :
                amap = sunpy.map.Map(AIA_Image) 
                fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': amap})
                AllAxes[0].set_title('HMI overlaid on AIA')
                O = AllAxes[0].imshow(amap.data,origin='lower',interpolation='nearest',cmap=amap.cmap,norm=colors.PowerNorm(gamma=0.5))
                #Overplot HMI contures
                levels = [30,50, 100, 150, 300, 500, 1000] * u.Gauss
                levels = np.concatenate((-1 * levels[::-1], levels))
                bounds = AllAxes[0].axis()
                cset = hmap.draw_contours(levels, axes=AllAxes[0], cmap='seismic', alpha=0.5)
                AllAxes[0].axis(bounds)
            else: 
                fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': hmap})
                AllAxes=plt.gcf().get_axes()
                AllAxes[0].set_title('HMI overlaid on AIA')
                hmap.plot_settings['norm'].vmin = hmivmin
                hmap.plot_settings['norm'].vmax = hmivmax
                hmap.plot(axes = AllAxes[0])

        hdul = fits.HDUList()
        seed_xyz_all = np.zeros([N_lines,3])

        c1 = fits.Column(name='x', array=seed_xyz_all[:,0], format='K');
        c2 = fits.Column(name='y', array=seed_xyz_all[:,1], format='K');
        c3 = fits.Column(name='z', array=seed_xyz_all[:,2], format='K')
        hdul.append(fits.TableHDU.from_columns([c1, c2, c3],name='seeds_xyz'))
        hdul[0].header.append(('ref_image', os.path.basename(HMI_Mag)))
        hdul[0].header.append(('unit', 'pix'))
        #hdul[0].header.append(('dx', hmap.scale.axis1.value ,'arcsec/pix'))
        #hdul[0].header.append(('dy', hmap.scale.axis2.value ,'arcsec/pix'))
        #hdul[0].header.append(('dz', hmap.scale.axis1.value ,'arcsec/pix'))
        hdul[0].header.append(('dx', hmap.scale.axis1.to('arcsec/pix').value ,'arcsec/pix'))
        hdul[0].header.append(('dy', hmap.scale.axis2.to('arcsec/pix').value ,'arcsec/pix'))
        hdul[0].header.append(('dz', hmap.scale.axis1.to('arcsec/pix').value ,'arcsec/pix'))

        if Seeds_file is None and Select_seed_region is True:
            cid = fig[0].canvas.mpl_connect('button_press_event', lambda event: select_seed_regions(event, fig,AllAxes))
            plt.show()
            xi = xi_all[-2::] #obtained from util function
            yi = yi_all[-2::]
            bot = SkyCoord(xi[0]*u.arcsec,yi[0]*u.arcsec, frame = hmap.coordinate_frame)
            top = SkyCoord(xi[1]*u.arcsec,yi[1]*u.arcsec, frame = hmap.coordinate_frame)
            xi = [hmap.world_to_pixel(bot).x.value,hmap.world_to_pixel(top).x.value]
            yi = [hmap.world_to_pixel(bot).y.value,hmap.world_to_pixel(top).y.value]
            nonzero_px_loc = np.where(((hmap.data > B_pos_thresold) | (hmap.data < B_neg_thresold))) #[location of y, location of x]
            ind1 = np.where((nonzero_px_loc[1] >= xi[0]) & (nonzero_px_loc[1] <= xi[1]))
            ind2 = np.where((nonzero_px_loc[0] >= yi[0]) & (nonzero_px_loc[0] <= yi[1]))
            ind = list(np.intersect1d(ind1[0],ind2[0]))
            nonzero_px_loc = [nonzero_px_loc[0][ind],nonzero_px_loc[1][ind]]
            #ran_ind = list(np.random.randint(0, len(nonzero_px_loc[0]),size=N_lines))
            #seed_xyz = [nonzero_px_loc[0][ran_ind],nonzero_px_loc[1][ran_ind],np.random.randint(0,Seed_Zheight+1,size=N_lines)]
            loc = [None]*N_lines

            #Update some headers
            hdul[0].header.append(('bot_left_x', int(xi[0])))
            hdul[0].header.append(('bot_left_y', int(yi[0])))
            hdul[0].header.append(('top_right_x', int(xi[1])))
            hdul[0].header.append(('top_right_y', int(yi[1])))

        elif Seeds_file is None and Select_seed_region is False:
            nonzero_px_loc = np.where((LOS_mag > B_pos_thresold) | (LOS_mag < B_neg_thresold)) #[location of y, location of x]
            loc = [None]*N_lines

            #Update some headers
            hdul[0].header.append(('bot_left_x', int(0)))
            hdul[0].header.append(('bot_left_y', int(0)))
            hdul[0].header.append(('top_right_x', len(hmap.data[0,:])))
            hdul[0].header.append(('top_right_y', len(hmap.data[:,0])))

        else: ##Not functional yet!!
             nonzero_px_loc = None ; 
             Seed_Zheight = None 
             loc = np.loadtxt(Seeds_file,unpack=True) #To be read from an ASCII file
        l_ind = 0       
        iteration = 0 
        #dN = 50 #Number of iteration in each YT streams.
        while l_ind < N_lines:
            ran_ind = np.random.randint(0, len(nonzero_px_loc[0]),size=dN)   
            seed_xyz = np.zeros([dN,3])
            seed_xyz[:,0] = nonzero_px_loc[0][ran_ind] ; seed_xyz[:,1] = nonzero_px_loc[1][ran_ind]; 
            seed_xyz[:,2] = np.random.randint(0,Seed_Zheight+1,size=1)[0]

            streamlines = Streamlines(ds,seed_xyz, xfield='Bx', yfield='By', zfield='Bz',get_magnitude=True,)        
            streamlines.integrate_through_volume()
            strm_ind = 0
            for stream in streamlines.streamlines:
                if l_ind < N_lines:
                    stream = stream[np.all(stream != 0.0, axis=1)]
                    stream = stream[stream[:,0] <= dim_[0]]
                    stream = stream[stream[:,1] <= dim_[1]]
                    if stream.shape[0] > 3: #consider atleast 3 points for the field-lines.
                        stream_z = stream[:,2]#position in Z-directions
                        if stream_z[0] < min_footpoint_z and stream_z[-1] < min_footpoint_z: #Consider only the lines looks like closed.
                            fields_ = ds.find_field_values_at_points([("stream", "Bx"), ("stream", "By"),("stream", "Bz")],stream)
                            bx1_field = fields_[0].to_value('G') 
                            bx2_field = fields_[1].to_value('G')
                            bx3_field = fields_[2].to_value('G')
                            seed_xyz_all[l_ind,:] = seed_xyz[strm_ind,:]  

                            #Writing to file
                            c1 = fits.Column(name='x', array=np.array(stream[:,0]), format='D'); 
                            c2 = fits.Column(name='y', array=np.array(stream[:,1]), format='D'); 
                            c3 = fits.Column(name='z', array=np.array(stream[:,2]), format='D')
                            c4 = fits.Column(name='|Bx|',array=bx1_field,format='D')
                            c5 = fits.Column(name='|By|',array=bx2_field,format='D')
                            c6 = fits.Column(name='|Bz|',array=bx3_field,format='D')
                            hdul.append(fits.TableHDU.from_columns([c1, c2, c3,c4,c5,c6],name='L'+format('%d'%l_ind)))
                            l_ind += 1
                strm_ind += 1
            print('Repeating YT: '+format('%d'%(iteration)))
            print('Success lines#: '+format('%d'%(l_ind))) 
            iteration += 1
            if l_ind > N_lines+5000 : 
                print('SunX_message: Exceeding maximum iterations for number of loops. Do you want to continue (y/n)?')
                inp = str(input())
                if inp == 'y': l_ind = 0
                else : break 
        print('Total number of iterations : '+format('%d'%iteration))
        hdul[1].data['x'][:] = seed_xyz_all[:,0] #Update the seed points
        hdul[1].data['y'][:] = seed_xyz_all[:,1]
        hdul[1].data['z'][:] = seed_xyz_all[:,2]
 
        hdul.writeto(os.path.join(OutDir,OutFileName+'.fits'),overwrite=True)           
        return None

 
    def loop_WCS(self, hmap, loop_coor_pix):
        '''
            Convert the loop coordinate in pixcel unit to WCS coordinate associated with given HMI-map
            Inputs:
                hmap -> Sunpy map object of HMI magnetogram
                loop_coor_pix -> [[x-pix],[y-pix],[z-pix]]
        '''

        hcc_frame = sunpy.coordinates.Heliocentric(observer=hmap.observer_coordinate, obstime=hmap.date)
        hmap_origin = SkyCoord(Tx=0*u.arcsec, Ty=0*u.arcsec, frame=hmap.coordinate_frame)
        origin_pix = hmap.world_to_pixel(hmap_origin)
        scale_factor = (hmap.scale.axis1*u.pix).to('Mm',equivalencies=sunpy.coordinates.utils.solar_angle_equivalency(hmap.observer_coordinate)) / u.pix  #size of pixcel
       
        #Convert to WCS coordinates
        x = (loop_coor_pix[0]*u.pix - origin_pix.x) * scale_factor
        y = (loop_coor_pix[1]*u.pix - origin_pix.y) * scale_factor            
        z = (loop_coor_pix[2]*u.pix) * scale_factor + hmap.rsun_meters
        coord = SkyCoord(x=x,y=y,z=z,frame=hcc_frame)
        return coord

    def plot_seeds(self,HMI_Mag = None, ExtPo_loop_par_file = None):
        
        if HMI_Mag is None: HMI_Mag = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','cutouts','HMIcut_lev1.5_region_01.fits')
        if ExtPo_loop_par_file is None: ExtPo_loop_par_file = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','PF','PF_loops_xyzB.fits')
        with fits.open(ExtPo_loop_par_file) as hdul:
            seeds = hdul[1].data
        hmap = sunpy.map.Map(HMI_Mag)
        
        plt.imshow(hmap.data,origin='lower');
        plt.plot(seeds['y'],seeds['x'],'*r')
        plt.show()

    def plot_loop(self,zlim=[0,100],HMI_Mag = None, PlotAIA = True, AIA_Image=None, N_loops=None, ExtPo_loop_par_file = None, mode='2D',lw=1,color='C0',alpha=0.7,OplotSeeds=False,plot_loop_WCS = True,Proj_corr =False,Loop_indx = None,StoreOutputs = False):
        '''
        Purpose: Plot the loops on HMI/AIA images 
        Inputs: 
            HMI_Mag -> HMI fits file
            AIA_Image -> AIA file
            N_loops -> Number of loops to be calculated. Note that the draw loops may be smaller than this number, 
                       depending on if some loops are ignorable, it will not plot.
            ExtPo_loop_par_file -> Extrpolated loop parameters fits file, obtained from traced field function.
            mode -> mode of plot either 2D or 3D
            OplotSeeds -> If True, seed points will be overplotted
            plot_loop_WCS -> (Only for mode = 2D) If True, then the loop coordinates will be transformed to WCS coordinate and then will plot. 
                             In this case the loops looks smoother, because the values will not be integer pixcels.
            Loop_inds -> 1D array of loop-index which will plot. If not given, all loops will be plotted

        Biswajit, Nov-10-2023
        '''

        if HMI_Mag is None: HMI_Mag = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','cutouts','HMIcut_lev1.5_region_01.fits')
        if ExtPo_loop_par_file is None: ExtPo_loop_par_file = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','PF','PF_loops_xyzB.fits')
        if AIA_Image is None and PlotAIA == True: AIA_Image = os.path.join(self.config['SelectEvent']['data_dir'],'AIA','cutouts','AIAcut_lev1.5_region_01.fits')
        hmap = sunpy.map.Map(HMI_Mag)
        if PlotAIA == True:amap = sunpy.map.Map(AIA_Image)

        if mode == '3D':
            fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': '3d'})
            AllAxes=plt.gcf().get_axes()
            ax = AllAxes[0]
            #AllAxes[0].set_title('HMI overlaid on AIA')
            #3D plot
            data = hmap.data
            x_g = np.arange(len(hmap.data[0,:]))
            y_g = np.arange(len(hmap.data[:,0]))
            extent = [x_g[0], x_g[-1],y_g[0], y_g[-1]]
            X, Y = np.meshgrid(np.linspace(x_g[0], x_g[-1],len(data[0,:])), np.linspace(y_g[0], y_g[-1],len(data[:,0])))
            ax.set_zlim(zlim); 
            ax.set_xlim([x_g[0], x_g[-1]]); ax.set_ylim([y_g[0], y_g[-1]])
            cset = ax.contourf(X, Y, data, 200, zdir='z', offset=0.0, cmap=hmap.cmap,extent = extent,origin='lower',alpha=0.7,zorder=0)
            '''
            #Plot AIA
            data = amap.data
            x_g = np.arange(len(amap.data[0,:]))
            y_g = np.arange(len(amap.data[:,0]))
            extent = [x_g[0], x_g[-1],y_g[0], y_g[-1]]
            X, Y = np.meshgrid(np.linspace(x_g[0], x_g[-1],len(data[0,:])), np.linspace(y_g[0], y_g[-1],len(data[:,0])))
            #ax.set_zlim([0.,40.]); 
            ax.set_xlim([x_g[0], x_g[-1]]); ax.set_ylim([y_g[0], y_g[-1]])
            cset = ax.contourf(X, Y, data, 200, zdir='z', offset=0.0, cmap=amap.cmap,extent = extent,origin='lower',alpha=0.6,zorder=0)
            '''
        else:
            fig = plt.subplots(1, 1, figsize=(6, 6),subplot_kw={'projection': hmap})
            AllAxes=plt.gcf().get_axes()
            ax = AllAxes[0]
            if PlotAIA == True: 
                AllAxes[0].set_title('HMI overlaid on AIA')
                #O = AllAxes[0].imshow(amap.data,origin='lower',interpolation='nearest',cmap=amap.cmap,norm=colors.PowerNorm(gamma=0.5))
                amap.plot(axes=AllAxes[0],autoalign=True)
                #Overplot HMI contures
                levels = [50, 100, 150, 300, 500, 1000] * u.Gauss
                levels = np.concatenate((-1 * levels[::-1], levels))
                bounds = AllAxes[0].axis()
                cset = hmap.draw_contours(levels, axes=AllAxes[0], cmap='seismic', alpha=0.5)
                AllAxes[0].axis(bounds)
            else: hmap.plot(axes = AllAxes[0])
            #hmap.plot()
        with fits.open(ExtPo_loop_par_file) as hdul:
            nloops = len(hdul)
            if N_loops is not None:
                if N_loops > nloops: 
                     print('%SunX_message: N_loops is exeeding maximum available loops. Set to maximum value of ->'+format('%d'%nloops))
                else: nloops = N_loops
            seeds = hdul[1].data
            if Loop_indx is None: loop_sll = range(2,nloops)
            else: loop_sll = 2+np.array(Loop_indx)
            for l in loop_sll:
                data = hdul[l].data
                if Proj_corr is False:
                    if plot_loop_WCS is True and mode == '2D':data_WCS = self.loop_WCS(hmap,[data['y'],data['x'],data['z']])
                    if mode == '3D':
                        #if plot_loop_WCS is True: ax.plot_coord(data_WCS, color='C0',lw=1)
                        ax.plot3D(data['y'], data['x'],data['z'],zorder=500,alpha=alpha,color=color)
                        if OplotSeeds is True: ax.plot3D(seeds['y'],seeds['x'],seed['z'],'*r',alpha=alpha)
                    else: 
                        if plot_loop_WCS is True: ax.plot_coord(data_WCS, color=color,lw=lw)
                        else: ax.plot(data['y'],data['x'],color = color,alpha=alpha,lw=lw)
                        if OplotSeeds is True: ax.plot(seeds['y'],seeds['x'],'*r',alpha=alpha)
                else: ax.plot(data['xp'],data['yp'],color = color,alpha=alpha,lw=lw)
        print('Plotted loop numbers: ',len(loop_sll))
        if StoreOutputs is True: fig[0].savefig(ExtPo_loop_par_file[0:-5]+'.png')
        plt.show()
        

    def get_loop_parameters(self,N_loops=None, ExtPo_loop_par_file = None,define_Bbase = 1, Min_Z = 1,OutDir = None, OutFileName = None,dxdydz_Mm = None):

        '''
        define_Bbase -> height of coronal hase in Mm
        Min_Z -> Minimum loop height in Mm
        dxdydz_Mm -> [dx,dy,dz] ; pixcel size in Mm for the magnetic data cube
        '''
        data_dir = self.config['SelectEvent']['data_dir']
        if OutDir is None: OutDir = os.path.join(data_dir,'HMI','PF');
        if os.path.isdir(OutDir) is False: os.mkdir(OutDir)
        if  OutFileName is None: OutFileName = 'PF_loops_AvgPar'

        #define_Bbase = int(define_Bbase) # define coronal base. default = 3 -> 3*0.3625 = 1.09 Mm

        f_out=open(os.path.join(OutDir,OutFileName+'.dat'),'w')
        f_out.write('#Output of fieldextrap.get_loop_parameters()"\n')
        f_out.write('#Area(cm2)='+''+'\n')#+str(Area_AIA193)+'\n')
        f_out.write("#loop_ind, Full_length(Mm), LOS_B_left_foot(G), LOS_B_right_foot, ABS_B_left_foot, ABS_B_right_foot, Avg_B(G), ABS_B_left_defined_base, ABS_B_right_defined_base, Avg_B_above_base, Loop_height\n")

        if ExtPo_loop_par_file is None: ExtPo_loop_par_file = os.path.join(self.config['SelectEvent']['data_dir'],'HMI','PF','PF_loops_xyzB.fits')
        with fits.open(ExtPo_loop_par_file) as hdul:
            nloops = len(hdul)
            data_all = hdul
            headers = data_all[0].header
            if dxdydz_Mm is None : dxpix = headers['dx']*self.arcs2Mm ; dypix = headers['dy']*self.arcs2Mm ; dzpix = headers['dz']*self.arcs2Mm 
            else: dxpix = dxdydz_Mm[0] ; dypix = dxdydz_Mm[1] ; dzpix = dxdydz_Mm[1]
            if N_loops is not None: nloops = N_loops
            loop_index_all = [] 
            length_all = []
            LOS_B_left_foot_all = []
            LOS_B_right_foot_all = []
            ABS_B_left_foot_all = []
            ABS_B_right_foot_all = []
            Avg_B_full_all = []
            ABS_B_left_defined_base_all = []
            ABS_B_right_defined_base_all = []
            Avg_B_above_base_all = []
            loop_height_all = []
            if nloops > len(data_all) : 
                nloops = len(data_all)
                print('%SunX_massege : Exceeding maximum available loops. Set to maximum number of loops ->'+format('%d'%nloops))
            for l in range(2,nloops):
                data = data_all[l].data
                x = data['x']*dxpix ; y = data['y']*dypix ; z = data['z']*dzpix #in Mm
                Bx = data['|Bx|'] ; By = data['|By|'] ; Bz = data['|Bz|']
                if max(z) >= Min_Z: #Consider the loops whose minimum height is Min_Z (Mm).
                    ind_above_base = np.where(z >= define_Bbase)[0] #grid indices above defined base height
                    delta_length = np.sqrt((x[1::]-x[0:-1])**2 + (y[1::]-y[0:-1])**2 + (z[1::]-z[0:-1])**2) #in Mm
                    length = np.sum(delta_length) #Mm
                    mod_B = np.sqrt(Bx**2 + By**2 + Bz**2)
                    dl_half = delta_length/2.0
                    dl1 = np.zeros(len(Bz))
                    dl1[1:-1] = dl_half[0:-1]+dl_half[1::] #consider each grid point as a middle point
                    dl1[0] = dl_half[0] ; dl1[-1] = dl_half[-1] #for end two points consider only delta_length/2
         
                    Avg_B_full = np.sum(mod_B*dl1) / np.sum(dl1)
                    
                    loop_sl_no = int(l-2)
                    
                    f_out.write('%d\t'%loop_sl_no)
                    f_out.write('%0.4f\t'%length)
                    f_out.write('%0.4f\t'%Bz[0])
                    f_out.write('%0.4f\t'%Bz[-1])
                    f_out.write('%0.4f\t'%mod_B[0])
                    f_out.write('%0.4f\t'%mod_B[-1])
                    f_out.write('%0.4f\t'%Avg_B_full)
                   
                    loop_index_all += [loop_sl_no]
                    length_all += [length] 
                    LOS_B_left_foot_all += [Bz[0]]
                    LOS_B_right_foot_all += [Bz[-1]]
                    ABS_B_left_foot_all += [mod_B[0]]
                    ABS_B_right_foot_all += [mod_B[-1]]
                    Avg_B_full_all += [Avg_B_full]
                    if len(ind_above_base) > 2: #If minimum 2 grid point exist then only estimate
                        Avg_B_above_base = np.sum(mod_B[ind_above_base]*dl1[ind_above_base]) / np.sum(dl1[ind_above_base])
                        f_out.write('%0.4f\t'%mod_B[ind_above_base][0])
                        f_out.write('%0.4f\t'%mod_B[ind_above_base][-1])
                        f_out.write('%0.4f\t'%Avg_B_above_base)
                        ABS_B_left_defined_base_all += [mod_B[ind_above_base][0]]
                        ABS_B_right_defined_base_all += [mod_B[ind_above_base][-1]]
                        Avg_B_above_base_all += [Avg_B_above_base]
                    else:
                        f_out.write('%d\t'%0)
                        f_out.write('%d\t'%0)
                        f_out.write('%d\t'%0)
                        ABS_B_left_defined_base_all += [0]
                        ABS_B_right_defined_base_all += [0]
                        Avg_B_above_base_all += [0]
                    f_out.write('%f\n'%max(z))
                    loop_height_all += [max(z)]
                    #print(length,Avg_B_full)
            f_out.close()     

        return loop_index_all, length_all, LOS_B_left_foot_all, LOS_B_right_foot_all, ABS_B_left_foot_all, ABS_B_right_foot_all, Avg_B_full_all, ABS_B_left_defined_base_all, ABS_B_right_defined_base_all, Avg_B_above_base_all,loop_height_all


'''
python3 setup.py install

import simar as fld

m = fld.fieldextrap(configfile='config.dat')
m.get_data()
m.data_processing()
m.select_region()
m.field_line()
''' 
