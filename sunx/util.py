from astropy.time import Time
import os,glob
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sunpy.map
import sunpy.coordinates
import astropy.table
from sunpy.net import Fido, attrs as a
from scipy.ndimage.interpolation import shift
from scipy.io import readsav
from astropy.io import fits

import os
import subprocess
import warnings
from collections import OrderedDict
import tempfile
import xml.etree.ElementTree as ET
import xml.dom.minidom as xdm


def pardir():
    return '/Users/bmondal/BM_Works/softwares/SunX_main/'

def nearest_time(items, pivot):
    return min(items, key=lambda x: abs((x - pivot).value))

def ind_6(ch):
    ch = str(ch)
    if len(ch) == 5: ch = '0'+ch
    if len(ch) == 4: ch = '00'+ch
    if len(ch) == 3: ch = '000'+ch
    if len(ch) == 2: ch = '0000'+ch
    if len(ch) == 1: ch = '00000'+ch
    return ch

def ind_5(ch):
    ch = str(ch)
    if len(ch) == 4: ch = '0'+ch
    if len(ch) == 3: ch = '00'+ch
    if len(ch) == 2: ch = '000'+ch
    if len(ch) == 1: ch = '0000'+ch
    return ch

def ind_4(ch):
    ch = str(ch)
    if len(ch) == 3: ch = '0'+ch
    if len(ch) == 2: ch = '00'+ch
    if len(ch) == 1: ch = '000'+ch
    return ch

def ind_3(x):
    x = str(x)
    if len(x) == 2: x = '0'+x
    if len(x) == 1: x = '00'+x
    return(x)

def ind_2(x):
    x = str(x)
    if len(x) == 1: x = '0'+x
    return(x)

def find_str(File, string):
    for no,line in enumerate(File):
        if line.find(string) != -1:
            return line,no
    return None,None

def search_and_replace_line(file_path, search_text_start, replace_text):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.startswith(search_text_start):
            lines[i] = replace_text + '\n'  # Add a newline character to maintain file formatting
            found = True
            break

    if not found:
        print("Search text not found in the file.")

    with open(file_path, 'w') as file:
        file.writelines(lines)


#=== Function to save/load python ductionary ===
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
'''
def get_center_pix(event,fig,configfile,FOV_xy_width):
    global center_xi,center_yi
    if event.button == 3: #Right click
        ix, iy = event.xdata, event.ydata
        b=event.inaxes.format_coord(ix,iy).split( )#transform to projected frame.
        center_xi = int(b[0][0:-1])
        center_yi = int(b[1][0:-1])
        FOV_new = '[['+format('%0.2f'%(center_xi-(FOV_xy_width[0]/2))+','+format('%0.2f'%(center_yi-(FOV_xy_width[1]/2)))+'],'+'['+format('%0.2f'%(center_xi+(FOV_xy_width[0]/2))+','+format('%0.2f'%(center_yi+(FOV_xy_width[1]/2)))+']]'
        search_and_replace_line(configfile, 'FOV', FOV_new)                    
'''
nstart = 1
region=1
xi_all_trans=[];yi_all_trans=[]
def select_ar_onclick(event,fig,AllAxes,f_Area,amap,hmap,outfile_dir='./',choose_FOV_center = False, FOV_xy_width = [200,200]):
     global nstart,region,xi_all_trans,yi_all_trans
     if event.button == 3: #Right click
         ix, iy = event.xdata, event.ydata
         b=event.inaxes.format_coord(ix,iy).split( )#transform to projected frame.
         ix_t = int(b[0][0:-1].replace('−', '-'))
         iy_t = int(b[1][0:-1].replace('−', '-'))
         if choose_FOV_center is False:
             print(b)
             if nstart == 1:
                 print('Region: ',region)
             #xi_all += [ix]
             #yi_all += [iy]
             xi_all_trans += [ix_t]
             yi_all_trans += [iy_t]
             if nstart == 2:
                 print('Do you want to redraw (y/n):')
                 input1 = input()
                 if input1 == 'n':
                     f_Area.write('%d\t'%region)
                     f_Area.write('%d\t'%xi_all_trans[0])
                     f_Area.write('%d\t'%xi_all_trans[1])
                     f_Area.write('%d\t'%yi_all_trans[0])
                     f_Area.write('%d\t'%yi_all_trans[1])

                     bottom_left = SkyCoord(xi_all_trans[0]*u.arcsec,yi_all_trans[0]*u.arcsec, frame = amap.coordinate_frame)
                     top_right = SkyCoord(xi_all_trans[1]*u.arcsec,yi_all_trans[1]*u.arcsec, frame = amap.coordinate_frame)
                     sub_temp = amap.submap(bottom_left=bottom_left, top_right=top_right)
                     bottom_left = SkyCoord(xi_all_trans[0]*u.arcsec,yi_all_trans[0]*u.arcsec, frame = hmap.coordinate_frame)
                     top_right = SkyCoord(xi_all_trans[1]*u.arcsec,yi_all_trans[1]*u.arcsec, frame = hmap.coordinate_frame)
                     sub_temp_hmi = hmap.submap(bottom_left=bottom_left, top_right=top_right)
                     #sub_temp.plot(axes=AllAxes[1])
                     n_pix = len(np.where(sub_temp.data > 0)[0])
                     area = n_pix * 0.6 * 0.6 * (725*1.0e5)**2
                     f_Area.write('%d\t'%n_pix)
                     f_Area.write('%e\n'%area)
                     print(n_pix,area,xi_all_trans,yi_all_trans)
                     #save the AIA cutout in local directory
                     adir = os.path.join(outfile_dir,'AIA') 
                     adir_ = os.path.join(adir,'cutouts')
                     if os.path.isdir(adir_) is False: os.mkdir(adir_)
                     sub_temp.save(os.path.join(adir_,'AIAcut_lev1.5_region_'+ind_3(region)+'.fits'),overwrite=True)
                     '''
                     fig__ = plt.figure()
                     ax__ = fig__.add_subplot(projection=amap)
                     sub_temp.plot(axes=ax__)
                     fig__.savefig(os.path.join(adir_,'AIAcut_lev1.5_region_'+ind_3(region)+'.png'))
                     '''
                     #save the HMI cutout in local directory
                     hdir = os.path.join(outfile_dir,'HMI')
                     hdir_ = os.path.join(hdir,'cutouts')
                     if os.path.isdir(hdir_) is False: os.mkdir(hdir_)
                     sub_temp_hmi.save(os.path.join(hdir_,'HMIcut_lev1.5_region_'+ind_3(region)+'.fits'),overwrite=True)
                     '''
                     fig__ = plt.figure()
                     ax__ = fig__.add_subplot(projection=hmap)
                     sub_temp_hmi.plot(axes=ax__)
                     fig__.savefig(os.path.join(hdir_,'HMIcut_lev1.5_region_'+ind_3(region)+'.png'))
                     '''


                     region += 1
                     xi_all_trans=[]; yi_all_trans=[]
                 nstart = 0
                 #fig.canvas.draw() #redraw the figure
                 fig[0].savefig(os.path.join(outfile_dir,'SelectedRegion.png'))
             nstart += 1
         else: 
             xi_all_trans = [ix_t - FOV_xy_width[0]/2, ix_t + FOV_xy_width[0]/2]
             yi_all_trans = [iy_t - FOV_xy_width[1]/2, iy_t + FOV_xy_width[1]/2]
             f_Area.write('%d\t'%region)
             f_Area.write('%d\t'%xi_all_trans[0])
             f_Area.write('%d\t'%xi_all_trans[1])
             f_Area.write('%d\t'%yi_all_trans[0])
             f_Area.write('%d\t'%yi_all_trans[1]) 

             bottom_left = SkyCoord(xi_all_trans[0]*u.arcsec,yi_all_trans[0]*u.arcsec, frame = amap.coordinate_frame)
             top_right = SkyCoord(xi_all_trans[1]*u.arcsec,yi_all_trans[1]*u.arcsec, frame = amap.coordinate_frame)
             sub_temp = amap.submap(bottom_left=bottom_left, top_right=top_right)
             bottom_left = SkyCoord(xi_all_trans[0]*u.arcsec,yi_all_trans[0]*u.arcsec, frame = hmap.coordinate_frame)
             top_right = SkyCoord(xi_all_trans[1]*u.arcsec,yi_all_trans[1]*u.arcsec, frame = hmap.coordinate_frame)
             sub_temp_hmi = hmap.submap(bottom_left=bottom_left, top_right=top_right)
             #sub_temp.plot(axes=AllAxes[1])
             n_pix = len(np.where(sub_temp.data > 0)[0])
             area = n_pix * 0.6 * 0.6 * (725*1.0e5)**2
             f_Area.write('%d\t'%n_pix)
             f_Area.write('%e\n'%area)
             print(n_pix,area,xi_all_trans,yi_all_trans)
             #save the AIA cutout in local directory
             adir = os.path.join(outfile_dir,'AIA')
             print('Choose the FOV center from mouse right-click')
             adir_ = os.path.join(adir,'cutouts')
             if os.path.isdir(adir_) is False: os.mkdir(adir_)
             sub_temp.save(os.path.join(adir_,'AIAcut_lev1.5_region_'+ind_3(1)+'.fits'),overwrite=True)
             #save the HMI cutout in local directory
             hdir = os.path.join(outfile_dir,'HMI')
             hdir_ = os.path.join(hdir,'cutouts')
             if os.path.isdir(hdir_) is False: os.mkdir(hdir_)
             sub_temp_hmi.save(os.path.join(hdir_,'HMIcut_lev1.5_region_'+ind_3(1)+'.fits'),overwrite=True)

nstart = 1
xi_all = []; yi_all = []
def select_seed_regions(event,fig,AllAxes):
    global nstart,xi_all,yi_all
    
    if event.button == 3: #Right click
        ix, iy = event.xdata, event.ydata
        b=event.inaxes.format_coord(ix,iy).split( )#transform to projected frame.
        ix_t = int(b[0][0:-1].replace('−', '-'))
        iy_t = int(b[1][0:-1].replace('−', '-'))
        print(ix_t, iy_t)
        xi_all += [ix_t]
        yi_all += [iy_t]
        if nstart == 2:
            print('Do you want to redraw (y/n):')
            input1 = input()
            if input1 == 'n':
                #xi_all=[]; yi_all=[]
                plt.close()
            nstart = 0
        nstart += 1

def randomp(pow,N,range_x=None,seed=None):
    '''
    ; PURPOSE:
    ;       Generates an array of random numbers distributed as a power law. Adupted from randomp.pro in IDL.
    ; INPUTS:
    ;       pow:  Exponent of power law.
    ;               The pdf of X is f_X(x) = A*x^pow, low <= x <= high
    ;               ASTRONOMERS PLEASE NOTE:  
    ;               pow is little gamma  = big gamma - 1 for stellar IMFs.
    ;       N:    Number of elements in generated vector.
    ;
    ; OPTIONAL INPUT KEYWORD PARAMETER:
    ;       RANGE_X:  2-element vector [low,high] specifying the range of 
    ;               output X values; the default is [5, 100].
    ;
    ; OPTIONAL INPUT-OUTPUT KEYWORD PARAMETER:
    ;       SEED:    Seed value for RANDOMU function.    As described in the 
    ;               documentation for RANDOMU, the value of SEED is updated on 
    ;               each call to RANDOMP, and taken from the system clock if not
    ;               supplied.   This keyword can be used to have RANDOMP give 
    ;               identical results on different runs.
    ; OUTPUTS:
    ;       X:    Vector of random numbers, distributed as a power law between
    ;               specified range
    ; EXAMPLE:
    ;       Create a stellar initial mass function (IMF) with 10000 stars
    ;       ranging from 0.5 to 100 solar masses and a Salpeter slope.  Enter:
    ;       MASS = randomp(-2.35, 10000, range_x=[0.5,100])

            MASS = randomp(-2.0, 1000000, range_x=[0.002,0.02]) 
            a1,b1,_=plt.hist(MASS,bins=50)  
            MASS = randomp(-1.5, 1000000, range_x=[0.002,0.02]) 
            a,b,_=plt.hist(MASS,bins=20,alpha=0.5)  
            plt.close('all') 
            plt.yscale('log') 
            plt.xscale('log') 
            plt.step((b1[1::]+b1[0:-1])/2.0,a1) 
            plt.step((b[1::]+b[0:-1])/2.0,a) 
            plt.show()                                                                                                                                                                      
    ; MODIFICATION HISTORY:
            Biswajit, 29.01.2022
    '''
    if range_x == None: range_x=[5,100]
    elif range_x != None:
        if len(range_x) != 2:
            message='Error - RANGE_X keyword must be a 2 element vector'
            return print(message)

    pow1 = pow + 1.0
    lo = range_x[0]
    hi = range_x[1]
    if lo > hi:
        temp=lo
        lo=hi
        hi=tmp
    if seed is not None: np.random.seed(seed)
    #np.random.seed(3000)
    r = np.random.uniform(low=0.0, high=1.0, size=N)#  randomu(s, n )
    if pow != -1.0:
       norm = 1.0/(hi**pow1 - lo**pow1)
       expo = np.log10(r/norm + lo**pow1)/pow1
       x = 10.0**expo
    else :
       norm = 1.0/(np.log(hi) - np.log(lo))
       x = np.exp(r/norm + np.log(lo))
    return x

def LoopCoordinate2map(HMI_magnetogram_file,Loop_coordinate_files,Extrp_voxel_size = [0.3625,0.3625,0.3625],Plot=False,aia_file=None):
    '''
     Purpose: Convert extrapolated loop coordinate to astropy map.
     Inputs:
         HMI_magnetogram_file -> HMI magnetograp fits file name
         Loop_coordinate_files -> Array of extrapolated loop files, where x,y,z 
                                  coordinates of each loop stored in three column. 
         Extrp_voxel_size -> size of [x, y, z] pixcels in Mm as defined in extrapolation routine.
         Plot -> If True, the loops will be plooted on HMI and AIA images
         aia_file -> AIA image file. If not provided then will be downloaded from Fido 
    '''
    scale_factor_fieldlines = np.array(Extrp_voxel_size) * u.Mm / u.pix

    m_hmi = sunpy.map.Map(HMI_magnetogram_file) #HMI fits file
    hcc_frame = sunpy.coordinates.Heliocentric(observer=m_hmi.observer_coordinate, obstime=m_hmi.date)   
    m_hmi_origin = SkyCoord(Tx=0*u.arcsec, Ty=0*u.arcsec, frame=m_hmi.coordinate_frame)
    origin_pix = m_hmi.world_to_pixel(m_hmi_origin)
    scale_factor = (m_hmi.scale.axis1*u.pix).to('Mm',equivalencies=sunpy.coordinates.utils.solar_angle_equivalency(m_hmi.observer_coordinate)) / u.pix  #size of pixcel
    
    # These fieldline coordinates are physical offsets from the bottom 
    # left corner of the image using a defined spatial scale.
    # However, the HCC coordinate system is defined from the origin 
    # as defined by the point normal to the observer location.
    # Additionally the z-axis is defined relative to the center of the Sun.
    # Thus, we need to apply these offsets in all three axes. 

    fieldline_coords = []
    for filename in Loop_coordinate_files:
        # Read in loop coordinates
        pix_coords = astropy.table.QTable.read(filename, format='ascii')
        #pix_coords.rename_columns(pix_coords.colnames, ['x', 'y', 'z'])
        pix_coords['col1'].unit = 'Mm' #x coordinate
        pix_coords['col2'].unit = 'Mm' #y coordinate
        pix_coords['col3'].unit = 'Mm' #z coordinate
        # Apply offsets in pixel space to account for origin shift
        coord_x = (pix_coords['col1'] / scale_factor_fieldlines[0] - origin_pix.x) * scale_factor
        coord_y = (pix_coords['col2'] / scale_factor_fieldlines[1] - origin_pix.y) * scale_factor
        coord_z = pix_coords['col3'] / scale_factor_fieldlines[2] * scale_factor + m_hmi.rsun_meters
        # Create coordinate object
        coord = SkyCoord(x=coord_x,
                         y=coord_y,
                         z=coord_z,
                         frame=hcc_frame)
        fieldline_coords.append(coord)

    if Plot == True:
        #Overplot the filld lines on HMI magnetogram
        fig = plt.figure()
        ax = fig.add_subplot(projection=m_hmi)
        m_hmi.plot(axes=ax)
        for c in fieldline_coords:
            ax.plot_coord(c, color='C0',)

        #Overplot the filld lines on AIA image
        if aia_file is None: 
            q = Fido.search(a.Time(m_hmi.date-12*u.s, end=m_hmi.date+12*u.s, near=m_hmi.date),
                a.Wavelength(193*u.AA),
                a.Instrument.aia)
            aia_file = Fido.fetch(q)
        m_aia = sunpy.map.Map(aia_file)
        if aia_file is None: m_aia_crop = m_aia.submap(m_hmi.bottom_left_coord, top_right=m_hmi.top_right_coord)
        else: m_aia_crop = m_aia
        fig = plt.figure()
        ax = fig.add_subplot(projection=m_aia_crop)
        m_aia_crop.plot(axes=ax,clip_interval=(10,99.99)*u.percent)
        for c in fieldline_coords:
            ax.plot_coord(c, color='C0', lw=1, alpha=0.75)
        plt.show()
    return fieldline_coords

def Make_AIAXRTcutout(instrument = '', DataDir='',image='',left_bot_arcsec=[-500,-500],right_top_arcsec=[500,500],OutDir='',Plot=False,StoreOutput=True,OutFile=None):
    '''
    Using the outputs of "Select_XBPs.py", it will produce the cutout images in all frames.
    May.01.2023, Biswajit
    '''
    map1=sunpy.map.Map(os.path.join(DataDir,image))
    bottom_left = SkyCoord((left_bot_arcsec[0])*u.arcsec,(left_bot_arcsec[1])*u.arcsec, frame = map1.coordinate_frame)
    top_right = SkyCoord((right_top_arcsec[0])*u.arcsec,(right_top_arcsec[1])*u.arcsec, frame = map1.coordinate_frame)
    sub_map1 = map1.submap(bottom_left=bottom_left, top_right=top_right) #Select the cutout from original image
    if OutFile is None:
        if instrument == 'XRT': OutFile = 'XRTcut_'+map1.meta['ec_fw1_']
        if instrument == 'AIA': OutFile = 'AIAcut_'+str(map1.meta['wavelnth'])
        if instrument == 'HMI': OutFile = 'HMIcut_'+'LOS'
    if StoreOutput is True: sub_map1.save(os.path.join(OutDir,OutFile+'.fits'),overwrite='True')
    if Plot == True: 
        sub_map1.peek()
    return sub_map1
def element(Z):
    Z=int(Z)
    par_dir = pardir()
    z,El,El_lon,el,el_lon = np.loadtxt(par_dir+'database/elements.dat',unpack=True,dtype=np.dtype([("c1", int), ("c2", "U12"),("c3", "U12"),("c4", "U12"),("c5", "U12")]))
    ind=np.where(z == Z)[0][0]
    return int(z[ind]),El[ind],El_lon[ind],el[ind],el_lon[ind]

def rebin1d(array,addbins):
    grid = []
    cum_array = []
    c1=0 ; end = 0
    while end < len(array):
        c2 = int(c1+addbins)
        if c2 >= len(array)-1: end = len(array); dl = array[-1]+array[c1] ; mid = dl/2; cum_array +=[sum(array[c1::])]
        else: end = c2; dl = array[c2]+array[c1]; mid = dl/2; cum_array += [sum(array[c1:c2])]
        grid += [mid]
        c1 = c2
    return np.array(cum_array),np.array(grid)
def get_binsize(array):
    dL = (shift(array, -1, cval=0.0) - shift(array, 1, cval=0.0)) * 0.5
    nlength = len(array)
    dL[0] = array[1] - array[0]
    dL[nlength-1] = (array[nlength-1]-array[nlength-2])
    return abs(dL)
    
def read_EQ_file(filename):#,element = None, ion = None):
    #Read EQ ionization file:
    EQ_file = filename
    file1 = open(EQ_file, 'r')
    Lines = file1.readlines()
    ind1 = 0
    s = [float(x) for x in Lines[0].split()]
    n_ele = int(s[1])
    elements_z = np.arange(n_ele)+1
    logT = np.array([float(x) for x in Lines[1].split()])

    ioneq = np.zeros([len(logT),n_ele,31]) #[logT, element, chrg_state]

    for ele in range(n_ele):
        if ele == 0: ind1 = 2
        else: ind1 = ind1+elements_z[ele-1]+1
        ind2 = ind1 + elements_z[ele]+1
        a = [ii.split('\t') for ii in Lines[ind1:ind2]]
        for chs in range(len(a)):
            IF = np.array([float(x) for x in a[chs][0].split()][2::])
            IF[IF < 1.0e-45] = 0 #Ignore all lower values as done by Chianti
            ioneq[:,ele,chs] = IF
    return logT,ioneq #logT[temperatures], ioneq[logT, element, chrg_state]

def IDLsav2fitsBcube(IDL_file,OutFile,dr,alpha,seehafer='NA',sub_b00='NA'):
    nlfff_data = readsav(IDL_file)
    Bz = nlfff_data['bz'] #[z,y,x]
    Bz = np.transpose(Bz,axes=(1,2,0)) #[x,y,z]
    
    Nx = len(Bz[:,0,0]) ; Ny = len(Bz[0,:,0]) ; Nz = len(Bz[0,0,:])
    Bcube = np.zeros((Nx, Ny, Nz, 3))
    Bcube[:,:,:,0] = np.transpose(nlfff_data['by'],axes=(1,2,0))
    Bcube[:,:,:,1] = np.transpose(nlfff_data['bx'],axes=(1,2,0))
    Bcube[:,:,:,2] = Bz
    
    #Store Bcube to a fits file:
    hdul = fits.HDUList()
    hdul.append(fits.ImageHDU(Bcube,name='Bcube'))
    hdul[0].header.append(('dr', str(dr))) #pixcel length in arcsec
    hdul[0].header.append(('Nz',Nz))
    hdul[0].header.append(('alpha',alpha))
    hdul[0].header.append(('seehafer',seehafer))
    hdul[0].header.append(('sub_b00',sub_b00))
    hdul.writeto(OutFile,overwrite=True)

def Make_SunX_input_format_from_IDLtrace(loop_data_dir,HMI_Mag,OutFile,pix_length = 0.3625):
    hmap = sunpy.map.Map(HMI_Mag)
    
    OutDir = loop_data_dir
    #OutFileName = 'Loop_xyzB.fits'
    
    loop_files = glob.glob(os.path.join(loop_data_dir,'LoopInd*.dat'))
    
    
    hdul = fits.HDUList()
    seed_xyz_all = np.zeros([20,3])
    
    c1 = fits.Column(name='x', array=seed_xyz_all[:,0], format='K'); #format='K' -> integer
    c2 = fits.Column(name='y', array=seed_xyz_all[:,1], format='K');
    c3 = fits.Column(name='z', array=seed_xyz_all[:,2], format='K')
    hdul.append(fits.TableHDU.from_columns([c1, c2, c3],name='seeds_xyz'))
    hdul[0].header.append(('ref_image', os.path.basename(HMI_Mag)))
    hdul[0].header.append(('unit', 'pix'))
    hdul[0].header.append(('dx', hmap.scale.axis1.to('arcsec/pix').value ,'arcsec/pix'))
    hdul[0].header.append(('dy', hmap.scale.axis2.to('arcsec/pix').value ,'arcsec/pix'))
    hdul[0].header.append(('dz', hmap.scale.axis1.to('arcsec/pix').value ,'arcsec/pix'))
    '''
    #Update some headers
    hdul[0].header.append(('bot_left_x', int(xi[0])))
    hdul[0].header.append(('bot_left_y', int(yi[0])))
    hdul[0].header.append(('top_right_x', int(xi[1])))
    hdul[0].header.append(('top_right_y', int(yi[1])))
    '''
    for i in range(len(loop_files)):
        #Get the loop index from the file name in the format: 'LoopInd_0001_HMIcut_newOBS_XBP001_Extrapolated_*.dat'
        l_ind = os.path.basename(loop_files[i]).split('_')[1]
        da = np.loadtxt(loop_files[i],unpack=True)
        #seed_xyz_all[l_ind,:] = seed_xyz
        #Writing to file
        if da[2][0]/pix_length < 50 and da[2][-1]/pix_length < 50: #Consider only closed loops
            c1 = fits.Column(name='x', array=np.round(da[1]/pix_length), format='K');
            c2 = fits.Column(name='y', array=np.round(da[0]/pix_length), format='K');
            c3 = fits.Column(name='z', array=np.round(da[2]/pix_length), format='K')
            c4 = fits.Column(name='|By|',array=da[3],format='D')
            c5 = fits.Column(name='|Bx|',array=da[4],format='D')
            c6 = fits.Column(name='|Bz|',array=da[5],format='D')
            hdul.append(fits.TableHDU.from_columns([c1, c2, c3,c4,c5,c6],name='L'+l_ind))
    
    hdul.writeto(OutFile,overwrite=True)

#### model util functions
def nanoflareprof(qbkg=1.0e-5,tau=50, dur=20000, q0min=0.02, q0max=2.0, Fp=0.2,  L_half = 5.0, PrintOut = False,HeatingFunction = False, seed = None):
   '''
   Generates a random sequence of nanoflares from the minimum and maximum abailable magnatic energy associated with a loop.
   The delay between successive events is proportional to the magnitude of the first event. 
   This corresponds to a physical scenario in which strand footpoints are constantly driven by photospheric 
   convection, and a nanoflare occurs when a critical level of magnetic stress is reached.

   22-Jan-30, Biswajit.
   22-Feb-08, Biswajit, Modfied to match the time delay such that the average 
              heating rate is equal to the upward poyinting flux from the photosphere.

   Inputs:
     q0min -       minimum heating rate
     q0max -       maximum heating rate
     tau - half duration of nanoflare (s) (triangluar profile assumed)
     dur - duration of simulation (s)
     L_half - loop half length in Mm
     Fp - poynting flux rate in erg/cm2/s

     Optional:
       PrintOut
       HeatingFunction
       seed

   Outputs:
     Peak_time - peak time of the events.
     Peak_heat - peak heating rate (erg cm^-3 s^-1)
     time -        time array (1 sec increment) # if HeatingFunction = True
     heat -        corresponding array of heating rate (erg cm^-3 s^-1) # if HeatingFunction = True

   Example:

         Peak_time,Peak_heat = nano_seq_mod2(qbkg=1.0e-5,tau=50, dur=600000, q0min=0.002, q0max=0.02,  L_half = 10.0, PrintOut = False,HeatingFunction = False, seed = None)
          
         a1,b1,_=plt.hist(np.log10(Peak_heat),bins=50) 
         Peak_time,Peak_heat = nano_seq_mod(qbkg=1.0e-5,tau=50, dur=60000, q0min=0.002, q0max=0.02,  L_half = 10.0, PrintOut = False,HeatingFunction = False, seed = None)
          
         b1=10**b1 
         de1 = b1[1::]-b1[0:-1] 
         a,b,_=plt.hist(np.log10(Peak_heat),bins=20,alpha=0.5) 
         b=10**b 
         plt.close('all')  
         plt.yscale('log')  
         plt.xscale('log')  
         de = b[1::]-b[0:-1] 
         plt.step((b1[1::]+b1[0:-1])/2.0,a1/de1)  
         plt.step((b[1::]+b[0:-1])/2.0,a/de)  
         plt.show()                                                                                                                                                               
   Note:  the first 500 s of the simulation should be ignored because they could be affected by the initial conditions
   '''
   time = np.arange(dur + 1)
   if HeatingFunction is True : heat = np.zeros(int(dur + 1))

   Prop_const = tau*L_half*1.0e8/Fp #Proportionality constant. 
   if seed is not None : np.random.seed(seed)
   t1 = int(500*np.random.uniform(low=0.0, high=1.0, size=1)[0])   # first nanoflare begins randomly in the first 500 s
   q00 = np.random.uniform(low=q0min, high=q0max, size=1)[0]
   if HeatingFunction is True :
       for i in range(tau+1):              #triangular profile rise
           heat[t1+i] = q00*i/tau
       for i in range(tau+1, (2*tau)+1): heat[t1+i] = q00*(2.*tau - i)/tau  #;  decay

   Peak_heat = [q00] #peak heating rate of each triangular profile
   Peak_time = [t1+tau] #peak time

   k = 0
   tnew = t1 + q00*Prop_const
   delay_arr = [tnew - t1]
   delay_good = []
   while (tnew+2*tau < dur):
       k = k + 1
       q0k = np.random.uniform(low=q0min, high=q0max, size=1)[0]
       if HeatingFunction is True :
           for i in range(tau+1): heat[int(tnew+i)] = heat[int(tnew+i)] + q0k*i/tau
           for i in range(tau+1, (2*tau)+1) : heat[int(tnew+i)] = heat[int(tnew+i)] + q0k*(2.*tau - i)/tau

       Peak_heat += [q0k]
       Peak_time += [tnew+tau]
       told = tnew
       tnew = told + q0k*Prop_const
       delay_arr += [tnew - told]
       if (tnew >= 500): delay_good += [tnew - told]
   delay_arr = np.array(delay_arr)
   delay_good = np.array(delay_good)
   if HeatingFunction is True :
       h_cor = L_half*1.0e8 #  coronal scale height
       heat = heat + qbkg
       mean_heat = np.mean(heat[500:int(dur)])
       Mean_energy_flux = mean_heat*h_cor #erg/cm2/s
   ss = np.where(delay_good != 0.)
   delay_good = delay_good[ss]
   mean_delay = np.mean(delay_good)
   median_delay = np.median(delay_good)

   ss = np.where(delay_arr != 0.)
   delay_arr = delay_arr[ss]
   mean_delay_all = np.mean(delay_arr)
   median_delay_all = np.median(delay_arr)
   h_cor = L_half*1.0e8 #5.e9  #;  coronal scale height
   if PrintOut == 'True':
       print(' ')
       if HeatingFunction is True : print('mean energy flux = ', Mean_energy_flux)
       print('mean delay = ', mean_delay)
       print('median delay = ', median_delay)
       print(' ')
       print('Including the first 500 s:')
       print('mean delay = ', mean_delay_all)
       print('median delay = ', median_delay_all)
   if HeatingFunction is True :return time,heat,np.array(Peak_time),np.array(Peak_heat),Mean_energy_flux
   else: return np.array(Peak_time),np.array(Peak_heat)

def Average_Heating_Rate(alpha=2.0,q0max=2.0,q0min=0.02):
    '''
    It will return the average peak heating rate of a nano-flare sequence.
    
    Inputs:
       alpha -        power-law index of peak heating rate probability distribution
       q0min -        minimum heating rate (erg/cm3/s)
       q0max -        maximum heating rate (erg/cm3/s)

    Outputs:
       <Q0> - average peak heating rate in erg/cm3/s

    --- Biswajit 08/02/2022.
    '''

    if alpha == 1: Q0 = (q0max - q0min) / np.log(q0max/q0min)
    elif alpha == 2: Q0 = (np.log(q0max/q0min) * q0max * q0min) / (q0max - q0min)
    else:
       alp1 = 1.0-alpha
       alp2 = 2.0-alpha
       const2 = alp1/alp2
       Q0 = const2 * (q0max**alp2 - q0min**alp2)/(q0max**alp1 - q0min**alp1)
    return Q0 #erg/cm3/s

def Estimate_PoyntingFlux(L_half = 5.0,Vs=1.0, c = 0.3,Fp = None,B_avg = 4.0,B_Coronal_base=4.0):
    L_half = L_half*1.0e8 #in cm
    if Fp is not None: Fp = Fp
    else:
        Vs = Vs*1.0e5 #in cm/s
        Fp = (c*Vs*B_avg*B_Coronal_base)/(4.0* np.pi) #Poynting_flux
    return Fp

def char_delay(tau=50, Q0=0.2, L_half = 5.0,B_avg = 4.0,Vs=1.0, c = 0.3,Fp = None,B_Coronal_base=4.0):
    '''
    Purpose: Estimate the characteristic delay between two nanoflares associated with a loop.

    OutPut: Characteristic delay time in s

    Inputs:
        tau - half duration of nanoflare (s) (triangluar profile assumed)
        Q0 - average peak heating rate in erg/cm3/s 
        alpha - slope of heating frequency
        L_half - loop half length in Mm
        B_avg - average magnetic field assiciated with the loop
        Vs - Photospheric Share velocity in km/s
        c - ratio of share component and Z-components of magnetic field, i.e.,  Bs/Bz
        Fp - Given poynting flux (erg/cm2/s). If it is 'None' then the Poynting flux will be = (c*Vs*B_avg*B_avg)/(4.0* np.pi)
    Biswajit, 08.Feb.2022
    '''
    L_half = L_half*1.0e8 #in cm
    if Fp is not None: Fp = Fp
    else:
        Vs = Vs*1.0e5 #in cm/s
        Fp = (c*Vs*B_avg*B_Coronal_base)/(4.0* np.pi) #Poynting_flux
    const = tau * L_half / Fp
    return const*Q0
def nanoflareprof_PoLaw(alpha=2.4,qbkg=1.0e-5,tau=50, dur=60000, delay=1000.0, q0min=0.02, q0max=2.0, Q0=0.2,  L_half = 5.0, PrintOut = False,HeatingFunction = False, seed = None):
    '''
    Generates a sequence of nanoflares from a power-law probability distribution of peak nanoflare 
    heating rate. The delay between successive events is proportional to the magnitude of the first event. 
    This corresponds to a physical scenario in which strand footpoints are constantly driven by photospheric 
    convection, and a nanoflare occurs when a critical level of magnetic stress is reached.


    18-jul-02, written, J. Klimchuk in IDL
    19-mar-19, J. Klimchuk, fixed bug to allow overlapping nanoflares
    21-apr-01, J. Klimchuk, corrected comment to indicate that tau is the half duration of the nanoflare, not total duration
    22-Jan-30, Biswajit, convert to python3 
                         # Note: qbkg is not required for EBTEL++ as it will be given separately. 
                         # Thus it will not be used in the retured array of Peak_heat
    22-Feb-08, Biswajit, Modfied to match the time delay such that the average heating rate is equal to the upward poyinting flux from the photosphere.

    Inputs:
      alpha -       power-law index of peak heating rate probability distribution
      q0min -       minimum heating rate
      q0max -       maximum heating rate
      delay -       characteristic delay between successive events (s)
      qbkg -        steady background heating rate
      tau - half duration of nanoflare (s) (triangluar profile assumed)
      dur - duration of simulation (s)
      L_half - loop half length in Mm
      Q0 - average peak heating rate in erg/cm3/s

    Outputs:
      Peak_time - peak time of the events.
      Peak_heat - peak heating rate (erg cm^-3 s^-1)
      time -        time array (1 sec increment) # if HeatingFunction = True
      heat -        corresponding array of heating rate (erg cm^-3 s^-1) # if HeatingFunction = True

    Example:

          Peak_time,Peak_heat = nano_seq_mod(alpha=2.4,qbkg=1.0e-5,tau=50, dur=600000, delay=1000.0, q0min=0.002, q0max=0.02, Q0=0.02,  L_half = 10.0, PrintOut = False,HeatingFunction = False, seed = None)
           
          a1,b1,_=plt.hist(np.log10(Peak_heat),bins=50) 
          Peak_time,Peak_heat = nano_seq_mod(alpha=1.5,qbkg=1.0e-5,tau=50, dur=600000, delay=1000.0, q0min=0.002, q0max=0.02, Q0=0.02,  L_half = 10.0, PrintOut = False,HeatingFunction = False, seed = None)
           
           
          b1=10**b1 
          de1 = b1[1::]-b1[0:-1] 
          a,b,_=plt.hist(np.log10(Peak_heat),bins=20,alpha=0.5) 
          b=10**b 
          plt.close('all')  
          plt.yscale('log')  
          plt.xscale('log')  
          de = b[1::]-b[0:-1] 
          plt.step((b1[1::]+b1[0:-1])/2.0,a1/de1)  
          plt.step((b[1::]+b[0:-1])/2.0,a/de)  
          plt.show()                                                                                                                                                                                         
    Note:  the first 10,000 s of the simulation should be ignored because they could be affected by the initial conditions
    
    '''
    num_nano = int(20.*dur/delay) #Here 2 is multiplied to increase the initial number of event sampling

    q0 = randomp(-alpha, num_nano, range_x=[q0min,q0max],seed = seed) #Events are selected from a random power-law distribution between q0min and q0max
    #q0 = np.random.uniform(low=q0min, high=q0max, size=num_nano) # Events are selected from uniform random number between q0min and q0max

    time = np.arange(dur + 1)
    if HeatingFunction is True : heat = np.zeros(int(dur + 1))

    delay_arr = np.zeros(num_nano - 1)
    delay_good = np.zeros(num_nano - 1)
    Prop_const = delay/Q0 #Proportionality constant. 
    #seed = !NULL
    if seed is not None : np.random.seed(seed)
    t1 = int(5000*np.random.uniform(low=0.0, high=1.0, size=1)[0])   # first nanoflare begins randomly in the first 5000 s
    if HeatingFunction is True :
        for i in range(tau+1): heat[t1+i] = q0[0]*i/tau  #;                   triangular profile rise
        for i in range(tau+1, (2*tau)+1): heat[t1+i] = q0[0]*(2.*tau - i)/tau  #;  decay

    Peak_heat = [q0[0]] #peak heating rate of each triangular profile
    Peak_time = [t1+tau] #peak time

    k = 0
    tnew = t1 + q0[0]*Prop_const
    delay_arr[0] = tnew - t1

    while (tnew+2*tau < dur):
        k = k + 1
        if HeatingFunction is True :
            for i in range(tau+1): heat[int(tnew+i)] = heat[int(tnew+i)] + q0[k]*i/tau
            for i in range(tau+1, (2*tau)+1) : heat[int(tnew+i)] = heat[int(tnew+i)] + q0[k]*(2.*tau - i)/tau

        Peak_heat += [q0[k]]
        Peak_time += [tnew+tau]
        told = tnew
        tnew = told + q0[k]*Prop_const
        delay_arr[k] = tnew - told
        if (tnew >= 10000): delay_good[k] = tnew - told

    if HeatingFunction is True :
        h_cor = L_half*1.0e8 #5.e9  #;  coronal scale height
        heat = heat + qbkg
        mean_heat = np.mean(heat[10000:int(dur)])
        Mean_energy_flux = mean_heat*h_cor #erg/cm2/s

    ss = np.where(delay_good != 0.)
    delay_good = delay_good[ss]
    mean_delay = np.mean(delay_good)
    median_delay = np.median(delay_good)

    ss = np.where(delay_arr != 0.)
    delay_arr = delay_arr[ss]
    mean_delay_all = np.mean(delay_arr)
    median_delay_all = np.median(delay_arr)
    h_cor = L_half*1.0e8 #5.e9  #;  coronal scale height
    if PrintOut == 'True':
        print(' ')
        if HeatingFunction is True : print('mean energy flux = ', Mean_energy_flux)
        print('mean delay = ', mean_delay)
        print('median delay = ', median_delay)
        print(' ')
        print('Including the first 10000 s:')
        print('mean delay = ', mean_delay_all)
        print('median delay = ', median_delay_all)

    if HeatingFunction is True :return time,heat,np.array(Peak_time),np.array(Peak_heat),Mean_energy_flux
    else: return np.array(Peak_time),np.array(Peak_heat)

def H_back(L,T):
    #function to estimate the required constant background heating rate to mentain an average temperature throughout the loop:(ref- Rajhans, Tripathi et.al _2021)
    '''
    L -> loop half length in cm
    T -> BKG temperature in k
    '''
    k = 8.12e-7 #in cgs
    cons = 0.4131267
    T2 = T**(7.0/2.0) # T in k
    L2 = L**2 # Half loop length in cm
    return cons * k * (T2/L2)

'''
def loop_skeleton(self,Extrapolation_File=None,No_loops=None, Area_File=None,AspectRatio=None, Fixed_radious=None, L_min = None,L_max = None,Thresold_Phot_B = None,Minimum_height = None, ConstPoyntingFlux=None, CoronalBase = None):
    config = self.config
    if Extrapolation_File is None: Extrapolation_File = config['SetupSimulation']['LoopParamFile']
    if os.path.isfile(Extrapolation_File) is False: raise Exception("%% simar_error : File not exist- "+Extrapolation_File)
    if No_loops is None:
        try:No_loops = config.getint('SetupSimulation','No_loops')
        except: No_loops = None
    if Area_File is None: Area_File = config.getboolean('SetupSimulation', 'Area_File')
    if AspectRatio is None: AspectRatio = config.getfloat('SetupSimulation','AspectRatio')
    if Fixed_radious is None: Fixed_radious = config.getfloat('SetupSimulation','Fixed_radious')
    if L_min is None: L_min = config.getfloat('SetupSimulation','L_min')
    if L_max is None: L_max = config.getfloat('SetupSimulation','L_max')
    if Thresold_Phot_B is None: Thresold_Phot_B = config.getfloat('SetupSimulation','Thresold_Phot_B')
    if Minimum_height is None: Minimum_height = config.getfloat('SetupSimulation','Minimum_height')
    if ConstPoyntingFlux is None: ConstPoyntingFlux = config.getboolean('SetupSimulation','ConstPoyntingFlux')
    if CoronalBase is None: CoronalBase = config.getfloat('SetupSimulation','CoronalBase')
    if CoronalBase is None: CoronalBase == 0.0


    #It will return the loop half lenght and Average-mag-potential-E distribution accociated with each XBP and the corresponding magnetic free energy. 
    # The number of loops will be such that the total area of all loops corresponds to the total area of the XBPs as read from "XBP_property_File".

    ##Extrapolation_dir -> the name of the extrapolation file got from 'Estimate_MagEnergy_LoopLengths.py'
    ##No_loops -> Number of loops to be considered. If None, then it will be computed from total AR-area.
    ##Area_File -> Area of individual XBPs will be read from the files store within the directory of Extrapolation_File directory for calculation. If Area_File != True, then the area store as the header of Extrapolation_File will be used (which is the area under the extrapolated regions estimated from the peripheri of loop-footpoints.) 
    ##AspectRatio -> ratio between the loop length and diameter 
    ##Fixed_radious -> Fixed radious of all the loops in Mm. If set 0, then it will not be used. The AspectRatio will use. 
    ##L_min & L_max-> minimum & maximum loop length to be consider.    
    ##Thresold_Phot_B -> Thresold phtospheric magnetic field for which the loops will be considered.

    ##Biswajit, 29.Jan.2022
    ##Biswajit, 22.Mar.2022, added functionality for pojected loop-length
    ##Biswajit, 29.Mar.2022, Removed the functionality of Area_File='n', Removed-- pojected loop-length.
    ##                       If ConstPoyntingFlux = True, it will return B_0, which is the magnetic field 
    ##                       at the base of the corona, as well as the average manetic field above "CoronalBase = 0.5 or 1.0 or 2.0 Mm".
    ##                       Added functionality "Minimum_height", the loops will be selected for which the height is >= Minimum_height.
    ##Biswajit, 10.Sept.2023, added functionality of 'No_loops'
    
    #Loop_No,  Length, MagField, B_phot_pos, B_phot_neg = np.loadtxt(Extrapolation_File, unpack = True)    
    #Loop_No,  Length, MagField, B_phot_pos, B_phot_neg, Length_projected = np.loadtxt(Extrapolation_File, unpack = True)
    #print(Extrapolation_File)
    loop_ind, Full_length, Bpos_foot, Bneg_foot, Avg_B, mod_B_ge0p5_pos, mod_B_ge0p5_neg, Avg_B_ge0p5, mod_B_ge1p0_pos, mod_B_ge1p0_neg, Avg_B_ge1p0, mod_B_ge2p0_pos, mod_B_ge2p0_neg, Avg_B_ge2p0,Loop_height = np.loadtxt(Extrapolation_File, unpack = True)
    ind  = np.where((Bpos_foot > Thresold_Phot_B) & (Bneg_foot < -Thresold_Phot_B) & (Loop_height >= Minimum_height))[0]
    Loop_No = loop_ind[ind]
    Length = Full_length[ind]
    Loop_height = Loop_height[ind]

    if ConstPoyntingFlux is True:
        MagField = Avg_B[ind]
        ind2 = np.where(MagField > 0.0)[0] #Neglect the small-loops which does-not satiesfy the criteria
        MagField = MagField[ind2]
        Length = Length[ind2]
        Loop_No = Loop_No[ind2]
    else:
        if CoronalBase == 0.0:#photosphere
            MagField = Avg_B[ind]
            B_0 = (Bpos_foot[ind]+ abs(Bneg_foot[ind]))/2.0
        if CoronalBase == 0.5:
            MagField = Avg_B_ge0p5[ind]
            B_0 = (mod_B_ge0p5_pos[ind]+ mod_B_ge0p5_neg[ind])/2.0
        elif  CoronalBase == 1.0:
            MagField = Avg_B_ge1p0[ind]
            B_0 = (mod_B_ge1p0_pos[ind]+ mod_B_ge1p0_neg[ind])/2.0
        elif  CoronalBase == 2.0:
            MagField = Avg_B_ge2p0[ind]
            B_0 = (mod_B_ge2p0_pos[ind]+ mod_B_ge2p0_neg[ind])/2.0
        else: sys.exit('%% Error: data not exist')
        ind2 = np.where(MagField > 0.0)[0] #Neglect the small-loops which does-not satiesfy the criteria
        MagField = MagField[ind2]
        B_0 = B_0[ind2]
        Length = Length[ind2]
        Loop_No = Loop_No[ind2]
    L_half = []
    Avg_B = []
    B_Coronal_base = []
    L_ind_ = [] #required for MaGIXS simulation
    if No_loops is None:
        #if Area_File != True :
        #    fi=open(Extrapolation_File,'r')
        #    l=fi.readlines()[-10::]
        #    pos_Area = float(find_str(l, '#Area_pos')[0].split('=')[-1])
        #    neg_Area = float(find_str(l, '#Area_neg')[0].split('=')[-1])
        #    Half_Area = (pos_Area+neg_Area)/2.0
        #else:
        fi=open(Extrapolation_File,'r')
        l=fi.readlines()[0:3]
        Area_full = float(find_str(l, 'Area(cm2)=')[0].split('=')[-1])
        Half_Area = Area_full/2.0
        #print(Area_File1_str[0],xbp_no,Half_Area*2)
        A=0.0
        c=0
        while A <= Half_Area:
            if  Length[c] >= L_min and Length[c] <= L_max:
                if Fixed_radious == 0 :CS_area = 3.14* (Length[c]*1.0e8/(2.0*AspectRatio))**2
                else: CS_area = 3.14*Fixed_radious*Fixed_radious*1.0e16 #in cm2
                A += CS_area
                Avg_B += [MagField[c]]
                L_half += [Length[c]/2.0]
                L_ind_ += [Loop_No[c]]
                if ConstPoyntingFlux is not True: B_Coronal_base += [B_0[c]]
            c+=1
            if c == len(Length) : c = 0
    else:
        c = 0
        while c <= No_loops:
            if  Length[c] >= L_min and Length[c] <= L_max:
                Avg_B += [MagField[c]]
                L_half += [Length[c]/2.0]
                L_ind_ += [Loop_No[c]]
                if ConstPoyntingFlux is not True: B_Coronal_base += [B_0[c]]
            c+=1
            if c == len(Length) : c = 0

    if ConstPoyntingFlux is True:return np.array(L_half),np.array(Avg_B),np.array(L_ind_) #in Mm and G
    else: return np.array(L_half),np.array(Avg_B), np.array(B_Coronal_base),np.array(L_ind_)
'''
def Loop_avail_Erange(c,B_avg,tau_half,frac_minE = 0.01):
    '''
    It will return the minimum and maximum energy available within a loop.
    '''
    q0max = (c*B_avg)**2 / (tau_half*8.0*np.pi) #(erg/cm3/s) maximum heating rate that a loop can afforts
    q0min = frac_minE*q0max #Consider as 1% of q0max
    return q0max,q0min
def klimchuk_rad(log_temperature):
        if( log_temperature <= 4.97 ):
            chi = 1.09e-31
            alpha = 2.0
        elif( log_temperature <= 5.67 ):
            chi = 8.87e-17
            alpha = -1.0
        elif( log_temperature <= 6.18 ):
            chi = 1.90e-22;
            alpha = 0.0;
        elif( log_temperature <= 6.55 ):
            chi = 3.53e-13
            alpha = -3.0/2.0
        elif( log_temperature <= 6.90 ):
            chi = 3.46e-25
            alpha = 1.0/3.0
        elif( log_temperature <= 7.63 ):
            chi = 5.49e-16
            alpha = -1.0
        else :
            #// NOTE: free-free radiation is included in the parameter values for log_10 T > 7.63
            chi = 1.96e-27
            alpha = 1.0/2.0

        return (chi * (10.0**(alpha*log_temperature))),chi,alpha

def t_rad(T,n,K0,b):
    return ((3*1.38e-16*(T**1-b)) / (K0*n))
def t_cond(T,n,L_half,b):
    kk_ = 10**-6 
    return (21/2)*(1.38e-16/kk_)*(n*(L_half**2)*(T**(-5/2)))
def exponential_func(x,b,a):
    return a * np.exp(-b * x)
