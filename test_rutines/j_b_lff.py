import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def j_b_lff(bz0, z, alpha1=0.0,seehafer=False,sub_b00=False, sub_plan=False):

    '''
    Python implementation of J.R.Costa j_b_lff.pro extrapolation routine, which is available within SSW.
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
    print('alpha=',alpha)
    kx=np.outer(np.ones([1,ny],dtype=int),kx)
    ky=np.outer(ky,np.ones([nx],dtype=int))
    
    fbz0 = np.fft.fftn(bz0e, norm='forward')
    kz=np.sqrt(np.maximum(kx**2 + ky**2 - alpha**2, 0))  # Positive solutions, see Nakagawa e Raadu p. 132
    ex__ = kz**2 + alpha**2
    ex__[ex__ < kx[0,1]**2] = kx[0,1]**2
    argx = fbz0 * (-1j) * (kx * kz - alpha * ky) / ex__
    ex__ = kz**2 + alpha**2
    ex__[ex__ < ky[1,0]**2] = ky[1,0]**2
    argy = fbz0 * (-1j) * (ky * kz + alpha * kx) / ex__
    bx = np.zeros((nx1, ny1, nz))
    by = np.zeros((nx1, ny1, nz))
    bz = np.zeros((nx1, ny1, nz))
    for j in range(nz):
        bx[:, :, j] = np.real(np.fft.ifftn(argx * np.exp(-kz * z[j]),norm='forward')[:nx1, :ny1])
        by[:, :, j] = np.real(np.fft.ifftn(argy * np.exp(-kz * z[j]),norm='forward')[:nx1, :ny1])
        bz[:, :, j] = np.real(np.fft.ifftn(fbz0 * np.exp(-kz * z[j]),norm='forward')[:nx1, :ny1]) + b00
        print(np.fft.ifftn(fbz0 * np.exp(-kz * z[j])))#bz[:,:,0])
    return bx, by, bz

def gx_bz2lff(Bz, Nz=None, dr=None, alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False):
    '''
    Python implementation of gx_bz2lff.pro routine available in SSW to run j_b_lff.pro
    '''

    sz = Bz.shape
    Nx,Ny = sz[0],sz[1]
    N = [Nx,Ny]
    if len(sz) >= 4 : Bz = np.reshape(Bz[:,:,:,1],(Nx,Ny,sz[3]))
    if len(sz) >= 3 : Bz = np.reshape(Bz[:,:,sz[2]],(Nx,Ny))
    if Nz is None: Nz = min(Nx, Ny) #;Nz specifies the height of the extrapolation
    N.extend([Nz])
    if dr is None: dr = [1.0, 1.0, 1.0] #;dr represents voxel size in each direction (arcsec) 
    z = np.arange(Nz) * dr[2] / dr[0]
    bxout, byout, bzout = j_b_lff(Bz, z, alpha1=alpha1, seehafer=seehafer, sub_b00=sub_b00, sub_plan=sub_plan)
    Bcube = np.zeros((Nx, Ny, Nz, 3))
    Bcube[:, :, :, 0] = bxout
    Bcube[:, :, :, 1] = byout
    Bcube[:, :, :, 2] = bzout
    return Bcube


'''
# Example usage
bz0 = np.random.rand(32, 32)  # Replace with your input data
z = np.linspace(0, 1, 10)  # Replace with your z values
alpha1 = 0.0

#'/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/HMI/HMIcut_newOBS_XBP001.fits'
'''

#Create a bi-pole
x=np.linspace(-10,10, num=100)
y=np.linspace(-10,10, num=100)
x, y = np.meshgrid(x, y)
bz0 = 50*(np.exp(-0.1*(x-5)**2-0.1*(y-5)**2) - np.exp(-0.1*(x+5)**2-0.1*(y+5)**2))

#plt.imshow(bz0,origin='lower')
#plt.show()

#bx, by, bz = j_b_lff(bz0, z, alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False)
bcube = gx_bz2lff(bz0, Nz=50, dr=[0.5,0.5,0.5], alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False)


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x,y,z, cmap=cm.jet)
#plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z)
ax.plot_surface(X, Y, Z_pred)
plt.show()
'''


#compare the python and idl results
from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np

sav_data = readsav('t.sav')#read idl output

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.plot_surface(x, y, bcube[:,:,0,2])
#ax2.plot_surface(x, y, bz0)
'''
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(x, y, sav_data['bcube'][0,10,:,:])
#ax.plot_surface(X, Y, Z_pred)
plt.show()
'''

tt=readsav('tt.sav')
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(tt['x'], tt['y'], sav_data['bcube'][2,0,:,:])
#ax.plot_surface(tt['x'], tt['y'], tt['bz0'])
plt.show()
