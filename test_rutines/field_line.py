import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.interpolate import interp1d

def FIELD_INTERP(Vx, Vy, Vz, x, y, z, coord):
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
    print(i0,j0,k0)
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
        q[0] = Vx_x + Vx_y + Vx_z - 2.0*Vx[i0,j0,k0]
    
        #; -- interpolate Vy --
    
        Vy_x = np.interp(xp,x,Vy[:,j0,k0])
        Vy_y = np.interp(yp,y,Vy[i0,:,k0])
        Vy_z = np.interp(zp,z,Vy[i0,j0,:])
        q[1] = Vy_x + Vy_y + Vy_z - 2.0*Vy[i0,j0,k0]
    
        #; -- interpolate Vz --
    
        Vz_x = np.interp(xp,x,Vz[:,j0,k0])
        Vz_y = np.interp(yp,y,Vz[i0,:,k0])
        Vz_z = np.interp(zp,z,Vz[i0,j0,:])
        q[2] = Vz_x + Vz_y + Vz_z - 2.0*Vz[i0,j0,k0]
        '''
        #; -- interpolate Vx --
        interp = interp1d(x,Vx[:,j0,k0], fill_value="extrapolate",kind='cubic')
        Vx_x = interp(xp)
        interp = interp1d(y,Vx[i0,:,k0], fill_value="extrapolate",kind='cubic')
        Vx_y = interp(yp)
        interp = interp1d(z,Vx[i0,j0,:], fill_value="extrapolate",kind='cubic')
        print(zp)
        Vx_z = interp(zp)
        q[0] = Vx_x + Vx_y + Vx_z - 2.0*Vx[i0,j0,k0]
    
        #; -- interpolate Vy --
    
        interp = interp1d(x,Vy[:,j0,k0], fill_value="extrapolate",kind='cubic')
        Vy_x = interp(xp)
        interp = interp1d(y,Vy[i0,:,k0], fill_value="extrapolate",kind='cubic')
        Vy_y = interp(yp)
        interp = interp1d(z,Vy[i0,j0,:], fill_value="extrapolate",kind='cubic')
        Vy_z = interp(zp)
        q[1] = Vy_x + Vy_y + Vy_z - 2.0*Vy[i0,j0,k0]
    
        #; -- interpolate Vz --
        interp = interp1d(x,Vz[:,j0,k0], fill_value="extrapolate",kind='cubic')
        Vz_x = interp(xp)
        interp = interp1d(y,Vz[i0,:,k0], fill_value="extrapolate",kind='cubic')
        Vz_y = interp(yp)
        interp = interp1d(z,Vz[i0,j0,:], fill_value="extrapolate",kind='cubic')
        Vz_z = interp(zp)
        q[2] = Vz_x + Vz_y + Vz_z - 2.0*Vz[i0,j0,k0]
        '''
    return q

def field_line(Vx, Vy, Vz, x, y, z, seed, method="RK2", maxstep=None,minstep=None, step=None, tol=None):
    #'''
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

    #EXAMPLE:
    #
    #  * compute a field line tangent to the vector field (Bx1,Bx2) in 2D at the 
    #    point with coordinate (-1,2) using the Bogacki-Shampine method with relative
    #    tolerance 1.e-4:
    #
    #    pl = field_line(Bx1, Bx2, 0.0 x1, x2, 0.0, seed=[-1,2], pl, method="BS23", tol=1.e-4)
    #    oplot, pl[0,*], pl[1,*]  ; overplot on current window
    #
    #  * Same as before but in 3D and at the point [-1,2,0.5]:
    #
    #    pl = field_line(Bx1, Bx2, Bx3, x1, x2, x3, seed=[-1,2,0.5], pl, method="BS23", tol=1.e-4)
    #------------------------
    #--- Biswajit Aug-31-2023
    #------------------------
    #'''

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
    if np.any(condt0 == False) == True: Exception("%% simar_error : Given seed point falls outside grid range.") #Make sure initial seed point falls 
                           
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
                k1  = FIELD_INTERP(Vx, Vy,Vz, x, y, z, xk0)
                xk1 = xk0 + 0.5*dh*k1
                k2 = FIELD_INTERP(Vx, Vy, Vz, x, y, z, xk1)
                xfwd[:,k] = xk0 + dh*k2

            '''
            #; ----------------------------------------------------------
            #;   Explicit Runge-Kutta method with 4th order accuracy.
            #;   Fixed step size. Requires 4 function evaluations.
            #; ----------------------------------------------------------
            if method == "RK4":
                k = k+1

                k1  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk0)
                xk1 = xk0 + 0.5*dh*k1

                k2  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + 0.5*dh*k2

                k3 = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*k3

                k4 = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xfwd[:,k] = xk0 + dh*(k1 + 2.0*(k2 + k3) + k4)/6.0

            #; ---------------------------------------------------------------
            #;  Explicit Runge-Kutta-Fehlberg pair (2,3) with adaptive 
            #;  step size. It is also known as Bogacki-Shampine and provide
            #;  third-order accuracy using a total of 4 function evaluations.
            #; ---------------------------------------------------------------
            
            if method == "BS23": #; -- use BS23

                k1  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk0)
                xk1 = xk0 + dh*a21*k1

                k2  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*(a31*k1 + a32*k2)

                k3  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*(a41*k1 + a42*k2 + a43*k3)

                k4  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
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
                    if (kfail > max_fail): raise Exception("%% simar_error : Too many failures!")

            #; ---------------------------------------------------------------
            #;  Cash-Karp fifth-order method using a (4,5) pair.
            #;  Provides adaptive step-size control with monitoring of local 
            #;  truncation error. It requires 6 function evaluations.
            #; ---------------------------------------------------------------
            if method == "CK45": 

                k1  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk0)
                xk1 = xk0 + dh*a21*k1

                k2  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*(a31*k1 + a32*k2)

                k3  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*(a41*k1 + a42*k2 + a43*k3)

                k4  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*(a51*k1 + a52*k2 + a53*k3 + a54*k4)

                k5  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
                xk1 = xk0 + dh*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)

                k6  = FIELD_INTERP(Vx, Vy, Vz, x, y, z,xk1)
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
                    if (kfail > max_fail):raise Exception("%% simar_error : Too many failueres!")
            '''   
            inside_domain = 1
            condt = np.greater(xfwd[:,k],dbeg[:]) == np.less(xfwd[:,k],dend[:])
            if np.any(condt == False) == True: inside_domain = 0 #Check whether we're still inside the domain.
        #ENDWHILE
        if s == -1:
            xbck  = xfwd
            k_bck = k
    #ENDFOR
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

    

import package as fld
from scipy.io import readsav
import subprocess
'''
m = fld.fieldextrap(configfile='package/config.dat')

## create LOZ magnetogram
x=np.linspace(-10,10, num=100)
y=np.linspace(-10,10, num=100)
x, y = np.meshgrid(x, y)
bz0 = 50*(np.exp(-0.1*(x-5)**2-0.1*(y-5)**2) - np.exp(-0.1*(x+5)**2-0.1*(y+5)**2))
plt.imshow(bz0,origin='lower')
plt.show()
bcube = m.gx_bz2lff(bz0, Nz=50, dr=[0.5,0.5,0.5], alpha1=0.0, seehafer=False, sub_b00=False, sub_plan=False)
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.plot_surface(x, y, bcube[:,:,30,2])

#plot the idl outputs obtained by runing 'run_idl_jb_lff.pro' (run it from idl)
#idl_command="ssw -e "+"'"+"run_idl_jb_lff,x="+str(list(x))+",y="+list(y)+",bz0="+list(bz0)+"'"
#spectr = subprocess.check_output([idl_command],shell=True)
sav_data = readsav('package/test_idl_out.sav')#read idl output
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(sav_data['x'], sav_data['y'], sav_data['bcube'][2,30,:,:])
plt.show()

Vx = bcube[:,:,:,0]
Vy = bcube[:,:,:,1]
Vz = bcube[:,:,:,2]
x = np.arange(100,dtype=float)
y = np.arange(100, dtype=float)
z = np.arange(50,dtype=float)
IntpFunc = initialize_3D_InterpFuncs(Vx, Vy, Vz, x, y, z)
pl = field_line(Vx, Vy, Vz, x, y ,z, seed=[25,25,10], method="BS23",InterpFunc=IntpFunc)


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#plt.scatter(sav['pl'][:,0],sav['pl'][:,1],sav['pl'][:,2])

fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
plt.scatter(pl[0,:],pl[1,:],pl[2,:])
plt.show()
'''

#test on actual magnetogram
magnetogram = readsav('/Users/bmondal/BM_Works/MaGIXS/MaGIXS_1/XBP_1/data/HMI/HMIcut_newOBS_XBP001_Extrapolated.sav')

bx1=magnetogram['Bx']#.T
bx2=magnetogram['By']#.T
bx3=magnetogram['Bz']#.T

bx1 = np.transpose(bx1, (1, 2, 0))
bx2 = np.transpose(bx2, (1, 2, 0))
bx3 = np.transpose(bx3, (1, 2, 0))


origin = 0
x1 = origin+np.arange(len(bx1[:,0,0]))#*0.36250001 #;in Mm (z-axis)
x2 = origin+np.arange(len(bx1[0,:,0]))#*0.36250001
x3 = np.arange(len(bx1[0,0,:]))#*0.36250001 #(x-axis)
#'''

def onclick(event):
    global fig2, bx1,bx2,bx3,x1,x2,x3
    #if event.button == 3:
    if event.button is MouseButton.LEFT:
        x_value, y_value = event.xdata, event.ydata
        loc = np.array([x_value,y_value,0.0])# * 0.36250001
        pl = field_line(bx1,bx2,bx3, x1, x2, x3, seed=loc, method="RK2")
        px=pl[0,:]
        py=pl[1,:]
        pz=pl[2,:]
        bz_z = []
        for j in range(len(px)):
            try:i0 = np.where(px[j] < x1)[0][0]
            except:i0 = 0
            try:j0 = np.where(py[j] < x2)[0][0]
            except:j0=0
            bbb = np.interp(pz[j],x3,bx3[i0,j0,:])
            bz_z += [bbb]
        bz_z = np.array(bz_z)
        ind = np.where(pz > 0.0)[0]
        fig2,ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.plot(bz_z[ind],pz[ind],'*')
        plt.show()

#'''

import numpy as np

def value_locate(xbins, x):
    # Find the indices where x falls into bins
    indices = np.searchsorted(xbins, x)
    return indices

'''
# Define the input arrays
xbins = np.array([0, 3, 5, 6, 12])
x = np.array([2, 5, 8, 10])

# Find the indices for x in xbins
loc = value_locate(xbins, x)
print(loc)
'''
x11_a = [225,225,10,50,100]
y11_a = [229,229,20,50,100]
for i in range(1):
    fig = plt.subplots(1, 1, figsize=(6, 6))
    AllAxes=plt.gcf().get_axes()
    plt.imshow(bx3[:,:,0],origin='lower')
    #cid = fig[0].canvas.mpl_connect('motion_notify_event',onclick)
    #plt.show()
    
    x11=x11_a[i]#229
    y11=y11_a[i]#225
    loc = np.array([x11,y11,0.0])# * 0.36250001
    pl = field_line(bx1,bx2,bx3, x1, x2, x3, seed=loc, method="RK2")
    px=pl[0,:]
    py=pl[1,:]
    pz=pl[2,:]
    '''
    bz_z = []
    for j in range(len(px)):
        try:i0 = np.where(px[j] < x1)[0][0]
        except:i0 = 0
        try:j0 = np.where(py[j] < x2)[0][0]
        except:j0=0
        interp = interp1d(x3,bx3[i0,j0,:], fill_value="extrapolate")
        bbb = interp(pz[j])
        #bbb = np.interp(pz[j],x3,bx3[i0,j0,:])
        bz_z += [bbb]
    bz_z = np.array(bz_z)
    '''
    igx=value_locate (x1, px)
    igy=value_locate (x2, py)
    igz=value_locate (x3, pz)

    bz_z = bx3[igx,igy,igz]

    ind = np.where(pz > 0.0)[0]
    fig2 = plt.subplots(1, 1, figsize=(6, 6))
    plt.plot(bz_z[ind],pz[ind],'--')
    #plt.plot(pz[ind],'--');plt.plot(py[ind],'--');plt.plot(px[ind],'--')
    plt.show()
 
