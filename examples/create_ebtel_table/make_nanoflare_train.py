import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def Average_Heating_Rate(B,L, a=1.84, b=-1.79, c=2.25e-5):
    '''
    It will return the average peak heating rate of a nano-flare: Q ~ C * B^a * L^b.
    
    Inputs:
        B - is in Gauss
        L - is in Mm

    Outputs:
       <Q0> - average peak heating rate in erg/cm3/s

    --- Biswajit April/02/2026.
    '''
    Q0 = c * B**a * L**b #in J/m3/s
    Q0 = Q0 * 10 #in erg/cm3/s
    return Q0 #erg/cm3/s

def lognormal_pdf(x, mu, sigma, amplitude):
    return (amplitude / (x * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))
def broken_powerlaw(x, m1, m2, c, xb):
    """
    Broken power law:
    
    f(x) = c * x^m1               for x <= xb
         = c * xb^(m1-m2) * x^m2  for x > xb
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)

    # low-slope region
    mask1 = x <= xb
    y[mask1] = c * (x[mask1] ** m1)

    # high-slope region (continuous at xb)
    mask2 = x > xb
    y[mask2] = c * (xb ** (m1 - m2)) * (x[mask2] ** m2)

    return y

def random_sample_mixed_distribution(N,
                              mu, sigma, A, #log-normal parameters
                              m1, m2, c, xb,#broken polw parameters
                              xmin, xmax, xc,
                              ngrid=10000):

    '''
    Generate random number from a mixed distribution of log-normal and broken-powerlaw


    Biswajit -- Apr.29.2026
    '''

    # 1. Normalize lognormal part
    norm_logn, _ = quad(lognormal_pdf, xmin, xc, args=(mu, sigma, A))
    # 2. Normalize power law part
    norm_bpl, _ = quad(broken_powerlaw, xc, xmax, args=(m1, m2, c, xb))
    # total probability
    P_logn = norm_logn / (norm_logn + norm_bpl)
    # 3. grid sampling for inversion
    x1 = np.linspace(xmin, xc, ngrid)
    x2 = np.geomspace(xc, xmax, ngrid)

    pdf1 = lognormal_pdf(x1, mu, sigma, A)
    pdf2 = broken_powerlaw(x2, m1, m2, c, xb)

    #cdf1 = np.cumsum(pdf1)
    cdf1 = np.cumsum(pdf1) * (x1[1] - x1[0])
    cdf1 /= cdf1[-1]

    cdf2 = np.cumsum(pdf2)
    cdf2 /= cdf2[-1]

    inv_cdf1 = interp1d(cdf1, x1, bounds_error=False, fill_value=(xmin, xc))
    inv_cdf2 = interp1d(cdf2, x2, bounds_error=False, fill_value=(xc, xmax))

    # 4. sampling
    u = np.random.rand(N)
    samples = np.zeros(N)
    mask = u < P_logn

    samples[mask] = inv_cdf1(np.random.rand(np.sum(mask)))
    samples[~mask] = inv_cdf2(np.random.rand(np.sum(~mask)))

    return samples

def nanoflareprof_logNormal(
    mu=-2.435,          #log(E) of nanoflare energy in erg/cm3
    sigma = 0.907,      #sigma of log-normal distribution
    E_low = 0.02,       #Lower energy of log-normal distribition
    E_high = 1.4,       #Upper energy of log-normal distribition
    L_half = 50,        #Half loop length in Mm
    delay_param = [6.444, 1.7487, 3.806e4,  -2.699,-6.945,9.662e8, 2500, 10, 4860, 1000, 10000], #[mu, sigma, A, m1, m2, c, xb, xmin, xmax, xc, ngrid]**
    qbkg=1.0e-5,
    tau_half=20,
    dur=60000,
    HeatingFunction = False,
    seed = None,
    Test=False,
    ):
    '''
    Generates a sequence of nanoflares from a log-normal energy distribution of peak nanoflare 
    heating rate. The delay between successive events is random and follow a mixed-distribution.
 
    E ~ (1/(Sigma * x * np.sqrt(2*pi))) * exp^(-(ln(x) - mu)^2/2Sigma^2)

    Inputs:
      mu -       mean energy of log-normal energy distribution in the unit of erg/cm3/s
      sigma -       stander deviation of energy distribution
      qbkg -        steady background heating rate
      tau_half - half duration of nanoflare (s) (triangluar profile assumed)
      dur - duration of simulation (s)
      delay_param = [
                    mu,sigma,A,     #--> log-normal parameters
                    m1, m2, c, xb,  #--> broken-powerlaw parameters
                    min_delay, max_delay, broken_delay, 
                    ngrid,
                    ]
    Outputs:
      Peak_time - peak time of the events.
      Peak_heat - peak heating rate (erg cm^-3 s^-1)
      time -        time array (1 sec increment) # if HeatingFunction = True
      heat -        corresponding array of heating rate (erg cm^-3 s^-1) # if HeatingFunction = True

                                                                                                                                                                                            
    '''
    #Select event energy are selected from log-normal distribution in the unit of erg/cm2
    target_size = int(dur)
    q0 = np.array([])  # Initialize an empty array
    while len(q0) < target_size:
        needed = target_size - len(q0)
        new_samples = np.random.lognormal(mean=mu, sigma=sigma, size=needed * 2)
        valid_samples = new_samples[(new_samples >= E_low) & (new_samples <= E_high)]
        q0 = np.concatenate([q0, valid_samples])
    q0 = q0[:target_size]

    #Select random delay between two events from a minimum to maximum values from a mixed distribution

    delay_time = random_sample_mixed_distribution(len(q0),
                              *delay_param[0:-1],
                              ngrid=delay_param[-1])

    if Test:
        #Plot Energy distribution
        a,b=np.histogram(q0,bins='auto')
        xE = (b[1::]+b[0:-1])/2.0
        dx = (b[1::]-b[0:-1])
        yE = a/dx
        params_E, _ = curve_fit(lognormal_pdf, xE, yE, p0=[np.mean(np.log(q0)), np.std(np.log(q0)),len(q0)])
        #Plot delay distribution
        N = delay_time
        a,b,=np.histogram(N[N<delay_param[9]],bins='auto')
        delay = (b[1::]+b[0:-1])/2.0
        dx = (b[1::]-b[0:-1])
        n = a/dx
        initial_guess = [np.mean(np.log(delay)), 0.1, 1.0]
        params, _ = curve_fit(lognormal_pdf, delay,n, p0=initial_guess)

        a,b=np.histogram(N[N>delay_param[9]],bins='auto')
        delay2 = (b[1::]+b[0:-1])/2.0
        dx = (b[1::]-b[0:-1])
        n2 = a/dx

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0,0].set_title('InPut')
        axs[0,1].set_title('InPut')

        axs[0,0].step(xE,yE)
        axs[0,0].set_xscale('log')
        axs[0,0].set_yscale('log')
        axs[0,0].set_xlabel('erg cm$^{-3}$ s$^{-1}$')
        axs[0,0].set_ylabel('N/bin')

        x_plot = np.linspace(min(q0), max(q0), 100)
        axs[0,0].plot(x_plot, lognormal_pdf(x_plot,*params_E), label='LogNorm:\n$\mu = $'
               +format('%0.2f'%params_E[0])
               +'\n$\sigma$ = '+format('%0.2f'%params_E[1])
               +'\nN = '+format('%0.1f'%params_E[-1]))

        axs[0,1].scatter(delay,n)
        axs[0,1].scatter(delay2,n2)

        x_plot = np.linspace(min(delay), max(delay), 100)
        axs[0,1].plot(x_plot, lognormal_pdf(x_plot, *params), 'r-', label='LogNorm:\n$\mu = $'+format('%0.2f'%params[0])+'\n$\sigma$ = '+format('%0.2f'%params[1])+'\nN = '+format('%0.1f'%params[2]))

        x_plot = np.linspace(min(delay2), max(delay2), 100)

        params2, _ = curve_fit(broken_powerlaw, delay2,n2, p0=[-1,-1.5,1,3000],
            bounds=([-8,-8,0,2500],[-0.1,-0.1,np.inf,4000])
        )
        axs[0,1].plot(x_plot, broken_powerlaw(x_plot,*params2), label='bPolw:\nm1='+format('%0.2f'%params2[0])
             +'\nm2='+format('%0.2f'%params2[1])
             +'\nc='+format('%0.2f'%params2[2])
             +'\nEb='+format('%0.2f'%params2[3])
             )
        axs[0,1].set_yscale('log'); axs[0,1].set_xscale('log')
        axs[0,1].set_ylabel('N/bin')
        axs[0,1].set_xlabel('Delay')
        axs[0,0].legend()
        axs[0,1].legend()
    time = np.arange(dur + 1)
    if HeatingFunction is True : heat = np.zeros(int(dur + 1))

    #delay_arr = np.zeros(num_nano - 1)
    #delay_good = np.zeros(num_nano - 1)
    #seed = !NULL
    if seed is not None : np.random.seed(seed)
    t1 = int(100*np.random.uniform(low=0.0, high=1.0, size=1)[0])   # first nanoflare begins randomly in the first 100 s
    if HeatingFunction is True :
        for i in range(int(tau_half+1)): heat[t1+i] = q0[0]*i/tau_half  #;                   triangular profile rise
        for i in range(int(tau_half+1), int((2*tau_half)+1)): heat[t1+i] = q0[0]*(2.*tau_half - i)/tau_half  #;  decay

    Peak_heat = [q0[0]] #peak heating rate of each triangular profile
    Peak_time = [t1+tau_half] #peak time

    delay_taken = [delay_time[0]]
    k = 0
    tnew = t1 + delay_time[0]
    #delay_arr[0] = tnew - t1

    while (tnew+2*tau_half < dur):
        k = k + 1
        if HeatingFunction is True :
            for i in range(int(tau_half+1)): heat[int(tnew+i)] = heat[int(tnew+i)] + q0[k]*i/tau_half
            for i in range(int(tau_half+1), int((2*tau_half)+1)) : heat[int(tnew+i)] = heat[int(tnew+i)] + q0[k]*(2.*tau_half - i)/tau_half

        Peak_heat += [q0[k]]
        Peak_time += [tnew+tau_half]
        told = tnew
        tnew = told + delay_time[k]
        delay_taken += [delay_time[k]]
        #delay_arr[k] = tnew - told
        #if (tnew >= 10000): delay_good[k] = tnew - told

    delay_taken = np.array(delay_taken)
    if HeatingFunction is True :
        h_cor = L_half*1.0e8 #5.e9  #;  coronal scale height
        heat = heat + qbkg
        mean_heat = np.mean(heat[0:int(dur)])
        Mean_energy_flux = mean_heat*h_cor #erg/cm2/s

    #ss = np.where(delay_good != 0.)
    #delay_good = delay_good[ss]
    mean_delay = np.mean(delay_time)
    median_delay = np.median(delay_time)

    PrintOut = True
    if PrintOut == True:
        print(' ')
        if HeatingFunction is True : print('mean energy flux (1.0e7 erg/cm2/s)= ', Mean_energy_flux/1.0e7)
        print('mean delay = ', mean_delay)
        print('median delay = ', median_delay)


    if Test:
        a,b=np.histogram(np.array(Peak_heat),bins='auto')
        xE = (b[1::]+b[0:-1])/2.0
        dx = (b[1::]-b[0:-1])
        yE = a/dx
        params_E, _ = curve_fit(lognormal_pdf, xE, yE, p0=[np.mean(np.log(Peak_heat)), np.std(np.log(Peak_heat)),len(Peak_heat)])
        #fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[1,0].set_title('OutPut')
        axs[1,1].set_title('OutPut')

        axs[1,0].step(xE,yE)
        x_plot = np.linspace(min(Peak_heat), max(Peak_heat), 100)
        axs[1,0].plot(x_plot, lognormal_pdf(x_plot,*params_E), label='LogNorm:\n$\mu = $'
               +format('%0.2f'%params_E[0])
               +'\n$\sigma$ = '+format('%0.2f'%params_E[1])
               +'\nN = '+format('%0.1f'%params_E[-1]))

        axs[1,0].set_xscale('log')
        axs[1,0].set_yscale('log')
        axs[1,0].set_xlabel('erg cm$^{-3}$ s$^{-1}$')
        axs[1,0].set_ylabel('N/bin')

        #Plot delay distribution
        N = delay_taken
        a,b,=np.histogram(N[N<delay_param[9]],bins='auto')
        delay = (b[1::]+b[0:-1])/2.0
        dx = (b[1::]-b[0:-1])
        n = a/dx

        initial_guess = [np.mean(np.log(delay)), 0.1, 1.0]
        params, _ = curve_fit(lognormal_pdf, delay,n, p0=initial_guess)

        a,b=np.histogram(N[N>delay_param[9]],bins='auto')
        delay2 = (b[1::]+b[0:-1])/2.0
        dx = (b[1::]-b[0:-1])
        n2 = a/dx
        axs[1,1].scatter(delay,n)
        axs[1,1].scatter(delay2,n2)

        x_plot = np.linspace(min(delay), max(delay), 100)
        axs[1,1].plot(x_plot, lognormal_pdf(x_plot, *params), 'r-', label='LogNorm:\n$\mu = $'+format('%0.2f'%params[0])+'\n$\sigma$ = '+format('%0.2f'%params[1])+'\nN = '+format('%0.1f'%params[2]))

        x_plot = np.linspace(min(delay2), max(delay2), 100)

        params2, _ = curve_fit(broken_powerlaw, delay2,n2, p0=[-1,-1.5,1,3000],
            bounds=([-8,-8,0,2500],[-0.1,-0.1,np.inf,4000])
        )
        axs[1,1].plot(x_plot, broken_powerlaw(x_plot,*params2), label='bPolw:\nm1='+format('%0.2f'%params2[0])
             +'\nm2='+format('%0.2f'%params2[1])
             +'\nc='+format('%0.2f'%params2[2])
             +'\nEb='+format('%0.2f'%params2[3])
             )
        axs[1,1].set_yscale('log'); axs[1,1].set_xscale('log')
        axs[1,1].set_ylabel('N/bin')
        axs[1,1].set_xlabel('Delay')
        axs[1,0].legend()
        axs[1,1].legend()

    if Test or Test: plt.show()
    if HeatingFunction is True :
        if Test:
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            plt.plot(time,heat)
            plt.xlabel('Time (s)')
            plt.ylabel('erg cm$^{-3}$ s$^{-1}$')
            plt.show()
        return time,heat,np.array(Peak_time),np.array(Peak_heat),Mean_energy_flux
    else:
        return np.array(Peak_time),np.array(Peak_heat)


mu=12.84#13.02 #12.99#6.082#2.641
sigma = 1.1#0.91#0.907#0.394
norm = 4.835#2.1
dur=18816000#100000 #Original Multi-strand simulation were run for a duration of 33e5s
L_half = 48
tau=40

#distribution is in the unit of J/m2 from Shanwlee_2025
unit_conv_fact = 1000/(L_half*1.0e8) #erg/cm3
mu = mu + np.log(unit_conv_fact)

time,heat, Peak_time, Peak_heat, Mean_energy_flux = nanoflareprof_logNormal(
   mu=mu,		#log(E) of nanoflare energy
   sigma = sigma,	#sigma of log-normal distribution
   E_low = 0.02, 	#Lower log(energy) of log-normal distribition
   E_high = 3.0,	##Upper log(energy) of log-normal distribition
   L_half=L_half,	#Half loop length in Mm
   #delay_param = [6.444, 1.7487, 3.806e4,  -2.699,-6.945,9.662e8, 2500, 10, 4860, 1000, 10000],
   delay_param = [
                   6.31786474e+00, 1.71177384e+00, 3.30854734e+04,
                   -2.70969539e+00, -7.10910333e+00,  9.53518227e+08,                
                   2500, 10, 5000, 1000, 100000],
   qbkg=1.0e-5,
   tau_half=tau/2, 
   dur=dur, 
   HeatingFunction = True,
   Test=True,
   )


#Cross-check the energy distribution with original Shanwlee's distribution

data = np.load('./../data/distributions_for_EBTEL.npz')
data_N = data['energy_before']
data_E = data['energy']#*unit_conv_fact

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs=[axs]
a,b=np.histogram(np.array(Peak_heat)/unit_conv_fact,bins='auto')
xE = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
yE = a/dx    
axs[0].set_title('Energy distribution')

a0,b0=np.histogram(data['energy']*1000,bins='auto')
xE0 = (b0[1::]+b0[0:-1])/2.0
dx0 = (b0[1::]-b0[0:-1])
yE0 = a0/dx0    

#axs[0].step(data_E,data_N,label='Original Data')
axs[0].step(xE0,yE0,label='Original Data')

axs[0].step(xE,yE, alpha=0.3, label = 'New')#$\mu$:'+format('%0.4f'%(10**mu))+', L:'+format('%0.2f'%(2*L_half)))



axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('J m$^{-2}$')
axs[0].set_ylabel('N')
axs[0].legend()

plt.show()


#Cross-check the delay distribution with original Shanwlee's distribution

#delay_data = './../data/delay_data_multistrand.npz'
#data = np.load(delay_data)

delay_min = 0
delay_max = 1000

#delay_ = data['delay'][(data['delay']>delay_min) & (data['delay']<delay_max)]
delay_ = data['delay']

a,b,_=plt.hist(delay_,bins='auto')
plt.close()
delay = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
n = a/dx

#delay_high = data['delay'][(data['delay']>=delay_max)]
#a,b,_=plt.hist(delay_high,bins=20)
#plt.close()
#delay2 = (b[1::]+b[0:-1])/2.0
#dx = (b[1::]-b[0:-1])
#n2 = a/dx

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs=[axs]

# Plot original distribution
plt.scatter(delay,n, label='Original Data')
#plt.scatter(delay2,n2)#, label='Data2')

#plot new distribution
new_delay = Peak_time[1::] - Peak_time[0:-1]
a,b=np.histogram(np.array(new_delay),bins='auto')
delay__ = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
n__ = a/dx
plt.scatter(delay__,n__,label='New')


plt.yscale('log'); plt.xscale('log')
plt.ylabel('N')
plt.xlabel('Delay')
plt.legend()
plt.show()
