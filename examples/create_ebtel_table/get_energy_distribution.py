import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


data = np.load('./../data/distributions_for_EBTEL.npz')
duration = data['duration']
delay = data['delay']
data_E_before = data['energy_before']
data_E_all = data['energy']#*unit_conv_fact
data_E_after = data['energy_after']

a0,b0=np.histogram(data_E_before[data_E_before>0]*1000,bins=2000)#bins='auto')
xE0 = (b0[1::]+b0[0:-1])/2.0
dx0 = (b0[1::]-b0[0:-1])
yE0 = a0#/dx0    

plt.step(xE0,yE0,label='Original Data',alpha=0.5)

def lognormal_pdf(x, mu, sigma, amplitude):
    return (amplitude / (x * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))

initial_guess = [10, 0.8, 5e8]

a0,b0=np.histogram(data_E_before[data_E_before>400]*1000,bins=2000)#bins='auto')
xE0 = (b0[1::]+b0[0:-1])/2.0
dx0 = (b0[1::]-b0[0:-1])
yE0 = a0#/dx0    
params, _ = curve_fit(lognormal_pdf, xE0,yE0, p0=initial_guess)


plt.plot(xE0,lognormal_pdf(xE0,*params),color='r',label='LogNorm:\n$\mu = $'
               +format('%0.2f'%params[0])
               +'\n$\sigma$ = '+format('%0.2f'%params[1])
               +'\nN = '+format('%0.1f'%params[-1]))

plt.ylabel('Total N')
plt.xlabel('E (J/m2)')
plt.yscale('log');plt.xscale('log')
plt.legend()
plt.show()

'''
a0,b0=np.histogram(duration,bins=500)#bins='auto')
xE0 = (b0[1::]+b0[0:-1])/2.0
dx0 = (b0[1::]-b0[0:-1])
yE0 = a0#/dx0    
plt.step(xE0,yE0)
plt.ylabel('Total N')
plt.xlabel('duration')
plt.yscale('log');plt.xscale('log')
plt.show()
'''
