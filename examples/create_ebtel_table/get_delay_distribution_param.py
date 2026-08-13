'''
Estimate delay parameters.

Biswajit Apr.29.2026
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d


delay_data = './data/distributions_for_EBTEL.npz'

data = np.load(delay_data)

delay_min = 0
delay_max = 1000

#delay_ = np.log10(data['delay'][data['delay']>delay_min])
delay_ = data['delay'][(data['delay']>delay_min) & (data['delay']<delay_max)]

a,b,_=plt.hist(delay_,bins='auto')
plt.close()
delay = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
n = a/dx

delay_high = data['delay'][(data['delay']>delay_max)]
a,b,_=plt.hist(delay_high,bins=20)
plt.close()
delay2 = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
n2 = a/dx

# Define the lognormal PDF function with amplitude scaling
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

initial_guess = [np.mean(np.log(delay)), 0.1, 1.0]

params, _ = curve_fit(lognormal_pdf, delay,n, p0=initial_guess)

fit_mu, fit_sigma, fit_amp = params
print(f"Fit Results: mu={fit_mu:.4f}, sigma={fit_sigma:.4f}, amp={fit_amp:.4f}")


# Plotting
plt.scatter(delay,n)#, label='Data')
plt.scatter(delay2,n2)#, label='Data2')
#plt.axvline(x=delay[ind],color='k',alpha=0.5,ls='--')
#plt.axvline(x=delay[ind2-1],color='k',alpha=0.5,ls='--')

x_plot = np.linspace(min(delay), max(delay), 100)
plt.plot(x_plot, lognormal_pdf(x_plot, *params), 'r-', label='Fit:\n$\mu = $'+format('%0.2f'%fit_mu)+'\n$\sigma$ = '+format('%0.2f'%fit_sigma)+'\nN = '+format('%0.1f'%fit_amp))
plt.legend()

x_plot = np.linspace(min(delay2), max(delay2), 100)

plt.plot(x_plot, lognormal_pdf(x_plot, *params), 'r--')

params2, _ = curve_fit(broken_powerlaw, delay2,n2, p0=[-1,-1.5,1,3000],
    bounds=([-8,-8,0,2500],[-0.1,-0.1,np.inf,4000])
)
plt.plot(x_plot, broken_powerlaw(x_plot,*params2))
plt.yscale('log'); plt.xscale('log')
plt.ylabel('N')
plt.xlabel('Delay')

print(params)
print(params2)

plt.show()


# --- SAMPLER ---
def random_sample_mixed_distribution(N,
                              mu, sigma, A,
                              m1, m2, c, xb,
                              xmin, xmax, xc,
                              ngrid=5000):

    # 1. Normalize lognormal part
    norm_logn, _ = quad(lognormal_pdf, xmin, xc, args=(mu, sigma, A))
    # 2. Normalize power law part
    norm_bpl, _ = quad(broken_powerlaw, xc, xmax, args=(m1, m2, c, xb))
    # total probability
    P_logn = norm_logn / (norm_logn + norm_bpl)
    # 3. grid sampling for inversion
    x1 = np.linspace(xmin, xc, ngrid)
    #x2 = np.linspace(xc, xmax, ngrid)
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

N = random_sample_mixed_distribution(28000,
                              *params,
                              *params2,
                              10, 4860, 1000,
                              ngrid=5000)

a,b,_=plt.hist(N[N<delay_max],bins='auto')
plt.close()
delay = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
n = a/dx

initial_guess = [np.mean(np.log(delay)), 0.1, 1.0]
params, _ = curve_fit(lognormal_pdf, delay,n, p0=initial_guess)

a,b,_=plt.hist(N[N>delay_max],bins='auto')
plt.close()
delay2 = (b[1::]+b[0:-1])/2.0
dx = (b[1::]-b[0:-1])
n2 = a/dx

plt.scatter(delay,n)
plt.scatter(delay2,n2)

x_plot = np.linspace(min(delay), max(delay), 100)
plt.plot(x_plot, lognormal_pdf(x_plot, *params), 'r-', label='Fit:\n$\mu = $'+format('%0.2f'%fit_mu)+'\n$\sigma$ = '+format('%0.2f'%fit_sigma)+'\nN = '+format('%0.1f'%fit_amp))
plt.legend()   

x_plot = np.linspace(min(delay2), max(delay2), 100)

params2, _ = curve_fit(broken_powerlaw, delay2,n2, p0=[-1,-1.5,1,3000],
    bounds=([-8,-8,0,2500],[-0.1,-0.1,np.inf,4000])
)
plt.plot(x_plot, broken_powerlaw(x_plot,*params2))
plt.yscale('log'); plt.xscale('log')
plt.ylabel('N')
plt.xlabel('Delay')


plt.show()
