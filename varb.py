import numpy as np
import matplotlib.pyplot as plt

N = 30
P = 1
x = np.linspace(0, 1, num=N)
poly = [4., 2., 1.] #m, b
sigma = 0.3

y_true = np.polyval(poly, x)
d = y_true + np.random.normal(size = N, loc=0., scale=sigma)

def sig(rho):
    return np.log(1.+np.exp(rho))

def rho(sig):
    return np.log(np.exp(np.abs(sig))-1.)

def extract(lamb):
    N = len(lamb)
    return lamb[:N/2], lamb[N/2:]

def logposterior(poly, sigma, x):
    N = len(x)
    return (-(d - np.polyval(poly,x))**2/(2*sigma**2) 
            - N/2*np.log(2*np.pi*sigma**2))

def logq(poly, lamb):
    mus, sigs = extract(lamb) 
    norm = -np.log(2*np.pi*sigs**2)/2.
    return np.sum((poly - mus)**2/(2*sigs**2) + norm)

def grad_dkl(sigma, x, d, lamb, eps):
    mus, rhos = extract(lamb)
    sigs = sig(rhos)
    poly = mus + sigs * eps
    thetas = np.array([np.polyval(p, x) for p in poly])
    x_to_k = x[:,None] ** np.arange(poly.shape[1])[None, :]
    dkl_mu = -((d - thetas).dot(x_to_k)/sigma**2).mean(0)
    r = (poly - mus)/sigs
    dkl_rho = ((eps*(r/sigs + dkl_mu) 
               + (r**2 + 2)/sigs)/(1.+np.exp(rhos))).mean(0)
    return np.r_[dkl_mu, dkl_rho]

def varbayes(d, x, N, sigma = sigma, batchsize = 12, itnlim = 10,
             alpha = 0.1):
    lamb = np.random.randn(2*N)
    for i in range(itnlim):
        eps = np.random.randn(batchsize, N)
        d_dkl = grad_dkl(sigma, x, d, lamb, eps)
        step = d_dkl
        lamb -= alpha * step
    return np.r_[lamb[:N], sig(lamb[N:])]

def plot_fit(lamb, x, d):
    m, s = extract(lamb)
    l, u = np.polyval(m-s,x), np.polyval(m+s,x)
    f = np.polyval(m, x)
    plt.plot(x, d, '.', label='data')
    plt.fill_between(x, l, u, alpha = 0.1)
    plt.plot(x, f, label='best fit')
    plt.legend(loc='best')
    return f

def plot_fit_whisker(lamb, x, d, samples=1000):
    m, s = extract(lamb)
    for i in xrange(samples):
        f = np.polyval(m + np.random.randn(len(m))*s, x)
        plt.plot(x, f, 'k-', lw=1, alpha=0.05)

    plt.plot(x, d, 'o', label='data', mec='k', mfc='w', mew=1)

    f = np.polyval(m, x)
    plt.plot(x, f, label='best fit')

    f = np.polyval(np.polyfit(x,d, len(m)-1), x)
    plt.plot(x, f, label='polyfit')

    plt.legend(loc='best')
    return f


