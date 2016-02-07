import numpy as np
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import grad

N = 60
x = np.linspace(-1.25, 1.25, num=N)
poly = np.array([1., -0.5, -1., 0.5]) #m, b
sigma = 0.3

y_true = np.polyval(poly, x)
d = y_true + np.random.normal(size = N, loc=0., scale=sigma)

def extract(lamb):
    """
    lambda = [mus, sigmas]
    """
    N = len(lamb)
    return lamb[:N/2], lamb[N/2:]

def logposterior(poly, sigma, x, d):
    """
    logposterior for polynomial model
    """
    N = len(x)
    model = mypolyval(poly, x)
    return (-np.sum((d[:, None] - model)**2/(2*sigma**2))
            - N/2*np.log(2*np.pi*sigma**2))

def logq(poly, lamb):
    """
    Trial distribution
    Product distribution for polynomial model
    """
    mus, sigs = extract(lamb) 
    norm = -np.log(2*np.pi*sigs**2)/2.
    return np.sum((poly - mus)**2/(2*sigs**2) + norm)

def F_dF(lamb, sigma, x, d, eps):
    """
    Variational lower bound and its gradient. I am using log
    parameters for sigmas of variational model to enforce 
    positivity.
    input:
        lamb: Array of lenght 2*N [mus, rhos=log(sigmas)]
            mus and sigmas of variational model
        sigma: noise estimate
        x: array of length N, 
            points at which polynomial is evaluated
        d: array of length N, data points
        eps: array of shape (batchsize, N)
            Noise sample from trial distribution
            standard normal sampling
    output:
        F, dF: variation free energy, its derivatives
    """
    mus, rhos = extract(lamb)
    N = len(mus)
    sigs = np.exp(rhos)
    thetas = mus[None, :] + sigs[None,:] * eps
    model = np.array([np.polyval(t, x) for t in thetas])
    r = d[None, :] - model
    f = np.mean(np.sum(r**2, 1)/(2*sigma**2), 0) - rhos.sum()
    df_mu = -(r.dot(np.array([x**(N-i-1) for i in range(N)]).T)/sigma**2)
    df_rho = df_mu*eps*sigs - 1
    return f, np.r_[df_mu.mean(0), df_rho.mean(0)]

def varbayes(d, x, order, sigma = sigma, batchsize = 2000, itnlim = 50,
             iprint = 0, lr = 0.001, alpha = 0.5, **kwargs):
    """
    Perform variational bayes to minimize D_KL(q || p) using
    stochastic gradient descent

    inputs:
        d: data (array of length N)
        x: points polynomial is evaluated at (array of length N)
        order: (int) order of polynomial
        sigma : (float) estimate of noise variance
        batchsize : number of samples to use to estimate gradient
        itnlim: number of iterations to attempt SGD
        iprint: (int) greater than 0 will print stuff
        lr: SGD step stie
        alpha: exponential moving average time constant
    kwargs:
        lamb0: initial mus and rhos=log(sigmas) for trial distn
    
    returns:
        lamb: final trial distribution parameters
    """
    lamb = kwargs.get('lamb0', np.random.randn(2*order))
    step = np.zeros_like(lamb)
    F_av = 0 
    for i in range(itnlim):
        eta = np.random.randn(batchsize, order)
        F, dF = F_dF(lamb, sigma, x, d, eta) 
        #online exponential moving average
        F_av += ((1-alpha)/(1-alpha**(i+1)))*(F - F_av)
        step += ((1-alpha)/(1-alpha**(i+1)))*(lr*dF - step)
        lamb -= step 
        if iprint:
            print("Itn {}: F = {}".format(i, F_av))

    return np.r_[lamb[:order], np.exp(lamb[order:])]

def plot_fit(lamb, x, d):
    m, s = extract(lamb)
    l, u = np.polyval(m-s,x), np.polyval(m+s,x)
    f = np.polyval(m, x)
    plt.plot(x, d, '.', label='data')
    plt.fill_between(x, l, u, alpha = 0.1)
    plt.plot(x, f, label='best fit')
    plt.legend(loc='best')
    return f

def plot_fit_whisker(lamb, x, d, samples=1000, true_p = poly):
    m, s = extract(lamb)
    for i in xrange(samples):
        f = np.polyval(m + np.random.randn(len(m))*s, x)
        plt.plot(x, f, 'k-', lw=1, alpha=0.05)

    plt.plot(x, d, 'o', label='data', mec='k', mfc='w', mew=1)

    f = np.polyval(m, x)
    plt.plot(x, f, label='best fit')

    f = np.polyval(np.polyfit(x,d, len(m)-1), x)
    plt.plot(x, f, label='polyfit')
    if true_p is not None:
        plt.plot(x, np.polyval(true_p, x), label='True')
    plt.legend(loc='best')
    return f

if __name__ == '__main__':
    lamb = varbayes(d, x, 4, iprint=2)
    plot_fit_whisker(lamb, x, d)
    plt.show()

