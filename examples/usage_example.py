#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from argamma import ARG, ARGparams

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


def play_with_arg():
    param = ARGparams()
    print(param)

    argmodel = ARG(param=param)
    uarg = np.linspace(-50, 100, 100)
    argmodel.plot_abc(uarg)

    argmodel.plot_vsim()

    vol = argmodel.vsim_last(nsim=10)
    print(vol.shape)

    argmodel.plot_vlast_density(nsim=1000)


def try_simulation():

    rho = .9
    delta = .75
    dailymean = .2**2 / 365
    scale = dailymean * (1 - rho) / delta
    price_vol, price_ret = -16, .95

    phi = -.9

    param = ARGparams(scale=scale, rho=rho, delta=delta,
                      phi=phi, price_ret=price_ret)
    nobs = 500
    argmodel = ARG(param=param)
    vol = argmodel.vsim(nsim=1, nobs=nobs)
    ret = argmodel.rsim(vol)

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(vol.flatten())
    plt.subplot(2, 1, 2)
    plt.plot(ret.flatten())
    plt.show()


def estimate_mle():
    """Try MLE estimator."""
    param_true = ARGparams()
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
    param_final, results = argmodel.estimate_mle(param_start=param_true,
                                                 vol=vol)

    print('True parameter:', param_true)
    print('Final parameter: ', param_final)
    print(type(results))

    return param_final, results


def estimate_gmm():
    """Try GMM estimator."""
    param_true = ARGparams()
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
    uarg = np.linspace(.1, 10, 3) * 1j
    results = argmodel.estimate_gmm(param_start=param_true.theta_vol,
                                    vol=vol, uarg=uarg, zlag=2)

    print('True parameter:', param_true)
    print('Final parameter: ', ARGparams(theta_vol=results.theta))
    print('Std: ', results.tstat)
    results.print_results()


if __name__ == '__main__':

    #play_with_arg()
    param_final, results = estimate_mle()
    print(results.x)
    print(results.std_theta)
    print(results.tstat)
    estimate_gmm()
    try_simulation()
