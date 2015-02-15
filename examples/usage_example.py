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


def estimate_mle_vol():
    """Try MLE estimator with volatility."""
    param_true = ARGparams()
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
    argmodel.load_data(vol=vol)
    param_final, results = argmodel.estimate_mle(param_start=param_true,
                                                 model='vol')

    print('True parameter:', param_true)
    print('Final parameter: ', param_final)
    print(type(results))

    return param_final, results


def estimate_mle_ret():
    """Try MLE estimator with return."""
    rho = .9
    delta = 1.1
    dailymean = .2**2
    scale = dailymean * (1 - rho) / delta
    price_vol, price_ret = -16, .95
    phi = -.5

    param_true = ARGparams(scale=scale, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs)
    ret = argmodel.rsim(vol=vol).flatten()
    vol = vol.flatten()
    argmodel.load_data(vol=vol, ret=ret)

    pfinal_vol, results_vol = argmodel.estimate_mle(param_start=param_true,
                                                    model='vol')

    pfinal_ret, results_ret = argmodel.estimate_mle(param_start=pfinal_vol,
                                                    model='ret')

    return pfinal_vol, pfinal_ret, results_vol, results_ret


def estimate_mle_joint():
    """Try MLE estimator with volatility and return."""
    rho = .9
    delta = 1.1
    dailymean = .2**2
    scale = dailymean * (1 - rho) / delta
    price_vol, price_ret = -16, .95
    phi = -.5

    param_true = ARGparams(scale=scale, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs)
    ret = argmodel.rsim(vol=vol).flatten()
    vol = vol.flatten()
    argmodel.load_data(vol=vol, ret=ret)

    pfinal, results = argmodel.estimate_mle(param_start=param_true,
                                            model='joint')

    print(results)
    print(pfinal.get_theta())

    return pfinal, results


def estimate_gmm():
    """Try GMM estimator."""
    param_true = ARGparams()
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
    uarg = np.linspace(.1, 10, 3) * 1j
    param_final, results = argmodel.estimate_gmm(
        param_start=param_true.get_theta_vol(), vol=vol, uarg=uarg, zlag=2)

    print('True parameter:', param_true)
    print('Final parameter: ', param_final)
    print('Std: ', results.tstat)
    results.print_results()


if __name__ == '__main__':

    #play_with_arg()
    #try_simulation()

#    param_final, results = estimate_mle_vol()
#    print(results.x)
#    print(results.std_theta)
#    print(results.tstat)

#    pfinal_vol, pfinal_ret, results_vol, results_ret = estimate_mle_ret()
#
#    print(results_vol)
#    print(results_ret)
#    print(pfinal_vol.get_theta_vol())
#    print(pfinal_ret.get_theta_ret())

    pfinal, results = estimate_mle_joint()
#
#    estimate_gmm()
#
