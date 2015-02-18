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

    argmodel = ARG()
    uarg = np.linspace(-50, 100, 100)
    argmodel.plot_abc(uarg, param)

    argmodel.plot_vsim(param=param)

    vol = argmodel.vsim_last(nsim=10, param=param)
    print(vol.shape)

    argmodel.plot_vlast_density(nsim=1000, param=param)


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
    argmodel = ARG()
    vol = argmodel.vsim(nsim=1, nobs=nobs, param=param)
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param)

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(vol.flatten())
    plt.subplot(2, 1, 2)
    plt.plot(ret.flatten())
    plt.show()


def estimate_mle_vol():
    """Try MLE estimator with volatility."""
    param_true = ARGparams()
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)
    param_final, results = argmodel.estimate_mle(param_start=param_true,
                                                 model='vol')

    print('True parameter:', param_true.get_theta_vol())
    print('Final parameter: ', param_final.get_theta_vol())
    print(results)

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
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true)
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true).flatten()
    vol = vol.flatten()
    argmodel.load_data(vol=vol, ret=ret)

    pfinal_vol, results_vol = argmodel.estimate_mle(param_start=param_true,
                                                    model='vol')

    pfinal_ret, results_ret = argmodel.estimate_mle(param_start=pfinal_vol,
                                                    model='ret')

    print(results_vol)
    print(results_ret)
    print(pfinal_vol.get_theta_vol())
    print(pfinal_ret.get_theta_ret())

    return pfinal_vol, pfinal_ret, results_vol, results_ret


def estimate_mle_joint():
    """Try MLE estimator with volatility and return."""
    rho = .9
    delta = 1.1
    dailymean = .2**2
    scale = dailymean * (1 - rho) / delta
    price_ret = .95
    phi = -.5

    param_true = ARGparams(scale=scale, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true)
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true).flatten()
    vol = vol.flatten()
    argmodel.load_data(vol=vol, ret=ret)

    param_final, results = argmodel.estimate_mle(param_start=param_true,
                                            model='joint')

    print(results)
    print('True parameter:', param_true.get_theta())
    print('Final parameter: ', param_final.get_theta())
    print('Tstat: ', results.tstat)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(vol, label='Vol')
    axes[0].plot(argmodel.ret_cvar(param_final), label='Predicted')
    axes[0].legend()
    axes[1].plot(ret, label='Ret')
    axes[1].plot(argmodel.ret_cmean(param_final), label='Predicted')
    axes[1].legend()
    plt.show()

    return param_final, results


def estimate_gmm_vol():
    """Try GMM estimator."""
    param_true = ARGparams()
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)

    param_start = ARGparams()
    param_start.update(theta_vol=param_true.get_theta_vol()*.9)

    uarg = np.linspace(1, 1000, 3) * 1j
    param_final, results = argmodel.estimate_gmm(uarg=uarg, zlag=2,
        param_start=param_start, model='vol')

    print('True parameter:', param_true)
    print('Final parameter: ', param_final)
    print('Std: ', results.tstat)
    results.print_results()

    plt.plot(vol, label='Data')
    plt.plot(argmodel.vol_cmean(param_final), label='Predicted')
    plt.legend()
    plt.show()

    return param_final, results


def estimate_gmm_ret():
    """Try GMM estimator."""

    rho = .9
    delta = 1.1
    dailymean = .01**2
    scale = dailymean * (1 - rho) / delta
    price_ret = .95
    phi = -.5

    param_true = ARGparams(scale=scale, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true).flatten()
    argmodel.load_data(ret=ret)

    uarg = np.linspace(.1, 10, 3) * 1
    param_final, results = argmodel.estimate_gmm(uarg=uarg, zlag=2,
        param_start=param_true, model='ret')

    print('True parameter:', param_true.get_theta_ret())
    print('Final parameter: ', param_final.get_theta_ret())
    print('Std: ', results.tstat)
    results.print_results()

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(vol, label='Vol')
    axes[0].plot(argmodel.ret_cvar(param_final), label='Predicted')
    axes[0].legend()
    axes[1].plot(ret, label='Ret')
    axes[1].plot(argmodel.ret_cmean(param_final), label='Predicted')
    axes[1].legend()
    plt.show()

    return param_final, results


def estimate_gmm_joint():
    """Try GMM estimator."""

    rho = .9
    delta = 1.1
    dailymean = .01**2
    scale = dailymean * (1 - rho) / delta
    price_ret = .95
    phi = -.5

    param_true = ARGparams(scale=scale, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true)
    argmodel.load_data(ret=ret)

    rho = .5
    delta = 1.5
    scale = dailymean * (1 - rho) / delta
    phi = -.1

    param_start = ARGparams(scale=scale, rho=rho, delta=delta,
                            phi=phi, price_ret=price_ret)

    uarg = np.linspace(.1, 10, 3) * 1j/10
    param_final, results = argmodel.estimate_gmm(uarg=uarg, zlag=2,
        param_start=param_start, model='joint')

    print('True parameter:', param_true.get_theta())
    print('Final parameter: ', param_final.get_theta())
    print('Std: ', results.tstat)
    results.print_results()

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(vol, label='Vol')
    axes[0].plot(argmodel.vol_cmean(param_final), label='Predicted')
    axes[0].legend()
    axes[1].plot(ret, label='Ret')
    axes[1].plot(argmodel.ret_cmean(param_final), label='Predicted')
    axes[1].legend()
    plt.show()

    return param_final, results


def plot_cf():
    """Plot characteristic functions.

    """
    rho = .9
    delta = 1.1
    dailymean = .01**2
    scale = dailymean * (1 - rho) / delta
    price_ret = .95
    phi = -.5

    param_true = ARGparams(scale=scale, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true)
    argmodel.load_data(ret=ret)

    uarg = np.linspace(1, 10000, 2) * 1j

    fix, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].plot(vol)
    axes[1].plot(np.exp(-uarg * vol[:, np.newaxis]).real)
    axes[2].plot(np.exp(-uarg * vol[:, np.newaxis]).imag)
    plt.show()


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)

#    play_with_arg()

#    try_simulation()

#    param_final, results = estimate_mle_vol()

#    pfinal_vol, pfinal_ret, results_vol, results_ret = estimate_mle_ret()

#    pfinal, results = estimate_mle_joint()

    param_final, results = estimate_gmm_vol()

#    param_final, results = estimate_gmm_ret()

#    param_final, results = estimate_gmm_joint()

#    plot_cf()

