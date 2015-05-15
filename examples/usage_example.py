#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from argamma import ARG, ARGparams
from argamma.impvol import imp_vol, lfmoneyness, impvol_bisection

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


__all__ = ['plot_smiles']


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
    mean = .2**2 / 365
    price_vol, price_ret = -16, .95
    phi = -.9

    param = ARGparams(mean=mean, rho=rho, delta=delta,
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
    mean = .2**2
    price_vol, price_ret = -16, .95
    phi = -.5

    param_true = ARGparams(mean=mean, rho=rho, delta=delta,
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
    print('True parameter: ', param_true.get_theta_ret())
    print('Final parameter: ', pfinal_ret.get_theta_ret())

    return pfinal_vol, pfinal_ret, results_vol, results_ret


def estimate_mle_joint():
    """Try MLE estimator with volatility and return."""
    rho = .9
    delta = 1.1
    mean = .2**2
    price_ret = .95
    phi = -.5

    param_true = ARGparams(mean=mean, rho=rho, delta=delta,
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
    print('Tstat: ', results.tstat)
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
    mean = .01**2
    price_ret = .95
    phi = -.5

    param_true = ARGparams(mean=mean, rho=rho, delta=delta,
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
    print('Tstat: ', results.tstat)
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
    mean = .01**2
    price_ret = .95
    phi = -.5

    param_true = ARGparams(mean=mean, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true)
    argmodel.load_data(ret=ret)

    rho = .5
    delta = 1.5
    phi = -.1

    param_start = ARGparams(mean=mean, rho=rho, delta=delta,
                            phi=phi, price_ret=price_ret)

    uarg = np.linspace(.1, 10, 3) * 1j/10
    param_final, results = argmodel.estimate_gmm(uarg=uarg, zlag=2,
        param_start=param_start, model='joint')

    print('True parameter:', param_true.get_theta())
    print('Final parameter: ', param_final.get_theta())
    print('Tstat: ', results.tstat)
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
    mean = .01**2
    price_ret = .95
    phi = -.5

    param_true = ARGparams(mean=mean, rho=rho, delta=delta,
                           phi=phi, price_ret=price_ret)
    argmodel = ARG()
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
    argmodel.load_data(vol=vol)
    ret = argmodel.rsim(param=param_true)
    argmodel.load_data(ret=ret)

    uarg = np.linspace(.1, 100, 2) * 1j

    fix, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].plot(vol)
    axes[1].plot(np.exp(-uarg * vol[:, np.newaxis]).real)
    axes[2].plot(np.exp(-uarg * vol[:, np.newaxis]).imag)
    plt.show()


def plot_smiles(fname=None):
    """Plot model-implied volatility smiles.

    """
    price = 1
    nobs = 100
    moneyness = np.linspace(-.1, .1, nobs)
    riskfree, maturity = .0, 30/365
    call = np.ones_like(moneyness).astype(bool)
    call[moneyness < 0] = False
    current_vol = .2**2/365

    rho = .9
    delta = 1.1
    phi = -.5
    price_vol = -.1
    price_ret = .5

    points = 5

    sns.set_palette(sns.color_palette("binary_r", desat=.5))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    loc = 'upper center'

    phis = np.linspace(0, -.5, points)

    for phi in phis:
        param = ARGparams(mean=current_vol, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret, price_vol=price_vol)
        argmodel = ARG(param=param)

        premium = argmodel.option_premium(vol=current_vol, moneyness=moneyness,
                                          maturity=maturity,
                                          riskfree=riskfree, call=call)

        vol = impvol_bisection(moneyness, maturity, premium/price, call)
        axes[0].plot(moneyness*100, vol*100, label=str(phi))

    axes[0].legend(title='Leverage, $\phi$', loc=loc)
    axes[0].set_xlabel('Log-forward moneyness, $\log(F/S)$, %')
    axes[0].set_ylabel('Implied volatility, annualized %')

    maturities = np.linspace(30, 90, points) / 365
    phi = 0.

    for matur in maturities:
        param = ARGparams(mean=current_vol, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret, price_vol=price_vol)
        argmodel = ARG(param=param)

        premium = argmodel.option_premium(vol=current_vol, moneyness=moneyness,
                                          maturity=matur,
                                          riskfree=riskfree, call=call)

        vol = impvol_bisection(moneyness, matur, premium/price, call)
        axes[1].plot(moneyness*100, vol*100, label=str(int(matur*365)))

    axes[1].legend(title='Maturity, $T$', loc=loc)
    axes[1].set_xlabel('Log-forward moneyness, $\log(F/S)$, %')
    axes[1].set_ylabel('Implied volatility, annualized %')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.show()


def plot_outofthemoney():
    """Plot model-implied out-of-the-money premium.

    """
    nobs = 200
    moneyness = np.linspace(-.2, .2, nobs)
    riskfree, maturity = .1, 30/365
    call = np.ones_like(moneyness).astype(bool)
    call[moneyness < 0] = False
    current_vol = .2**2/365

    rho = .9
    delta = 1.1
    phi = -.5
    price_vol = -1000
    price_ret = .6

    param = ARGparams(mean=current_vol, rho=rho, delta=delta,
                      phi=phi, price_ret=price_ret, price_vol=price_vol)
    argmodel = ARG(param=param)


    premium = argmodel.option_premium(vol=current_vol, moneyness=moneyness,
                                      maturity=maturity,
                                      riskfree=riskfree, call=call)
    vol = impvol_bisection(moneyness, maturity, premium, call)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(moneyness, premium, label='premium')
    axes[1].plot(moneyness, vol, label='impvol')
    axes[0].legend()
    axes[1].legend()
    plt.show()


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')

#    play_with_arg()

#    try_simulation()

#    param_final, results = estimate_mle_vol()

#    pfinal_vol, pfinal_ret, results_vol, results_ret = estimate_mle_ret()

#    pfinal, results = estimate_mle_joint()

#    param_final, results = estimate_gmm_vol()

#    param_final, results = estimate_gmm_ret()

#    param_final, results = estimate_gmm_joint()

#    plot_cf()

    plot_smiles()

#    plot_outofthemoney()
