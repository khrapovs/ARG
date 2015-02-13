#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Likelihood functions for various models.

"""

import numpy as np
import scipy.stats as st
import numdifftools as nd
from statsmodels.tsa.tsatools import lagmat

from .argparams import ARGparams

__all__ = ['likelihood_vol', 'likelihood_ret']


def likelihood_vol(theta, vol=None, **kwargs):
    """Log-likelihood for ARG(1) volatility model.

    Parameters
    ----------
    theta : array_like
        Model parameters. [scale, rho, delta]
    vol : (nobs, ) array
        Observable time series

    Returns
    -------
    float
        Value of the log-likelihood function

    """
    assert vol.shape[0] > 1, "Volatility series is too short!"
    if theta.min() <= 0:
        return 1e10
    param = ARGparams(theta_vol=theta)
    degf = param.delta * 2
    nonc = param.rho * vol[:-1] / param.scale * 2
    logf = st.ncx2.logpdf(vol[1:], degf, nonc, scale=param.scale/2)
    return -logf[~np.isnan(logf)].mean()


def likelihood_ret(theta_ret, ret=None, vol=None, param_vol=None):
    """Log-likelihood for return model.

    Parameters
    ----------
    theta : array_like
        Model parameters. [scale, rho, delta]
    vol : (nobs, ) array
        Observable time series
    param_vol : ARGparams instance
        Parameters of the volatlity model

    Returns
    -------
    logf : float
        Value of the log-likelihood function

    """
    [phi, price_ret] = theta_ret
    [scale, rho, delta] = param_vol.theta_vol
    # scale = vol.mean() * (1. - rho) / delta

    a = lambda u: rho * u / (1 + scale * u)
    b = lambda u: delta * np.log(1 + scale * u)

    k = (scale * (1 + rho))**(-.5)
    psi = phi * k + (price_ret - .5) * (1 - phi**2)
    # vol /= (1 - phi**2)

    vollag = lagmat(vol, 1).flatten()[1:]
    vol, ret = vol[1:], ret[1:]

    # tau1 = rho * psi + a(- phi * k)
    # tau2 = scale * delta * psi + b(- phi * k)
    # r_mean  = tau1 * vollag + tau2
    r_mean = psi * vol + a(- phi * k) * vollag + b(- phi * k)
    r_var = vol * (1 - phi**2)

    return - st.norm.logpdf(ret, r_mean, np.sqrt(r_var)).mean()
