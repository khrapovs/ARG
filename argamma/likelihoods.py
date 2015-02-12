#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Likelihood functions for various models.

"""

import numpy as np
import scipy.stats as st
import numdifftools as nd

from .argparams import ARGparams

__all__ = ['likelihood_vol', 'likelihood_vol_grad', 'likelihood_vol_hess']


def likelihood_vol(theta, vol):
    """Log-likelihood for ARG(1) model.

    Parameters
    ----------
    theta : array_like
        Model parameters. [scale, rho, delta]
    vol : (nobs, ) array
        Observable time series

    Returns
    -------
    logf : float
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


def likelihood_vol_grad(theta, vol):
    """Gradient of the log-likelihood for ARG(1) model.

    Parameters
    ----------
    theta : list
        Model parameters. [scale, rho, delta]
    vol : (nobs, ) array
        Observable time series

    Returns
    -------
    array
        Gradient of the log-likelihood function

    """
    return nd.Gradient(lambda theta: likelihood_vol(theta, vol))(theta)


def likelihood_vol_hess(theta, vol):
    """Hessian for the log-likelihood for ARG(1) model.

    Parameters
    ----------
    theta : list
        Model parameters. [scale, rho, delta]
    vol : (nobs, ) array
        Observable time series

    Returns
    -------
    array
        Hessian of the log-likelihood function

    """
    return nd.Hessian(lambda theta: likelihood_vol(theta, vol))(theta)
