#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARG parameters class

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['ARGparams']


class ARGparams(object):
    """Class for ARG model parameters.

    Attributes
    ----------
    scale : float
    rho : float
    delta : float
    beta : float
    theta : array

    Raises
    ------
    AssertionError

    """
    def __init__(self, scale=.001, rho=.9, delta=1.1,
                 tau1=.5, tau2=1, phi=-.5,
                 price_vol=-10., price_ret=10.,
                 theta_vol=None, theta_ret=None):
        """Initialize the class instance.

        Parameters
        ----------
        scale : float
            Scale of the volatility ARG(1) process
        rho : float
            Persistence of the volatility ARG(1) process
        delta : float
            Overdispersion of the volatility ARG(1) process
        phi : float
            Correlation between return and volatility
        price1 : float
            Volatiltiy risk price
        price2 : float
            Equity risk price
        theta_vol : array
            Parameters of the volatility model
        theta_ret : array
            Parameters of the return model

        """
        if not theta_vol is None:
            assert len(theta_vol) == 3, \
                "Wrong number of parameters in theta_vol!"
            [scale, rho, delta] = theta_vol
        # Volatililty parameters
        self.scale = scale
        self.rho = rho
        self.delta = delta
        assert scale > 0, "Scale must be greater than zero!"
        self.beta = self.rho / self.scale
        # Parameters of the volatility model
        self.theta_vol = np.array([scale, rho, delta])

        if not theta_ret is None:
            assert len(theta_ret) == 2, \
                "Wrong number of parameters in theta_vol!"
            [phi, price_ret] = theta_ret
        # Return parameters
        # Correlation between return and volatility
        self.phi = phi
        # Volatility risk price
        self.price_vol = price_vol
        # Equity risk price
        self.price_ret = price_ret
        # Parameters of the return model
        self.theta_ret = np.array([phi, price_ret])

    def __str__(self):
        """This is what is shown when you print() the instance.

        """
        params = (self.scale, self.rho, self.delta)
        string = "scale = %.2f, rho = %.2f, delta = %.2f" % params
        return string
