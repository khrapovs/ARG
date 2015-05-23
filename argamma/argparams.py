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
    phi : float
    price_vol : float
    price_ret : float

    Raises
    ------
    AssertionError
    ValueError

    """
    def __init__(self, mean=.001, rho=.9, delta=1.1, phi=-.5,
                 price_vol=-1., price_ret=1.):
        """Initialize the class instance.

        Parameters
        ----------
        mean : float
            Mean of the volatility ARG(1) process
        rho : float
            Persistence of the volatility ARG(1) process
        delta : float
            Overdispersion of the volatility ARG(1) process
        phi : float
            Correlation between return and volatility (leverage)
        price_vol : float
            Volatiltiy risk price
        price_ret : float
            Equity risk price

        """
        # Volatililty parameters
        self.mean = mean
        self.rho = rho
        self.delta = delta
        self.scale = self.get_scale()
        assert mean > 0, "Mean must be greater than zero!"

        # Return parameters
        # Correlation between return and volatility (leverage)
        assert abs(phi) < 1, "Leverage must be inside (-1, 1)!"
        self.phi = phi
        # Volatility risk price
        self.price_vol = price_vol
        # Equity risk price
        self.price_ret = price_ret

    def update(self, theta_vol=None, theta_ret=None, theta=None):
        """Update model parameters from vectors.

        Parameters
        ----------
        theta_vol : array
            Parameters of the volatility model
        theta_ret : array
            Parameters of the return model

        """

        if theta_vol is not None:
            assert len(theta_vol) == 3, \
                "Wrong number of parameters in theta_vol!"
            if np.array(theta_vol).min() <= 0 or theta_vol[1] >= 1:
                raise ValueError("Inadmissible parameters!")
            else:
                [self.mean, self.rho, self.delta] = theta_vol

        if theta_ret is not None:
            assert len(theta_ret) == 2, \
                "Wrong number of parameters in theta_ret!"
            if abs(theta_ret[0]) >= 1:
                raise ValueError("Inadmissible parameters!")
            else:
                [self.phi, self.price_ret] = theta_ret

        if theta is not None:
            assert len(theta) == 5, \
                "Wrong number of parameters in theta!"
            if abs(theta[3]) >= 1 or np.array(theta[:3]).min() <= 0 \
                or theta[1] >= 1:
                    raise ValueError("Inadmissible parameters!")
            else:
                [self.mean, self.rho, self.delta, self.phi, self.price_ret] \
                    = theta
        # Update scale parameter
        self.scale = self.get_scale()

    def get_theta_vol(self):
        """Get volatility parameters in a vector.

        Returns
        -------
        (3,) array
            Volatility parameters

        """
        return np.array([self.mean, self.rho, self.delta])

    def get_theta_ret(self):
        """Get return parameters in a vector.

        Returns
        -------
        (2,) array
            Volatility parameters

        """
        return np.array([self.phi, self.price_ret])

    def get_theta(self):
        """Get model parameters in a vector.

        Returns
        -------
        (5,) array
            Volatility parameters

        """
        return np.hstack((self.get_theta_vol(), self.get_theta_ret()))

    def get_mean(self):
        """Get unconditional variance.

        Returns
        -------
        float
            Unconditional variance

        """
        return self.scale * self.delta / (1 - self.rho)

    def get_scale(self):
        """Get scale parameter.

        Returns
        -------
        float
            Scale parameter

        """
        return self.mean * (1 - self.rho) / self.delta

    def get_beta(self):
        """Get beta parameter.

        Returns
        -------
        float
            Beta parameter

        """
        return self.rho / self.scale

    def get_risk_prices(self):
        """Get risk prices, theta_1 (vol) and theta_2 (ret).

        Returns
        -------
        array
            Risk prices

        """
        return np.array([self.price_vol, self.price_ret])

    def __str__(self):
        """This is what is shown when you print() the instance.

        """
        params_vol = (self.mean, self.rho, self.delta)
        params_ret = (self.phi, self.price_ret)
        string = "mean = %.4f, rho = %.4f, delta = %.4f" % params_vol
        string += "\nphi = %.4f, price_ret = %.4f" % params_ret
        return string
