#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARG model class

References
----------
.. [1] Stanislav Khrapov and Eric Renault (2014)
    "Affine Option Pricing Model in Discrete Time",
    working paper, New Economic School.
    <https://sites.google.com/site/khrapovs/research
    /Renault-Khrapov-2012-Affine-Option-Pricing.pdf>

.. [2] Christian Gourieroux and Joann Jasiak (2006)
    "Autoregressive Gamma Processes",
    2006, Journal of Forecasting, 25(2), 129–152. doi:10.1002/for.978

.. [3] Serge Darolles, Christian Gourieroux, and Joann Jasiak (2006)
    "Structural Laplace Transform and Compound Autoregressive Models"
    Journal of Time Series Analysis, 27(4), 477–503.
    doi:10.1111/j.1467-9892.2006.00479.x

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from ARG.argparams import ARGparams

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class ARG(object):
    """Class for ARG model.

    .. math::
        E\left[\left.\exp\left\{ -uY_{t}\right\} \right|Y_{t-1}\right]\\
            =\exp\left\{ -a(u)Y_{t-1}-b(u)\right\}

    Attributes
    ----------
    param : ARGparams instance
        Parameters of the model
    vol : (nobs, ) array
        Time series data

    Methods
    -------
    afun(uarg)
        a(u) function
    bfun(uarg)
        b(u) function
    cfun(uarg)
        c(u) function
    plot_abc(uarg)
        Vizualize functions a, b, and c

    """
    def __init__(self, param=ARGparams()):
        """Initialize class instance.

        """
        self.param = param
        self.vol = None

    def afun(self, uarg):
        """Function a().

        .. math::
            a\left(u\right)=\frac{\rho u}{1+cu}

        Parameters
        ----------
        uarg : array

        Returns
        -------
        a(u) : array

        """
        return self.param.rho * uarg / (1 + self.param.scale * uarg)

    def bfun(self, uarg):
        """Function b().

        .. math::
            b\left(u\right)=\delta\log\left(1+cu\right)

        Parameters
        ----------
        uarg : array

        Returns
        -------
        b(u) : array

        """
        return self.param.delta * np.log(1 + self.param.scale * uarg)

    def cfun(self, uarg):
        """Function c().

        .. math::
            c\left(u\right)=\delta\log\left\{ 1+\frac{cu}{1-\rho}\right\}

        Parameters
        ----------
        uarg : array

        Returns
        -------
        c(u) : array

        """
        return self.param.delta \
            * np.log(1 + self.param.scale * uarg / (1-self.param.rho))

    def umean(self):
        """Unconditional mean of the process.

        .. math::
            E\left[Y_{t}\right]=\frac{c\delta}{1-\rho}

        Returns
        -------
        umean : float

        """
        return self.param.scale * self.param.delta / (1 - self.param.rho)

    def uvar(self):
        """Unconditional variance of the process.

        .. math::
            V\left[Y_{t}\right]=\frac{c^{2}\delta}{\left(1-\rho\right)^{2}}

        Returns
        -------
        uvar : float

        """
        return self.umean() / self.param.delta

    def ustd(self):
        """Unconditional variance of the process.

        .. math::
            \sqrt{V\left[Y_{t}\right]}

        Returns
        -------
        ustd : float

        """
        return self.uvar() ** .5

    def plot_abc(self, uarg):
        """Plot a() and b() functions on the same plot.

        """
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 3, 1)
        plt.plot(uarg, self.afun(uarg))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$a(u)$')
        plt.xlabel('$u$')

        plt.subplot(1, 3, 2)
        plt.plot(uarg, self.bfun(uarg))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$b(u)$')
        plt.xlabel('$u$')

        plt.subplot(1, 3, 3)
        plt.plot(uarg, self.cfun(uarg))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$c(u)$')
        plt.xlabel('$u$')

        plt.tight_layout()
        plt.show()

    def vsim(self, nsim=1, nobs=int(1e2), param=ARGparams()):
        """Simulate ARG(1) process.

        .. math::
            Z_{t}|Y_{t-1}\sim\mathcal{P}\left(\beta Y_{t-1}\right)
            Y_{t}|Z_{t}\sim\gamma\left(\delta+Z_{t},c\right)

        Parameters
        ----------
        nsim : int
            Number of series to simulate
        nobs : int
            Number of observations to simulate
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nsim, nobs) array
            Simulated data

        """
        vol = np.empty((nsim, nobs))
        vol[:, 0] = self.umean()
        for i in range(1, nobs):
            temp = np.random.poisson(self.param.beta * vol[:, i-1])
            vol[:, i] = self.param.scale \
                * np.random.gamma(self.param.delta + temp)
        return vol

    def vsim2(self, nsim=1, nobs=int(1e2), param=ARGparams()):
        """Simulate ARG(1) process.

        Parameters
        ----------
        nsim : int
            Number of series to simulate
        nobs : int
            Number of observations to simulate
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nsim, nobs) array
            Simulated data

        """
        vol = np.empty((nsim, nobs))
        vol[:, 0] = self.umean()
        for i in range(1, nobs):
            df = self.param.delta * 2
            nc = self.param.rho * vol[:, i-1]
            vol[:, i] = st.ncx2.rvs(df, nc, size=nsim)
        return vol * self.param.scale / 2

    def vsim_last(self, **args):
        """The last observation in the series of simulations.

        Parameters
        ----------
        args : dict
            Same parameters as in vsim function

        Returns
        -------
        (nsim, ) array
            Last observations

        TODO : This function could be less time consuming
            if intermediate values were not created.

        """
        return self.vsim(**args)[:, -1]

    def plot_vsim(self):
        """Plot simulated ARG process."""

        np.random.seed(seed=1)
        vol = self.vsim2(nsim=2)
        plt.figure(figsize=(8, 4))
        for voli in vol:
            plt.plot(voli)
        plt.show()

    def plot_vlast_density(self, nsim=100, nobs=100):
        """Plot the marginal density of ARG process."""

        plt.figure(figsize=(8, 4))
        vol = self.vsim_last(nsim=int(nsim), nobs=int(nobs))
        sns.distplot(vol, rug=True, hist=False)
        plt.show()

    def load_data(self, vol=None):
        """Load data to the class.

        Parameters
        ----------
        vol : (nobs, ) array
            Time series

        """
        if not vol is None:
            self.vol = vol
        else:
            raise(ValueError, "No data is given!")

if __name__ == '__main__':
    from usage_example import play_with_arg
    play_with_arg()
