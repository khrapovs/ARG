#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""ARG model class

References
----------

.. [1] Stanislav Khrapov and Eric Renault (2014)
    "Affine Option Pricing Model in Discrete Time",
    working paper, New Economic School.
    <https://sites.google.com/site/khrapovs/research/Renault-Khrapov-2012-Affine-Option-Pricing.pdf>

.. [2] Christian Gourieroux and Joann Jasiak (2006)
    "Autoregressive Gamma Processes",
    2006, *Journal of Forecasting*, 25(2), 129–152. doi:10.1002/for.978

.. [3] Serge Darolles, Christian Gourieroux, and Joann Jasiak (2006)
    "Structural Laplace Transform and Compound Autoregressive Models"
    *Journal of Time Series Analysis*, 27(4), 477–503.
    doi:10.1111/j.1467-9892.2006.00479.x

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as scs
from scipy.optimize import minimize
from statsmodels.tsa.tsatools import lagmat

from . import ARGparams, likelihood_vol
from MyGMM import GMM

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class ARG(GMM):

    r"""Class for ARG model.

    .. math::

        E\left[\left.\exp\left\{ -uY_{t}\right\} \right|Y_{t-1}\right]
            =\exp\left\{ -a(u)Y_{t-1}-b(u)\right\}

    Attributes
    ----------
    param : ARGparams instance
        Parameters of the model
    vol : 1d array
        Time series data

    Methods
    -------
    afun
        a(u) function
    bfun
        b(u) function
    cfun
        c(u) function
    dafun
        Derivative of a(u) function wrt scale, rho, and delta
    dbfun
        Derivative of b(u) function wrt scale, rho, and delta
    plot_abc
        Vizualize functions a, b, and c
    vsim
        Simulate ARG(1) process
    vsim2
        Simulate ARG(1) process
    load_data
        Load data to the class
    estimate_mle
        Estimate model parameters via Maximum Likelihood
    momcond
        Moment conditions for spectral GMM estimator

    """

    def __init__(self, param=ARGparams()):
        """Initialize class instance.

        """
        super(ARG, self).__init__()
        self.param = param
        self.vol = None

    def afun(self, uarg):
        r"""Function a().

        .. math::

            a\left(u\right)=\frac{\rho u}{1+cu}

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        return self.param.rho * uarg / (1 + self.param.scale * uarg)

    def dafun(self, uarg):
        r"""Derivative of function a() with respect to scale, rho, and delta.

        .. math::
            \frac{\partial a}{\partial c}\left(u\right)
                &=-\frac{\rho u^2}{\left(1+cu\right)^2} \\
            \frac{\partial a}{\partial \rho}a\left(u\right)
                &=\frac{u}{1+cu} \\
            \frac{\partial a}{\partial \delta}a\left(u\right)
                &=0

        Parameters
        ----------
        uarg : (nu, ) array

        Returns
        -------
        (3, nu) array

        """
        da_scale = -self.param.rho * uarg**2 / (self.param.scale*uarg + 1)**2
        da_rho = uarg / (self.param.scale*uarg + 1)
        da_delta = np.zeros_like(uarg)
        return np.vstack((da_scale, da_rho, da_delta))

    def bfun(self, uarg):
        r"""Function b().

        .. math::
            b\left(u\right)=\delta\log\left(1+cu\right)

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        return self.param.delta * np.log(1 + self.param.scale * uarg)

    def dbfun(self, uarg):
        r"""Derivative of function b() with respect to scale, rho, and delta.

        .. math::
            \frac{\partial b}{\partial c}\left(u\right)
                &=\frac{\delta u}{1+cu} \\
            \frac{\partial b}{\partial \rho}\left(u\right)
                &=0 \\
            \frac{\partial b}{\partial \delta}\left(u\right)
                &=\log\left(1+cu\right)

        Parameters
        ----------
        uarg : (nu, ) array

        Returns
        -------
        (3, nu) array

        """
        db_scale = self.param.delta * uarg / (1 + self.param.scale * uarg)
        db_rho = np.zeros_like(uarg)
        db_delta = np.log(1 + self.param.scale * uarg)
        return np.vstack((db_scale, db_rho, db_delta))

    def cfun(self, uarg):
        r"""Function c().

        .. math::
            c\left(u\right)=\delta\log\left\{1+\frac{cu}{1-\rho}\right\}

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        return self.param.delta \
            * np.log(1 + self.param.scale * uarg / (1-self.param.rho))

    def umean(self):
        r"""Unconditional mean of the process.

        .. math::
            E\left[Y_{t}\right]=\frac{c\delta}{1-\rho}

        Returns
        -------
        float

        """
        return self.param.scale * self.param.delta / (1 - self.param.rho)

    def uvar(self):
        r"""Unconditional variance of the process.

        .. math::
            V\left[Y_{t}\right]=\frac{c^{2}\delta}{\left(1-\rho\right)^{2}}

        Returns
        -------
        float

        """
        return self.umean() / self.param.delta

    def ustd(self):
        r"""Unconditional variance of the process.

        .. math::
            \sqrt{V\left[Y_{t}\right]}

        Returns
        -------
        float

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
        r"""Simulate ARG(1) process.

        .. math::

            Z_{t}|Y_{t-1}&\sim\mathcal{P}\left(\beta Y_{t-1}\right)\\
            Y_{t}|Z_{t}&\sim\gamma\left(\delta+Z_{t},c\right)

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
            vol[:, i] = scs.ncx2.rvs(df, nc, size=nsim)
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
            raise ValueError("No data is given!")

    def estimate_mle(self, param_start=ARGparams()):
        """Estimate model parameters via Maximum Likelihood.

        Returns
        -------
        param_final : ARGparams instance
            Estimated parameters

        """
        # Optimization options
        options = {'disp': False, 'maxiter': int(1e6)}
        results = minimize(likelihood_vol, param_start.theta,
                           args=(self.vol, ), method='L-BFGS-B',
                           options=options)
        param_final = ARGparams(theta=results.x)
        return param_final, results

    def momcond(self, theta, uarg=None, zlag=1):
        """Moment conditions for spectral GMM estimator.

        Parameters
        ----------
        theta : (3, ) array
            Vector of model parameters. [scale, rho, delta]
        uarg : (nu, ) array
            Grid to evaluate a and b functions
        zlag : int
            Number of lags to use for the instrument

        Returns
        -------
        moment : (nobs, nmoms) array
            Matrix of momcond restrictions
        dmoment : (nmoms, nparams) array
            Gradient of momcond restrictions. Mean over observations

        Raises
        ------
        ValueError

        """

        if uarg is None:
            raise ValueError("uarg is missing!")
        if self.vol is None:
            raise ValueError("vol data is missing!")

        vollag, vol = lagmat(self.vol, maxlag=zlag,
                             original='sep', trim='both')
        prevvol = vollag[:, 0][:, np.newaxis]
        # Number of observations after truncation
        nobs = vol.shape[0]
        # Number of moments
        nmoms = 2 * uarg.shape[0] * zlag
        # Number of instruments
        nparams = theta.shape[0]

        if theta.min() <= 0:
            moment = np.ones((nobs, nmoms)) * 1e10
            dmoment = np.ones((nmoms, nparams)) * 1e10
            return moment, dmoment
        # Change class attribute with the current theta
        self.param = ARGparams(theta=theta)

        # Must be (nobs, nu) array
        exparg = - prevvol * self.afun(uarg)
        exparg -= np.ones((nobs, 1)) * self.bfun(uarg)
        # Must be (nobs, nu) array
        error = np.exp(-vol * uarg) - np.exp(exparg)
        # Instruments, (nobs, ninstr) array
        instr = np.exp(-1j * vollag)
        # Must be (nobs, nmoms) array
        moment = error[:, np.newaxis, :] * instr[:, :, np.newaxis]
        moment = moment.reshape((nobs, nmoms/2))
        # (nobs, 2 * ninstr)
        moment = np.hstack([np.real(moment), np.imag(moment)])

        # Initialize derivative matrix
        dmoment = np.empty((nmoms, nparams))
        for i in range(nparams):
            dexparg = - prevvol * self.dafun(uarg)[i]
            dexparg -= np.ones((nobs, 1)) * self.dbfun(uarg)[i]
            derror = - np.exp(exparg) * dexparg

            derrorinstr = derror[:, np.newaxis, :] * instr[:, :, np.newaxis]
            derrorinstr = derrorinstr.reshape((nobs, nmoms/2))
            derrorinstr = np.hstack([np.real(derrorinstr),
                                     np.imag(derrorinstr)])
            dmoment[:, i] = derrorinstr.mean(0)

        return moment, dmoment


if __name__ == '__main__':
    from usage_example import play_with_arg, estimate_mle
    estimate_mle()