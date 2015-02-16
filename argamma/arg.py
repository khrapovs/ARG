#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ARG model
=========

References
----------

.. [1] Stanislav Khrapov and Eric Renault (2014)
    "Affine Option Pricing Model in Discrete Time",
    working paper, New Economic School.
    <http://goo.gl/yRVsZp>

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
import sympy as sp
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as scs
from scipy.optimize import minimize
import numdifftools as nd

from statsmodels.tsa.tsatools import lagmat

from .argparams import ARGparams
from mygmm import GMM

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"

__all__ = ['ARG']


class ARG(object):

    r"""Class for ARG model.

    .. math::

        E\left[\left.\exp\left\{ -uY_{t}\right\} \right|Y_{t-1}\right]
            =\exp\left\{ -a(u)Y_{t-1}-b(u)\right\}

    Attributes
    ----------
    param : ARGparams instance
        Parameters of the model
    vol : (nobs, ) array
        Volatility time series
    ret : (nobs, ) array
        Return time series

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
    umean
        Unconditional mean of the volatility process
    uvar
        Unconditional variance of the volatility process
    plot_abc
        Vizualize functions a, b, and c
    vsim
        Simulate ARG(1) volatility process
    vsim2
        Simulate ARG(1) volatility process
    rsim
        Simulate returns given volatility
    load_data
        Load data to the class
    likelihood_vol
        Log-likelihood for ARG(1) volatility model
    likelihood_ret
        Log-likelihood for return model
    estimate_mle
        Estimate model parameters via Maximum Likelihood
    momcond
        Moment conditions for spectral GMM estimator

    """

    def __init__(self, param=None):
        """Initialize class instance.

        Parameters
        ----------
        param : ARGparams instance
            Parameter object

        """
        #super(ARG, self).__init__()
        self.param = param
        self.vol = None
        self.ret = None

    def convert_to_q(self):
        """Convert physical (P) parameters to risk-neutral (Q) parameters.

        Returns
        -------
        ARGparams instance
            Risk-neutral parameters

        """
        factor = 1/(1 + self.param.scale \
            * (self.param.price_vol + self.alpha(self.param.price_ret)))
        if factor <= 0:
            raise ValueError('Invalid parameters in Q conversion!')
        scale = self.param.scale * factor
        beta = self.param.beta * factor
        rho = scale * beta
        return ARGparams(scale=scale, rho=rho, delta=self.param.delta)

    def load_data(self, vol=None, ret=None):
        """Load data into the model object.

        Parameters
        ----------
        vol : (nobs, ) array
            Volatility time series
        ret : (nobs, ) array
            Return time series

        """
        self.vol = vol
        self.ret = ret

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

    def afun_q(self, uarg):
        r"""Risk-neutral function a().

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        argmodel_q = ARG(param=self.convert_to_q())
        return argmodel_q.afun(uarg)

    def bfun_q(self, uarg):
        r"""Risk-neutral function b().

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        argmodel_q = ARG(param=self.convert_to_q())
        return argmodel_q.bfun(uarg)

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

    def center(self):
        """No-arb restriction parameter.

        Returns
        -------
        float
            Same dimension as uarg

        """
        return self.param.phi / (self.param.scale * (1 + self.param.rho))**.5

    def alpha(self, uarg):
        """Function alpha().

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        return ((self.param.price_ret-.5) * (1-self.param.phi**2) \
            + self.center()) * uarg - .5 * uarg**2 * (1 - self.param.phi**2)

    def beta(self, uarg):
        """Function beta().

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        return uarg * self.afun_q(- self.center())

    def gamma(self, uarg):
        """Function gamma().

        Parameters
        ----------
        uarg : array

        Returns
        -------
        array
            Same dimension as uarg

        """
        return uarg * self.bfun_q(- self.center())

    def umean(self):
        r"""Unconditional mean of the volatility process.

        .. math::
            E\left[Y_{t}\right]=\frac{c\delta}{1-\rho}

        Returns
        -------
        float

        """
        return self.param.scale * self.param.delta / (1 - self.param.rho)

    def uvar(self):
        r"""Unconditional variance of the volatility process.

        .. math::
            V\left[Y_{t}\right]=\frac{c^{2}\delta}{\left(1-\rho\right)^{2}}

        Returns
        -------
        float

        """
        return self.umean() / self.param.delta

    def ustd(self):
        r"""Unconditional standard deviation of the volatility process.

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

    def vsim(self, nsim=1, nobs=int(1e2), param=None):
        r"""Simulate ARG(1) process for volatility.

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

    def vsim2(self, nsim=1, nobs=int(1e2), param=None):
        """Simulate ARG(1) process for volatility.

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

    def ret_cmean(self):
        """Conditional mean of return.

        Returns
        -------
        (nsim, nobs) array
            Conditional mean

        """
        u = sp.Symbol('u')
        A1 = float(self.alpha(u).diff(u, 1).subs(u, 0))
        B1 = float(self.beta(u).diff(u, 1).subs(u, 0))
        C1 = float(self.gamma(u).diff(u, 1).subs(u, 0))
        vol = np.atleast_2d(self.vol)
        return A1 * vol[:, 1:] + B1 * vol[:, :-1] + C1

    def ret_cvar(self):
        """Conditional variance of return.

        Returns
        -------
        (nsim, nobs) array
            Conditional mean

        """
        u = sp.Symbol('u')
        A2 = float(-self.alpha(u).diff(u, 2).subs(u, 0))
        B2 = float(-self.beta(u).diff(u, 2).subs(u, 0))
        C2 = float(-self.gamma(u).diff(u, 2).subs(u, 0))
        vol = np.atleast_2d(self.vol)
        return A2 * vol[:, 1:] + B2 * vol[:, :-1] + C2

    def rsim(self):
        """Simulate returns given ARG(1) process for volatility.

        Returns
        -------
        (nsim, nobs) array
            Simulated data

        """
        # simulate returns
        ret = self.ret_cmean() + self.ret_cvar()**.5 \
            * np.random.normal(size=self.vol[:, 1:].shape)
        return np.hstack((np.zeros((self.vol.shape[0], 1)), ret))


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

    def estimate_mle(self, param_start=None, model=None):
        """Estimate model parameters via Maximum Likelihood.

        Parameters
        ----------
        param_start : ARGparams instance, optional
            Starting value for optimization
        model : str
            Type of model to estimate. Must be in:
                - 'vol'
                - 'ret'
                - 'joint'

        Returns
        -------
        param_final : ARGparams instance
            Estimated parameters
        results : OptimizeResult instance
            Optimization output

        """
        if param_start is None:
            param_start = ARGparams()
        # Optimization options
        options = {'disp': False, 'maxiter': int(1e6)}

        if model == 'vol':
            likelihood = self.likelihood_vol
            theta_start = param_start.get_theta_vol()
        elif model == 'ret':
            likelihood = self.likelihood_ret
            theta_start = param_start.get_theta_ret()
        elif model == 'joint':
            likelihood = self.likelihood_joint
            theta_start = param_start.get_theta()
        else:
            raise ValueError('Model type not supported')

        results = minimize(likelihood, theta_start, method='L-BFGS-B',
                           options=options)

        hess_mat = nd.Hessian(likelihood)(results.x)
        results.std_theta = np.diag(np.linalg.inv(hess_mat) \
            / len(self.vol))**.5
        results.tstat = results.x / results.std_theta

        if model == 'vol':
            self.param.update(theta_vol=results.x)
        elif model == 'ret':
            self.param.update(theta_ret=results.x)
        elif model == 'joint':
            self.param.update(theta=results.x)
        else:
            raise ValueError('Model type not supported')

        return self.param, results

    def likelihood_vol(self, theta_vol):
        """Log-likelihood for ARG(1) volatility model.

        Parameters
        ----------
        theta : array_like
            Model parameters. [scale, rho, delta]

        Returns
        -------
        float
            Value of the log-likelihood function

        """
        if theta_vol.min() <= 0 or theta_vol[0] <= 0:
            return 1e10
        self.param.update(theta_vol=theta_vol)
        degf = self.param.delta * 2
        nonc = self.param.rho * self.vol[:-1] / self.param.scale * 2
        scale = self.param.scale/2
        logf = scs.ncx2.logpdf(self.vol[1:], degf, nonc, scale=scale)
        return -logf[~np.isnan(logf)].mean()

    def likelihood_ret(self, theta_ret):
        """Log-likelihood for return model.

        Parameters
        ----------
        theta : array_like
            Model parameters. [phi, price_ret]

        Returns
        -------
        logf : float
            Value of the log-likelihood function

        """
        r_mean = self.ret_cmean().flatten()
        r_var = self.ret_cvar().flatten()

        return - scs.norm.logpdf(self.ret[1:], r_mean, np.sqrt(r_var)).mean()

    def likelihood_joint(self, theta):
        """Log-likelihood for joint model.

        Parameters
        ----------
        theta : array_like
            Model parameters.

        Returns
        -------
        logf : float
            Value of the log-likelihood function

        """
        try:
            return self.likelihood_vol(theta[:3]) \
                + self.likelihood_ret(theta[3:])
        except ValueError:
            return 1e10

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

        vollag, vol = lagmat(self.vol, maxlag=zlag,
                             original='sep', trim='both')
        prevvol = vollag[:, 0][:, np.newaxis]
        # Number of observations after truncation
        nobs = vol.shape[0]
        # Number of moments
        nmoms = 2 * uarg.shape[0] * zlag
        # Number of instruments
        nparams = theta.shape[0]

        if theta[0] <= 0:
            return np.ones((nobs, nmoms))*1e10, np.ones((nmoms, nparams))*1e10

        # Change class attribute with the current theta
        self.param.update(theta_vol=theta)

        if theta.min() <= 0:
            moment = np.ones((nobs, nmoms)) * 1e10
            dmoment = np.ones((nmoms, nparams)) * 1e10
            return moment, dmoment

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

    def estimate_gmm(self, param_start=None, **kwargs):
        """Estimate model parameters using GMM.

        Parameters
        ----------
        param_start : ARGparams instance
            Starting value for optimization
        uarg : array
            Grid to evaluate a and b functions
        zlag : int, optional
            Number of lags in the instrument. Default is 1

        Returns
        -------
        mygmm.Results instance
            GMM estimation results

        """
        estimator = GMM(self.momcond)
        results = estimator.gmmest(param_start, **kwargs)
        param_final = ARGparams()
        param_final.update(theta_vol=results.theta)
        return param_final, results


if __name__ == '__main__':
    pass
