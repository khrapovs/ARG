#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ARG model
=========

The code is an implementation of ARG model given in [1]_.
Its major features include:

    * simulation of stochastic volatility and returns
    * estimation using both MLE and GMM
    * option pricing

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

import warnings

import numpy as np
import numexpr as ne
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as scs
from scipy.optimize import minimize
import numdifftools as nd

from statsmodels.tsa.tsatools import lagmat

from .argparams import ARGparams
from .helper_functions import days_from_maturity
from argamma.mygmm import GMM
from argamma.fangoosterlee import cosmethod

__all__ = ['ARG']


class ARG(object):

    r"""Class for ARG model.

    .. math::

        E\left[\left.\exp\left\{ -uY_{t}\right\} \right|Y_{t-1}\right]
            =\exp\left\{ -a(u)Y_{t-1}-b(u)\right\}

    Attributes
    ----------
    vol
        Volatility series
    ret
        Asset return series
    param
        Parameters of the model
    maturity
        Maturity of the option or simply time horizon.
        Fraction of a year, i.e. 30/365
    riskfree
        Risk-free rate of return per day

    Methods
    -------
    afun
        a(u) function
    bfun
        b(u) function
    cfun
        c(u) function
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
    estimate_mle
        Estimate model parameters via Maximum Likelihood
    estimate_gmm
        Estimate model parameters using GMM
    cos_restriction
        Restrictions used in COS method of option pricing
    charfun
        Risk-neutral conditional characteristic function (one argument only)
    option_premium
        Model implied option premium via COS method

    """

    def __init__(self, param=None):
        """Initialize class instance.

        Parameters
        ----------
        param : ARGparams instance, optional
            Parameters of the model

        """
        self.vol = None
        self.ret = None
        self.param = param
        self.maturity = None
        self.riskfree = None

    def convert_to_q(self, param):
        """Convert physical (P) parameters to risk-neutral (Q) parameters.

        Parameters
        ----------
        param : ARGparams instance
            Physical (P) parameters

        Returns
        -------
        ARGparams instance
            Risk-neutral parameters

        """
        paramq = ARGparams()
        paramq.update(theta=param.get_theta(), price_vol=param.price_vol)
        factor = 1/(1 + param.get_scale() \
            * (param.price_vol + self.alpha(param.price_ret, param)))
        if factor <= 0 or factor > 1/param.rho**.5:
            print('Lower bound for theta1 is %.2f'
                % self.vol_price_lower_bound(param))
            raise ValueError('Invalid parameters in Q conversion!')
        delta = param.delta
        scale = param.get_scale() * factor
        beta = param.get_beta() * factor
        rho = scale * beta
        mean = scale * delta / (1 - rho)
        paramq.update(theta_vol=[mean, rho, delta])
        return paramq

    def load_data(self, vol=None, ret=None):
        """Load data into the model object.

        Parameters
        ----------
        vol : (nobs, ) array
            Volatility time series
        ret : (nobs, ) array
            Return time series

        """
        if vol is not None:
            self.vol = vol
        if ret is not None:
            self.ret = ret

    def afun(self, uarg, param):
        r"""Function a().

        .. math::

            a\left(u\right)=\frac{\rho u}{1+cu}

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
#        return param.rho * uarg / (1 + param.scale * uarg)
        rho = param.rho
        scale = param.get_scale()
        return ne.evaluate("rho * uarg / (1 + scale * uarg)")

    def bfun(self, uarg, param):
        r"""Function b().

        .. math::
            b\left(u\right)=\delta\log\left(1+cu\right)

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
#        return param.delta * np.log(1 + param.scale * uarg)
        scale = param.get_scale()
        delta = param.delta
        return ne.evaluate("delta * log(1 + scale * uarg)")

    def afun_q(self, uarg, param):
        r"""Risk-neutral function a().

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
        return self.afun(uarg, self.convert_to_q(param))

    def bfun_q(self, uarg, param):
        r"""Risk-neutral function b().

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
        return self.bfun(uarg, self.convert_to_q(param))

    def dafun(self, uarg, param):
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
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (3, nu) array

        """
        da_scale = -param.rho * uarg**2 / (param.get_scale()*uarg + 1)**2
        da_rho = uarg / (param.get_scale()*uarg + 1)
        da_delta = np.zeros_like(uarg)
        return np.vstack((da_scale, da_rho, da_delta))

    def dbfun(self, uarg, param):
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
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (3, nu) array

        """
        db_scale = param.delta * uarg / (1 + param.get_scale() * uarg)
        db_rho = np.zeros_like(uarg)
        db_delta = np.log(1 + param.get_scale() * uarg)
        return np.vstack((db_scale, db_rho, db_delta))

    def cfun(self, uarg, param):
        r"""Function c().

        .. math::
            c\left(u\right)=\delta\log\left\{1+\frac{cu}{1-\rho}\right\}

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
        return param.delta * np.log(1 + param.get_scale()*uarg / (1-param.rho))

    def center(self, param):
        """No-arb restriction parameter.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        float
            Same dimension as uarg

        """
        return param.phi / (param.get_scale() * (1 + param.rho))**.5

    def psi(self, param):
        """Function psi.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        float

        """
        return (param.price_ret-.5) * (1-param.phi**2) + self.center(param)

    def alpha(self, uarg, param):
        """Function alpha().

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
        return self.psi(param) * uarg - .5 * uarg**2 * (1 - param.phi**2)

    def beta(self, uarg, param):
        """Function beta(). Same for risk-neutral.

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
        return uarg * self.afun_q(- self.center(param), param)

    def gamma(self, uarg, param):
        """Function gamma(). Same for risk-neutral.

        Parameters
        ----------
        uarg : array
            Grid
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array
            Same dimension as uarg

        """
        return uarg * self.bfun_q(- self.center(param), param)

    def lgfun(self, uarg, varg, param):
        """Function l(u, v) and g(u, v) in joint characteristic function.

        Parameters
        ----------
        uarg : array
            Grid for volatility
        varg : array
            Grid for returns
        param : ARGparams instance
            Model parameters

        Returns
        -------
        lfun : array
        gfun : array

        """
        alpha = self.alpha(varg, param)
        lfun = self.afun(uarg + alpha, param) + self.beta(varg, param)
        gfun = self.bfun(uarg + alpha, param) + self.gamma(varg, param)
        return lfun, gfun

    def lgfun_q(self, uarg, varg, param):
        """Function l(u, v) and g(u, v) in joint risk-neutral
        characteristic function.

        Parameters
        ----------
        uarg : array
            Grid for volatility
        varg : array
            Grid for returns
        param : ARGparams instance
            Model parameters

        Returns
        -------
        lfun : array
        gfun : array

        """
        lfun1, gfun1 = self.lgfun(uarg + param.price_vol,
                                  varg + param.price_ret, param)
        lfun2, gfun2 = self.lgfun(param.price_vol, param.price_ret, param)
        lfun = lfun1 - lfun2
        gfun = gfun1 - gfun2
        return lfun, gfun

    def ch_fun_elements(self, varg, param):
        """Functions psi(v, n) and ups(v, n) in risk-neutral
        characteristic function of returns for n periods.

        Parameters
        ----------
        varg : array
            Grid for returns
        param : ARGparams instance
            Model parameters

        Returns
        -------
        psi : array
        ups : array

        """
        periods = days_from_maturity(self.maturity)
        ones = np.ones_like(periods)
        lfun, gfun = self.lgfun_q(0., varg, param)
        varg, psi, ups = varg * ones, lfun * ones, gfun * ones
        while True:
            if np.array_equal(periods, ones):
                return psi, ups
            cond = periods > 1
            periods[cond] -= 1
            lfun, gfun = self.lgfun_q(psi[:, cond], varg[:, cond], param)
            ups[:, cond] += gfun
            psi[:, cond] = lfun

    def store_ch_fun_elements(self, varg, param):
        """Functions psi(v, n) and ups(v, n) in risk-neutral
        characteristic function of returns for n periods.

        Parameters
        ----------
        varg : array
            Grid for returns
        param : ARGparams instance
            Model parameters

        Returns
        -------
        psi : array
        ups : array

        """
        periods = np.max(days_from_maturity(self.maturity))
        psi, ups = self.lgfun_q(0., varg, param)
        psidict, upsdict = {1: psi}, {1: ups}
        while True:
            if periods == 1:
                return psidict, upsdict
            periods -= 1
            lfun, gfun = self.lgfun_q(psi, varg, param)
            ups += gfun
            psi = lfun
            psidict[periods], upsdict[periods] = psi, ups

    def char_fun_ret_q(self, varg, param):
        r"""Conditional risk-neutral Characteristic function (return).

        Parameters
        ----------
        varg : array_like
            Grid for returns. Real values only.
        param : ARGparams instance
            Model parameters

        Returns
        -------
        array_like
            Characteristic function for each observation and each grid point

        Notes
        -----
        Conditional on :math:`\sigma_t` only
        All market data (vol, maturity, riskfree) can be vectors
        of the same size, and varg can be a vector of another size,
        but of transposed shape,
        i.e. vol = np.ones(5), and varg = np.ones((10, 1))

        """
        if self.vol is None:
            raise ValueError('Volatility is not loaded!')
        if self.maturity is None:
            raise ValueError('Maturity is not loaded!')
        if self.riskfree is None:
            raise ValueError('Risk-free rate is not loaded!')
        if np.iscomplex(varg).any():
            raise ValueError('Argument must be real!')

        psi, ups = self.ch_fun_elements(-1j * varg, param)
#        discount = np.exp(- 1j * varg * self.riskfree * self.maturity)
        return np.exp(- self.vol * psi - ups)

    def char_fun_vol(self, uarg, param):
        """Conditional Characteristic function (volatility).

        Parameters
        ----------
        uarg : array
            Grid. If real, then returns Laplace transform.
            If complex, then returns characteristi function.
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs-1, nu) array
            Characteristic function for each observation and each grid point

        """
        return np.exp(- self.vol[:-1, np.newaxis] * self.afun(uarg, param)
            - np.ones((self.vol[1:].shape[0], 1)) * self.bfun(uarg, param))

    def char_fun_ret(self, uarg, param):
        r"""Conditional Characteristic function (return).

        Parameters
        ----------
        uarg : array
            Grid. If real, then returns Laplace transform.
            If complex, then returns characteristi function.
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs-1, nu) array
            Characteristic function for each observation and each grid point

        Notes
        -----
        Conditional on current :math:`\sigma_{t+1}` and past :math:`\sigma_t`

        """
        return np.exp(-self.vol[1:, np.newaxis] * self.alpha(uarg, param)
            - self.vol[:-1, np.newaxis] * self.beta(uarg, param)
            - np.ones((self.vol[1:].shape[0], 1)) * self.gamma(uarg, param))

    def umean(self, param):
        r"""Unconditional mean of the volatility process.

        .. math::
            E\left[Y_{t}\right]=\frac{c\delta}{1-\rho}

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        float

        """
        return param.get_scale() * param.delta / (1 - param.rho)

    def uvar(self, param):
        r"""Unconditional variance of the volatility process.

        .. math::
            V\left[Y_{t}\right]=\frac{c^{2}\delta}{\left(1-\rho\right)^{2}}

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        float

        """
        return self.umean(param) / param.delta

    def ustd(self, param):
        r"""Unconditional standard deviation of the volatility process.

        .. math::
            \sqrt{V\left[Y_{t}\right]}

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        float

        """
        return self.uvar(param) ** .5

    def plot_abc(self, uarg, param):
        """Plot a() and b() functions on the same plot.

        """
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 3, 1)
        plt.plot(uarg, self.afun(uarg, param))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$a(u)$')
        plt.xlabel('$u$')

        plt.subplot(1, 3, 2)
        plt.plot(uarg, self.bfun(uarg, param))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$b(u)$')
        plt.xlabel('$u$')

        plt.subplot(1, 3, 3)
        plt.plot(uarg, self.cfun(uarg, param))
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
        (nobs, nsim) array
            Simulated data

        """
        vol = np.empty((nobs, nsim))
        vol[0] = param.get_mean()
        for i in range(1, nobs):
            temp = np.random.poisson(param.get_beta() * vol[i-1])
            vol[i] = param.get_scale() * np.random.gamma(param.delta + temp)
        return vol

    def vsim2(self, nsim=1, nobs=int(1e2), param=None):
        """Simulate ARG(1) process for volatility.

        Uses non-central Chi-square distribution to simulate in one step.

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
        (nobs, nsim) array
            Simulated data

        """
        vol = np.empty((nobs, nsim))
        vol[0] = param.get_mean()
        for i in range(1, nobs):
            df = param.delta * 2
            nc = param.rho * vol[i-1]
            vol[i] = scs.ncx2.rvs(df, nc, size=nsim)
        return vol * param.get_scale() / 2

    def vol_cmean(self, param):
        """Conditional mean of volatility.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional mean

        """
        return param.rho * self.vol[:-1] + param.delta * param.get_scale()

    def vol_cvar(self, param):
        """Conditional variance of volatility.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional variance

        """
        return (2 * param.rho * self.vol[:-1]
            + param.delta * param.get_scale()) * param.get_scale()

    def vol_kfun(self, param):
        """Conditional variance of volatility.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional mean

        """
        return self.umean(param) / ((2 * param.rho * self.umean(param)
            + param.delta * param.get_scale()) * param.get_scale())

    def ret_cmean(self, param):
        """Conditional mean of return.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional mean

        """
        A1 = self.psi(param)
        B1 = self.afun_q(- self.center(param), param)
        C1 = self.bfun_q(- self.center(param), param)
        return A1 * self.vol[1:] + B1 * self.vol[:-1] + C1

    def ret_cvar(self, param):
        """Conditional variance of return.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional mean

        """
#        u = sp.Symbol('u')
#        A2 = float(-self.alpha(u, param).diff(u, 2).subs(u, 0))
#        B2 = float(-self.beta(u, param).diff(u, 2).subs(u, 0))
#        C2 = float(-self.gamma(u, param).diff(u, 2).subs(u, 0))
#        return A2 * self.vol[1:] + B2 * self.vol[:-1] + C2
        return (1 - param.phi**2) * self.vol[1:]

    def erp(self, param):
        """Conditional equity risk premium.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            ERP

        """
        vrp = self.vol_cmean(param) - self.vol_cmean(self.convert_to_q(param))
        return (param.price_ret * (1-param.phi**2) * self.vol_cmean(param) +
            (self.center(param) - (1-param.phi**2)/2) * vrp)

    def vrp(self, param):
        """Conditional volatility risk premium.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            VRP

        """
        volp = self.vol_cmean(param)**.5
        volq = self.vol_cmean(self.convert_to_q(param))**.5
        return volp - volq

    def overdispersion(self, param):
        """Conditional overdispersion.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional overdispersion

        """
        return self.vol_cmean(param) / self.vol_cvar(param)

    def corr_series(self, param):
        """Conditional correlation time series.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Conditional correlation

        """
        return self.psi(param) * (self.psi(param)**2
            + (1-param.phi**2) * self.overdispersion(param)) ** (-.5)

    def approx_ratio(self, param):
        """Approximation ratio.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Approximation ratio

        """
        return 1 + param.phi**2 * (self.vol_kfun(param)
            / self.overdispersion(param) - 1)

    def rsim(self, param=None):
        """Simulate returns given ARG(1) process for volatility.

        Parameters
        ----------
        param : ARGparams instance
            Model parameters

        Returns
        -------
        (nobs, nsim) array
            Simulated data

        """
        # simulate returns
        ret = np.zeros_like(self.vol)
        ret[1:] = self.ret_cmean(param) + self.ret_cvar(param)**.5 \
            * np.random.normal(size=self.vol[1:].shape)
        return ret

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
        return self.vsim(**args)[-1]

    def plot_vsim(self, param=None):
        """Plot simulated ARG process."""

        np.random.seed(seed=1)
        vol = self.vsim2(nsim=2, param=param).T
        plt.figure(figsize=(8, 4))
        for voli in vol:
            plt.plot(voli)
        plt.show()

    def plot_vlast_density(self, nsim=100, nobs=100, param=None):
        """Plot the marginal density of ARG process."""

        plt.figure(figsize=(8, 4))
        vol = self.vsim_last(nsim=int(nsim), nobs=int(nobs), param=param)
        sns.distplot(vol, rug=True, hist=False)
        plt.show()

    def estimate_mle(self, param_start=None, model=None, bounds=None):
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
        bounds : list of tuples
            Bounds on parameters, i.e. [(min, max)]

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
            likelihood = lambda x: \
                self.likelihood_ret(x, param_start.get_theta_vol(),
                                    param_start.price_vol)
            theta_start = param_start.get_theta_ret()
        elif model == 'joint':
            likelihood = lambda x: \
                self.likelihood_joint(x, param_start.price_vol)
            theta_start = param_start.get_theta()
        else:
            raise ValueError('Model type not supported')

        results = minimize(likelihood, theta_start, method='L-BFGS-B',
                           options=options, bounds=bounds)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                hess_mat = nd.Hessian(likelihood)(results.x)
            except ValueError:
                hess_mat = nd.Jacobian(nd.Gradient(likelihood))(results.x)

        results.std_theta = np.diag(np.linalg.inv(hess_mat) \
            / len(self.vol))**.5
        results.tstat = results.x / results.std_theta

        param_final = param_start
        if model == 'vol':
            param_final.update(theta_vol=results.x)
        elif model == 'ret':
            param_final.update(theta_ret=results.x)
        elif model == 'joint':
            param_final.update(theta=results.x)
        else:
            raise ValueError('Model type not supported')

        return param_final, results

    def likelihood_vol(self, theta_vol):
        """Log-likelihood for ARG(1) volatility model.

        Parameters
        ----------
        theta : array_like
            Model parameters. [mean, rho, delta]

        Returns
        -------
        float
            Value of the log-likelihood function

        """
        param = ARGparams()
        try:
            param.update(theta_vol=theta_vol)
        except ValueError:
            return 1e10
        degf = param.delta * 2
        nonc = param.rho * self.vol[:-1] / param.get_scale() * 2
        scale = param.get_scale() / 2
        logf = scs.ncx2.logpdf(self.vol[1:], degf, nonc, scale=scale)
        return -logf[~np.isnan(logf)].mean()

    def likelihood_ret(self, theta_ret, theta_vol, price_vol):
        """Log-likelihood for return model.

        Parameters
        ----------
        theta_ret : array_like
            Model parameters. [phi, price_ret]
        theta_vol : array_like
            Volatility model parameters. [phi, price_ret]
        price_vol : float
            Price of volatility risk

        Returns
        -------
        float
            Value of the log-likelihood function

        """
        param = ARGparams()
        try:
            param.update(theta_ret=theta_ret, theta_vol=theta_vol,
                         price_vol=price_vol)
            if price_vol < self.vol_price_lower_bound(param):
                raise ValueError
            r_mean = self.ret_cmean(param)
            r_var = self.ret_cvar(param)
        except ValueError:
            return 1e10

        return - scs.norm.logpdf(self.ret[1:], r_mean, np.sqrt(r_var)).mean()

    def likelihood_joint(self, theta, price_vol):
        """Log-likelihood for joint model.

        Parameters
        ----------
        theta : array_like
            Model parameters
        price_vol : float
            Price of volatility risk

        Returns
        -------
        float
            Value of the log-likelihood function

        """
        theta_vol, theta_ret = theta[:3], theta[3:]
        return self.likelihood_vol(theta_vol) \
            + self.likelihood_ret(theta_ret, theta_vol, price_vol)

    def vol_price_lower_bound(self, param):
        """Get lower bound for volatility risk price.

        Parameters
        ----------
        param : ARGparams instance

        Returns
        -------
        float

        """
        return (param.rho**.5 - 1) / param.get_scale() \
            -self.alpha(param.price_ret, param)

    def momcond_vol(self, theta_vol, uarg=None, zlag=1):
        """Moment conditions (volatility) for spectral GMM estimator.

        Parameters
        ----------
        theta_vol : (3, ) array
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
        nmoms = 2 * uarg.shape[0] * (zlag+1)
        # Number of parameters
        nparams = theta_vol.shape[0]

        # Change class attribute with the current theta
        param = ARGparams()
        try:
            param.update(theta_vol=theta_vol)
        except ValueError:
            return np.ones((nobs, nmoms))*1e10, np.ones((nmoms, nparams))*1e10

        # Must be (nobs, nu) array
        error = np.exp(-vol * uarg) - self.char_fun_vol(uarg, param)[zlag-1:]
        # Instruments, (nobs, ninstr) array
        instr = np.hstack([np.exp(-1j * vollag), np.ones((nobs, 1))])
        # Must be (nobs, nmoms) array
        moment = error[:, np.newaxis, :] * instr[:, :, np.newaxis]
        moment = moment.reshape((nobs, nmoms//2))
        # (nobs, 2 * ninstr)
        moment = np.hstack([np.real(moment), np.imag(moment)])

        # Initialize derivative matrix
        dmoment = np.empty((nmoms, nparams))
        for i in range(nparams):
            dexparg = - prevvol * self.dafun(uarg, param)[i] \
                - np.ones((nobs, 1)) * self.dbfun(uarg, param)[i]
            derror = - self.char_fun_vol(uarg, param)[zlag-1:] * dexparg

            derrorinstr = derror[:, np.newaxis, :] * instr[:, :, np.newaxis]
            derrorinstr = derrorinstr.reshape((nobs, nmoms//2))
            derrorinstr = np.hstack([np.real(derrorinstr),
                                     np.imag(derrorinstr)])
            dmoment[:, i] = derrorinstr.mean(0)

        return moment, dmoment

    def moment_ret(self, theta_ret, theta_vol=None, uarg=None,
                   zlag=1, **kwargs):
        """Moment conditions (returns) for spectral GMM estimator.

        Parameters
        ----------
        theta_ret : (2, ) array
            Vector of model parameters. [phi, price_ret]
        theta_vol : (3, ) array
            Vector of model parameters. [mean, rho, delta]
        uarg : (nu, ) array
            Grid to evaluate a and b functions
        zlag : int
            Number of lags to use for the instrument

        Returns
        -------
        moment : (nobs, nmoms) array
            Matrix of momcond restrictions

        Raises
        ------
        ValueError

        """

        if uarg is None:
            raise ValueError("uarg is missing!")

        vollag, vol = lagmat(self.vol, maxlag=zlag,
                             original='sep', trim='both')
        # Number of observations after truncation
        nobs = vol.shape[0]
        # Number of moments
        nmoms = 2 * uarg.shape[0] * (zlag+1)
        # Change class attribute with the current theta
        param = ARGparams()
        try:
            param.update(theta_ret=theta_ret, theta_vol=theta_vol)
        except ValueError:
            return np.ones((nobs, nmoms))*1e10
        # Must be (nobs, nu) array
        try:
            cfun = self.char_fun_ret(uarg, param)[zlag-1:]
        except ValueError:
            return np.ones((nobs, nmoms))*1e10
        # Must be (nobs, nu) array
        error = np.exp(-self.ret[zlag:, np.newaxis] * uarg) - cfun
        # Instruments, (nobs, ninstr) array
        instr = np.hstack([np.exp(-1j * vollag), np.ones((nobs, 1))])
        # Must be (nobs, nmoms) array
        moment = error[:, np.newaxis, :] * instr[:, :, np.newaxis]
        moment = moment.reshape((nobs, nmoms//2))
        # (nobs, 2 * ninstr)
        moment = np.hstack([np.real(moment), np.imag(moment)])

        return moment

    def momcond_ret(self, theta_ret, **kwargs):
        """Moment conditions (returns) for spectral GMM estimator.

        Parameters
        ----------
        theta_ret : (2, ) array
            Vector of model parameters. [phi, price_ret]

        Returns
        -------
        moment : (nobs, nmoms) array
            Matrix of momcond restrictions
        dmoment : (nmoms, nparams) array
            Gradient of momcond restrictions. Mean over observations

        """
        mom = self.moment_ret(theta_ret, **kwargs)
        dmom = self.dmoment_ret(theta_ret, **kwargs)
        return mom, dmom

    def dmoment_ret(self, theta_ret, **kwargs):
        """Derivative of moments (returns) for spectral GMM estimator.

        Parameters
        ----------
        theta_ret : (2, ) array
            Vector of model parameters. [phi, price_ret]

        Returns
        -------
        (nmoms, nparams) array
            Gradient of moment restrictions. Mean over observations

        """
        mom = lambda theta: self.moment_ret(theta, **kwargs).mean(0)
        with np.errstate(divide='ignore'):
            return nd.Jacobian(mom)(theta_ret)

    def moment_joint(self, theta, **kwargs):
        """Moment conditions (joint) for spectral GMM estimator.

        Parameters
        ----------
        theta : (5, ) array
            Vector of model parameters. [phi, price_ret]

        Returns
        -------
        moment : (nobs, nmoms) array
            Matrix of momcond restrictions

        """
        theta_vol, theta_ret = theta[:3], theta[3:]
        mom_vol = self.momcond_vol(theta_vol, **kwargs)[0]
        mom_ret = self.moment_ret(theta_ret, **kwargs)
        return np.hstack([mom_vol, mom_ret])

    def dmoment_joint(self, theta, **kwargs):
        """Derivative of moment conditions (joint) for spectral GMM estimator.

        Parameters
        ----------
        theta : (5, ) array
            Vector of model parameters. [phi, price_ret]

        Returns
        -------
        (nmoms, nparams) array
            Gradient of moment restrictions. Mean over observations

        """
        mom = lambda theta: self.moment_joint(theta, **kwargs).mean(0)
        with np.errstate(divide='ignore'):
            return nd.Jacobian(mom)(theta)

    def momcond_joint(self, theta, **kwargs):
        """Moment conditions (joint) for spectral GMM estimator.

        Parameters
        ----------
        theta : (5, ) array
            Vector of model parameters. [phi, price_ret]

        Returns
        -------
        moment : (nobs, nmoms) array
            Matrix of momcond restrictions
        dmoment : (nmoms, nparams) array
            Gradient of momcond restrictions. Mean over observations

        """
        mom = self.moment_joint(theta, **kwargs)
        dmom = self.dmoment_joint(theta, **kwargs)
        return mom, dmom

    def estimate_gmm(self, param_start=None, model='vol', **kwargs):
        """Estimate model parameters using GMM.

        Parameters
        ----------
        param_start : ARGparams instance
            Starting value for optimization
        model : str
            Type of the model to estimate. Must be in:
                - 'vol'
                - 'ret'
                - 'joint'
        uarg : array
            Grid to evaluate a and b functions
        zlag : int, optional
            Number of lags in the instrument. Default is 1

        Returns
        -------
        param_final : ARGparams instance
            Estimated model parameters
        mygmm.Results instance
            GMM estimation results

        """
        if model == 'vol':
            estimator = GMM(self.momcond_vol)
            results = estimator.gmmest(param_start.get_theta_vol(), **kwargs)
        elif model == 'ret':
            estimator = GMM(self.momcond_ret)
            results = estimator.gmmest(param_start.get_theta_ret(),
                                       theta_vol=param_start.get_theta_vol(),
                                       **kwargs)
        elif model == 'joint':
            estimator = GMM(self.momcond_joint)
            results = estimator.gmmest(param_start.get_theta(), **kwargs)
        else:
            raise ValueError('Model type not supported')

        param_final = param_start
        if model == 'vol':
            param_final.update(theta_vol=results.theta)
        elif model == 'ret':
            param_final.update(theta_vol=param_start.get_theta_vol(),
                               theta_ret=results.theta)
        elif model == 'joint':
            param_final.update(theta=results.theta)
        else:
            raise ValueError('Model type not supported')

        return param_final, results

    def cos_restriction(self):
        """Restrictions used in COS method of option pricing.

        Parameters
        ----------
        riskfree : array_like
            Risk-free rate of returns, per day
        maturity : array_like
            Maturity, fraction of the year, i.e. 30/365

        Returns
        -------
        alim : array_like
        blim : array_like

        Notes
        -----
        This method is used by COS method of option pricing

        """
        L = 100.
        periods = days_from_maturity(self.maturity)
        c1 = self.riskfree * periods
        c2 = self.vol * periods

        alim = c1 - L * c2**.5
        blim = c1 + L * c2**.5

        return alim, blim

    def charfun(self, varg):
        """Risk-neutral conditional characteristic function.

        Parameters
        ----------
        varg : array
            Grid for evaluation of CF. Real values only.

        Returns
        -------
        array
            Same dimension as varg

        Notes
        -----
        This method is used by COS method of option pricing

        """
        if self.param is None:
            raise ValueError('Parameters are not set!')

        return self.char_fun_ret_q(varg, self.param)

    def option_premium(self, vol=None, moneyness=None, maturity=None,
                       riskfree=None, call=None, data=None):
        """Model implied option premium via COS method.

        Parameters
        ----------
        vol : array_like
            Current variance per day
        moneyness : array_like
            Log-forward moneyness, np.log(strike/price) - riskfree * maturity
        maturity : float, optional
            Maturity of the option or simply time horizon.
            Fraction of a year, i.e. 30/365
        riskfree : float, optional
            Risk-free rate of return per day
        call : bool array_like
            Call/Put flag
        data : pandas DataFrame, record array, or dictionary of arrays
            Structured data. Mandatory labels:
            vol, moneyness, maturity, riskfree, call

        Returns
        -------
        array_like
            Model implied option premium via COS method

        """
        if data is not None:
            try:
                data = data.to_records()
            except:
                pass
            maturity = data['maturity']
            riskfree = data['riskfree']
            vol = data['vol']
            moneyness = data['moneyness']
            call = data['call']

        self.maturity = maturity
        self.riskfree = riskfree
        self.vol = vol

        return cosmethod(self, moneyness=moneyness, call=call)


if __name__ == '__main__':

    pass
