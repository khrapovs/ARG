#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ARG class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import scipy.optimize as so

from argamma import (ARG, ARGparams,
                     likelihood_vol, likelihood_ret)
from mygmm import Results

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class ARGTestCase(ut.TestCase):
    """Test ARG, ARGparams model classes."""

    def test_param_class_vol(self):
        """Test parameter class for volatility parameters."""

        param = ARGparams()

        self.assertIsInstance(param.scale, float)
        self.assertIsInstance(param.rho, float)
        self.assertIsInstance(param.delta, float)
        self.assertIsInstance(param.beta, float)

        scale, rho, delta = .1, 0, 0
        theta_true = [scale, rho, delta]
        param = ARGparams(scale=scale, rho=rho, delta=delta)

        self.assertIsInstance(param.theta_vol, np.ndarray)
        np.testing.assert_array_equal(param.theta_vol, theta_true)

        scale, rho, delta = 1, 2, 3
        theta_true = [scale, rho, delta]
        param = ARGparams(theta_vol=theta_true)

        self.assertEqual(param.scale, scale)
        self.assertEqual(param.rho, rho)
        self.assertEqual(param.delta, delta)

        theta = [0, 0]
        self.assertRaises(AssertionError, lambda: ARGparams(theta_vol=theta))

    def test_param_class_ret(self):
        """Test parameter class for return parameters."""

        param = ARGparams()

        self.assertIsInstance(param.phi, float)
        self.assertIsInstance(param.price_ret, float)
        self.assertIsInstance(param.price_vol, float)

        phi, price_vol, price_ret = -.5, -5, 5
        theta_true = [phi, price_ret]
        param = ARGparams(phi=phi, price_ret=price_ret)

        self.assertIsInstance(param.theta_ret, np.ndarray)
        np.testing.assert_array_equal(param.theta_ret, theta_true)

        phi, price_vol, price_ret = 1, 2, 3
        theta_true = [phi, price_ret]
        param = ARGparams(theta_ret=theta_true)

        self.assertEqual(param.phi, phi)
        self.assertEqual(param.price_ret, price_ret)

    def test_uncond_moments(self):
        """Test unconditional moments of the ARG model."""

        argmodel = ARG(ARGparams())

        self.assertIsInstance(argmodel.umean(), float)
        self.assertIsInstance(argmodel.uvar(), float)
        self.assertIsInstance(argmodel.ustd(), float)

        # TODO : test using symbolic library
        # that these moments coincide with derivatives of a, b, c

    def test_abc_functions(self):
        """Test functions a, b, c of ARG model."""

        argmodel = ARG(ARGparams())
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, 10)

            self.assertIsInstance(argmodel.afun(uarg), np.ndarray)
            self.assertIsInstance(argmodel.bfun(uarg), np.ndarray)
            self.assertIsInstance(argmodel.cfun(uarg), np.ndarray)

            self.assertEqual(uarg.shape, argmodel.afun(uarg).shape)
            self.assertEqual(uarg.shape, argmodel.bfun(uarg).shape)
            self.assertEqual(uarg.shape, argmodel.cfun(uarg).shape)

    def test_abc_derivatives(self):
        """Test derivatives of functions a, b, c of ARG model."""

        argmodel = ARG(ARGparams())
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, 10)

            self.assertIsInstance(argmodel.dafun(uarg), np.ndarray)
            self.assertIsInstance(argmodel.dbfun(uarg), np.ndarray)

            self.assertEqual(argmodel.dafun(uarg).shape, (3, uarg.shape[0]))
            self.assertEqual(argmodel.dbfun(uarg).shape, (3, uarg.shape[0]))

    def test_simulations(self):
        """Test simulation of ARG model."""

        argmodel = ARG(ARGparams())

        self.assertIsInstance(argmodel.vsim(), np.ndarray)
        self.assertIsInstance(argmodel.vsim2(), np.ndarray)
        self.assertIsInstance(argmodel.rsim(argmodel.vsim2()), np.ndarray)

        shapes = []
        nsim, nobs = 1, 1
        shapes.append((nsim, nobs))
        nsim, nobs = 2, 2
        shapes.append((nsim, nobs))
        nsim, nobs = int(1e3), int(1e3)
        shapes.append((nsim, nobs))

        for shape in shapes:
            nsim, nobs = shape

            self.assertEqual(argmodel.vsim(nsim=nsim, nobs=nobs).shape, shape)
            vol = argmodel.vsim2(nsim=nsim, nobs=nobs)
            self.assertEqual(vol.shape, shape)
            self.assertEqual(argmodel.rsim(vol=vol).shape, shape)

            self.assertEqual(argmodel.vsim_last(nsim=nsim, nobs=nobs).shape,
                             (nsim, ))
            self.assertGreater(argmodel.vsim(nsim=nsim, nobs=nobs).min(), 0)
            self.assertGreater(argmodel.vsim2(nsim=nsim, nobs=nobs).min(), 0)

    def test_likelihoods(self):
        """Test likelihood functions."""

        theta = np.array([1, 1, 1])
        vol = np.array([1, 2, 3])

        self.assertIsInstance(likelihood_vol(theta, vol), float)
        self.assertIsInstance(likelihood_vol_grad(theta, vol), np.ndarray)
        self.assertIsInstance(likelihood_vol_hess(theta, vol), np.ndarray)

        self.assertEqual(likelihood_vol_grad(theta, vol).shape, theta.shape)
        self.assertEqual(likelihood_vol_hess(theta, vol).shape,
                         (theta.shape[0], theta.shape[0]))

        vol = np.array([1])

        self.assertRaises(AssertionError, lambda: likelihood_vol(theta, vol))

    def test_estimate_mle(self):
        """Test MLE estimation."""
        param_true = ARGparams()
        argmodel = ARG(param=param_true)
        nsim, nobs = 1, 500
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
        param_final, results = argmodel.estimate_mle(param_start=param_true,
                                                     vol=vol, model='vol')
        ratio = param_true.theta_vol / param_final.theta_vol

        self.assertIsInstance(param_final, ARGparams)
        self.assertIsInstance(results, so.optimize.OptimizeResult)
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=1e1)

        self.assertIsInstance(results.std_theta, np.ndarray)
        self.assertEqual(results.std_theta.shape, param_true.theta_vol.shape)

        self.assertIsInstance(results.tstat, np.ndarray)
        self.assertEqual(results.tstat.shape, param_true.theta_vol.shape)

    def test_momcond(self):
        """Test moment condition method."""
        theta = np.array([1, 1, 1])
        uarg = np.array([-1, 0, 1])
        argmodel = ARG(ARGparams())
        self.assertRaises(ValueError, lambda: argmodel.momcond(theta))
        fun = lambda: argmodel.momcond(theta, uarg=uarg)
        self.assertRaises(ValueError, fun)

        nsim, nobs = 1, 10
        instrlag = 2
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()

        moment, dmoment = argmodel.momcond(theta, vol=vol, uarg=uarg,
                                           zlag=instrlag)

        np.testing.assert_array_equal(argmodel.param.theta_vol, theta)

        momshape = (vol.shape[0]-instrlag, 2*uarg.shape[0]*instrlag)
        dmomshape = (2*uarg.shape[0]*instrlag, theta.shape[0])

        self.assertEqual(moment.shape, momshape)
        self.assertEqual(dmoment.shape, dmomshape)

    def test_gmmest(self):
        """Test GMM estimation."""
        param_true = ARGparams()
        argmodel = ARG(param=param_true)
        nsim, nobs = 1, 500
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
        uarg = np.linspace(.1, 10, 3) * 1j

        results = argmodel.estimate_gmm(param_true.theta_vol, vol=vol,
                                        uarg=uarg, zlag=2)

        self.assertIsInstance(results, Results)
        self.assertIsInstance(results.theta, np.ndarray)
        self.assertEqual(results.theta.shape[0], param_true.theta_vol.shape[0])


if __name__ == '__main__':
    ut.main()
