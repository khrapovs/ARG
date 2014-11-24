#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ARG class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import scipy.optimize as so

from ARG import ARGparams, ARG, likelihood_vol

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class ARGTestCase(ut.TestCase):
    """Test ARG, ARGparams model classes."""

    def test_param_class(self):
        """Test parameter class."""

        param = ARGparams()

        self.assertIsInstance(param.scale, float)
        self.assertIsInstance(param.rho, float)
        self.assertIsInstance(param.delta, float)
        self.assertIsInstance(param.beta, float)

        scale, rho, delta = .1, 0, 0
        theta_true = [scale, rho, delta]
        param = ARGparams(scale=scale, rho=rho, delta=delta)
        self.assertIsInstance(param.theta, np.ndarray)
        np.testing.assert_array_equal(param.theta, theta_true)

        scale, rho, delta = 1, 2, 3
        theta_true = [scale, rho, delta]
        param = ARGparams(theta=theta_true)
        self.assertEqual(param.scale, scale)
        self.assertEqual(param.rho, rho)
        self.assertEqual(param.delta, delta)

        theta = [0, 0]
        self.assertRaises(AssertionError, lambda: ARGparams(theta=theta))

    def test_uncond_moments(self):
        """Test unconditional moments of the ARG model."""

        argmodel = ARG()

        self.assertIsInstance(argmodel.umean(), float)
        self.assertIsInstance(argmodel.uvar(), float)
        self.assertIsInstance(argmodel.ustd(), float)

        # TODO : test using symbolic library
        # that these moments coincide with derivatives of a, b, c

    def test_abc_functions(self):
        """Test functions a, b, c of ARG model."""

        argmodel = ARG()
        uarg = np.linspace(-50, 100, 100)

        self.assertIsInstance(argmodel.afun(uarg), np.ndarray)
        self.assertIsInstance(argmodel.bfun(uarg), np.ndarray)
        self.assertIsInstance(argmodel.cfun(uarg), np.ndarray)

        self.assertEqual(uarg.shape, argmodel.afun(uarg).shape)
        self.assertEqual(uarg.shape, argmodel.bfun(uarg).shape)
        self.assertEqual(uarg.shape, argmodel.cfun(uarg).shape)

    def test_abc_derivatives(self):
        """Test derivatives of functions a, b, c of ARG model."""

        argmodel = ARG()
        uarg = np.linspace(-50, 100, 100)

        self.assertIsInstance(argmodel.dafun(uarg), np.ndarray)
        self.assertIsInstance(argmodel.dbfun(uarg), np.ndarray)

        self.assertEqual(argmodel.dafun(uarg).shape, (3, uarg.shape[0]))
        self.assertEqual(argmodel.dbfun(uarg).shape, (3, uarg.shape[0]))

    def test_simulations(self):
        """Test simulation of ARG model."""

        argmodel = ARG()

        self.assertIsInstance(argmodel.vsim(), np.ndarray)
        self.assertIsInstance(argmodel.vsim2(), np.ndarray)

        nsim, nobs = 1, 1
        shape = (nsim, nobs)
        self.assertEqual(argmodel.vsim(nsim=nsim, nobs=nobs).shape, shape)
        self.assertEqual(argmodel.vsim2(nsim=nsim, nobs=nobs).shape, shape)
        self.assertEqual(argmodel.vsim_last(nsim=nsim, nobs=nobs).shape,
                         (nsim, ))

        nsim, nobs = 2, 2
        shape = (nsim, nobs)
        self.assertEqual(argmodel.vsim(nsim=nsim, nobs=nobs).shape, shape)
        self.assertEqual(argmodel.vsim2(nsim=nsim, nobs=nobs).shape, shape)
        self.assertEqual(argmodel.vsim_last(nsim=nsim, nobs=nobs).shape,
                         (nsim, ))

        nsim, nobs = int(1e3), int(1e3)
        self.assertGreater(argmodel.vsim(nsim=nsim, nobs=nobs).all(), 0)
        self.assertGreater(argmodel.vsim2(nsim=nsim, nobs=nobs).all(), 0)

    def test_likelihoods(self):
        """Test likelihood functions."""

        theta = np.array([1, 1, 1])
        vol = np.array([1, 2, 3])
        self.assertIsInstance(likelihood_vol(theta, vol), float)
        vol = np.array([1])
        self.assertRaises(AssertionError, lambda: likelihood_vol(theta, vol))

    def test_load_data(self):
        """Test load data method."""
        vol = np.array([1, 2, 3])
        argmodel = ARG()
        argmodel.load_data(vol=vol)
        np.testing.assert_array_equal(argmodel.vol, vol)

    def test_estimate_mle(self):
        """Test MLE estimation."""
        param_true = ARGparams()
        argmodel = ARG(param=param_true)
        nsim, nobs = 1, 500
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
        argmodel.load_data(vol=vol)
        param_final, results = argmodel.estimate_mle(param_start=param_true)
        ratio = param_true.theta / param_final.theta

        self.assertIsInstance(param_final, ARGparams)
        self.assertIsInstance(results, so.optimize.OptimizeResult)
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=1e1)


if __name__ == '__main__':
    ut.main()
