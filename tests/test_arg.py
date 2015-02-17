#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ARG class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import scipy.optimize as so

from argamma import ARG, ARGparams
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

        self.assertIsInstance(param.get_theta_vol(), np.ndarray)
        np.testing.assert_array_equal(param.get_theta_vol(), theta_true)

        scale, rho, delta = 1, 2, 3
        theta_true = [scale, rho, delta]
        param.update(theta_vol=theta_true)

        self.assertEqual(param.scale, scale)
        self.assertEqual(param.rho, rho)
        self.assertEqual(param.delta, delta)

    def test_param_class_ret(self):
        """Test parameter class for return parameters."""

        param = ARGparams()

        self.assertIsInstance(param.phi, float)
        self.assertIsInstance(param.price_ret, float)
        self.assertIsInstance(param.price_vol, float)

        phi, price_vol, price_ret = -.5, -5, 5
        theta_true = [phi, price_ret]
        param = ARGparams(phi=phi, price_ret=price_ret)

        self.assertIsInstance(param.get_theta_ret(), np.ndarray)
        np.testing.assert_array_equal(param.get_theta_ret(), theta_true)

        phi, price_vol, price_ret = 1, 2, 3
        theta_true = [phi, price_ret]
        param.update(theta_ret=theta_true)

        self.assertEqual(param.phi, phi)
        self.assertEqual(param.price_ret, price_ret)

    def test_param_class_joint(self):
        """Test parameter class for joint parameters."""

        scale, rho, delta = 1, 2, 3
        phi, price_vol, price_ret = -.5, -5, 5
        param = ARGparams(scale=scale, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret)

        theta_true = [scale, rho, delta, phi, price_ret]

        self.assertIsInstance(param.get_theta(), np.ndarray)
        np.testing.assert_array_equal(param.get_theta(), theta_true)

        theta_true = np.arange(5)
        param.update(theta=theta_true)

        np.testing.assert_array_equal(param.get_theta(), theta_true)

    def test_conversion_to_q(self):
        """Test conversion to Q measure."""
        scale, rho, delta = 1, 2, 3
        phi, price_vol, price_ret = -.5, -5, 5
        param = ARGparams(scale=scale, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret, price_vol=price_vol)
        argmodel = ARG(param=param)
        param_q = argmodel.convert_to_q()

        self.assertIsInstance(param_q, ARGparams)

        factor = 1/(1 + scale * (price_vol + argmodel.alpha(price_ret)))
        scale = scale * factor
        beta = param.beta * factor
        rho = scale * beta

        self.assertEqual(param_q.scale, scale)
        self.assertEqual(param_q.rho, rho)
        self.assertEqual(param_q.delta, delta)

    def test_uncond_moments(self):
        """Test unconditional moments of the ARG model."""

        scale, rho, delta = 1, 2, 3
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
            uarg = np.linspace(-50, 100, i)

            self.assertIsInstance(argmodel.afun(uarg), np.ndarray)
            self.assertIsInstance(argmodel.bfun(uarg), np.ndarray)
            self.assertIsInstance(argmodel.cfun(uarg), np.ndarray)

            self.assertEqual(uarg.shape, argmodel.afun(uarg).shape)
            self.assertEqual(uarg.shape, argmodel.bfun(uarg).shape)
            self.assertEqual(uarg.shape, argmodel.cfun(uarg).shape)

    def test_ret_functions(self):
        """Test functions alpha, beta, gamma of ARG model."""

        argmodel = ARG(ARGparams())
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, i)

            self.assertIsInstance(argmodel.alpha(uarg), np.ndarray)
            self.assertIsInstance(argmodel.beta(uarg), np.ndarray)
            self.assertIsInstance(argmodel.gamma(uarg), np.ndarray)

            self.assertEqual(uarg.shape, argmodel.alpha(uarg).shape)
            self.assertEqual(uarg.shape, argmodel.beta(uarg).shape)
            self.assertEqual(uarg.shape, argmodel.gamma(uarg).shape)

    def test_abc_derivatives(self):
        """Test derivatives of functions a, b, c of ARG model."""

        argmodel = ARG(ARGparams())
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, i)

            self.assertIsInstance(argmodel.dafun(uarg), np.ndarray)
            self.assertIsInstance(argmodel.dbfun(uarg), np.ndarray)

            self.assertEqual(argmodel.dafun(uarg).shape, (3, uarg.shape[0]))
            self.assertEqual(argmodel.dbfun(uarg).shape, (3, uarg.shape[0]))

    def test_load_data(self):
        """Test load data into the model object."""
        argmodel = ARG(ARGparams())

        self.assertEqual(argmodel.vol, None)
        self.assertEqual(argmodel.ret, None)

        argmodel.load_data(vol=0)

        self.assertEqual(argmodel.vol, 0)
        self.assertEqual(argmodel.ret, None)

        argmodel.load_data(ret=1)

        self.assertEqual(argmodel.vol, 0)
        self.assertEqual(argmodel.ret, 1)

        argmodel.load_data(ret=2, vol=3)

        self.assertEqual(argmodel.vol, 3)
        self.assertEqual(argmodel.ret, 2)

    def test_simulations(self):
        """Test simulation of ARG model."""

        argmodel = ARG(ARGparams())

        self.assertIsInstance(argmodel.vsim(), np.ndarray)
        self.assertIsInstance(argmodel.vsim2(), np.ndarray)

        vol = argmodel.vsim2()
        argmodel.load_data(vol=vol)

        self.assertIsInstance(argmodel.rsim(), np.ndarray)

        shapes = []
        nsim, nobs = 1, 1
        shapes.append((nobs, nsim))
        nsim, nobs = 2, 2
        shapes.append((nobs, nsim))
        nsim, nobs = int(1e3), int(1e3)
        shapes.append((nobs, nsim))

        for shape in shapes:
            nsim, nobs = shape

            self.assertEqual(argmodel.vsim(nsim=nsim, nobs=nobs).shape, shape)
            vol = argmodel.vsim2(nsim=nsim, nobs=nobs)
            argmodel.load_data(vol=vol)
            self.assertEqual(vol.shape, shape)
            self.assertEqual(argmodel.rsim().shape, shape)

            self.assertEqual(argmodel.vsim_last(nsim=nsim, nobs=nobs).shape,
                             (nsim, ))
            self.assertGreater(argmodel.vsim(nsim=nsim, nobs=nobs).min(), 0)
            self.assertGreater(argmodel.vsim2(nsim=nsim, nobs=nobs).min(), 0)

    def test_likelihoods(self):
        """Test likelihood functions."""

        theta_vol = np.array([1e-5, .9, 1.1])
        theta_ret = np.array([-.5, 1])
        theta = np.ones(5)
        vol = np.array([1, 2, 3])
        ret = np.array([4, 5, 6])

        argmodel = ARG(param=ARGparams())
        argmodel.load_data(vol=vol, ret=ret)

        self.assertIsInstance(argmodel.likelihood_vol(theta_vol), float)
        self.assertIsInstance(argmodel.likelihood_ret(theta_ret), float)
        self.assertIsInstance(argmodel.likelihood_joint(theta), float)

    def test_estimate_mle(self):
        """Test MLE estimation."""
        param_true = ARGparams()
        argmodel = ARG(param=param_true)
        nsim, nobs = 1, 500
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
        argmodel.load_data(vol=vol)
        fun = lambda: argmodel.estimate_mle(param_start=param_true,
                                            model='zzz')

        self.assertRaises(ValueError, fun)

        param_final, results = argmodel.estimate_mle(param_start=param_true,
                                                     model='vol')
        ratio = param_true.get_theta_vol() / param_final.get_theta_vol()

        self.assertIsInstance(param_final, ARGparams)
        self.assertIsInstance(results, so.optimize.OptimizeResult)
        np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=1e1)

        self.assertIsInstance(results.std_theta, np.ndarray)
        self.assertEqual(results.std_theta.shape,
                         param_true.get_theta_vol().shape)

        self.assertIsInstance(results.tstat, np.ndarray)
        self.assertEqual(results.tstat.shape, param_true.get_theta_vol().shape)

    def test_momcond_exceptions(self):
        """Test moment condition method."""
        theta = np.array([1, 1, 1])
        uarg = np.array([-1, 0, 1])
        argmodel = ARG(ARGparams())
        self.assertRaises(ValueError, lambda: argmodel.momcond_vol(theta))
        fun = lambda: argmodel.momcond_vol(theta, uarg=uarg)
        self.assertRaises(ValueError, fun)

    def test_momcond_vol(self):
        """Test moment condition method."""
        theta = np.array([1, 1, 1])
        uarg = np.array([-1, 0, 1])
        argmodel = ARG(ARGparams())

        nsim, nobs = 1, 10
        instrlag = 2
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
        argmodel.load_data(vol=vol)

        moment, dmoment = argmodel.momcond_vol(theta, uarg=uarg, zlag=instrlag)

        np.testing.assert_array_equal(argmodel.param.get_theta_vol(), theta)

        momshape = (vol.shape[0]-instrlag, 2*uarg.shape[0]*instrlag)
        dmomshape = (2*uarg.shape[0]*instrlag, theta.shape[0])

        self.assertEqual(moment.shape, momshape)
        self.assertEqual(dmoment.shape, dmomshape)

    def test_momcond_ret(self):
        """Test moment condition method."""
        theta = np.array([1, 1])
        uarg = np.array([-1, 0, 1])
        argmodel = ARG(ARGparams())

        nsim, nobs = 1, 10
        instrlag = 2
        vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
        argmodel.load_data(vol=vol)
        ret = argmodel.rsim()
        argmodel.load_data(ret=ret)

        moment, dmoment = argmodel.momcond_ret(theta, uarg=uarg, zlag=instrlag)

        # np.testing.assert_array_equal(argmodel.param.get_theta_ret(), theta)

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
        argmodel.load_data(vol=vol)
        uarg = np.linspace(.1, 10, 3) * 1j

        param_final, results = argmodel.estimate_gmm(
            param_true.get_theta_vol(), uarg=uarg, zlag=2)

        self.assertIsInstance(results, Results)
        self.assertIsInstance(param_final.get_theta_vol(), np.ndarray)
        self.assertEqual(results.theta.shape[0],
                         param_true.get_theta_vol().shape[0])


if __name__ == '__main__':
    ut.main()
