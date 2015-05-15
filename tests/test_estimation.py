#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ARG class estimation capabilities.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import scipy.optimize as sco

from argamma import ARG, ARGparams
from argamma.mygmm import Results


class ARGestimationTestCase(ut.TestCase):
    """Test ARG model class estimation capabilities."""

    def test_likelihoods(self):
        """Test likelihood functions."""

        theta_vol = np.array([1e-2, .9, 1.1])
        theta_ret = np.array([-.5, 1])
        theta = np.ones(5)
        vol = np.array([1, 2, 3])
        ret = np.array([4, 5, 6])
        param = ARGparams()
        param.update(theta_vol=theta_vol, theta_ret=theta_ret)

        argmodel = ARG()
        argmodel.load_data(vol=vol, ret=ret)

        self.assertIsInstance(argmodel.likelihood_vol(theta_vol), float)
        self.assertIsInstance(argmodel.likelihood_ret(theta_ret, theta_vol),
                              float)
        self.assertIsInstance(argmodel.likelihood_joint(theta), float)

    def test_estimate_mle(self):
        """Test MLE estimation."""
        param_true = ARGparams()
        argmodel = ARG()
        nsim, nobs = 1, 500
        vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
        argmodel.load_data(vol=vol)
        fun = lambda: argmodel.estimate_mle(param_start=param_true,
                                            model='zzz')

        self.assertRaises(ValueError, fun)

        param_final, results = argmodel.estimate_mle(param_start=param_true,
                                                     model='vol')
        ratio = param_true.get_theta_vol() / param_final.get_theta_vol()

        self.assertIsInstance(param_final, ARGparams)
        self.assertIsInstance(results, sco.optimize.OptimizeResult)
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
        argmodel = ARG()
        self.assertRaises(ValueError, lambda: argmodel.momcond_vol(theta))
        fun = lambda: argmodel.momcond_vol(theta, uarg=uarg)
        self.assertRaises(ValueError, fun)

    def test_momcond_vol(self):
        """Test moment condition method."""
        theta = np.array([1, 1, 1])
        uarg = np.array([-1, 0, 1])
        param = ARGparams()
        argmodel = ARG()

        nsim, nobs = 1, 10
        instrlag = 2
        vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param).flatten()
        argmodel.load_data(vol=vol)

        moment, dmoment = argmodel.momcond_vol(theta, uarg=uarg, zlag=instrlag)

        momshape = (vol.shape[0]-instrlag, 2*uarg.shape[0]*(instrlag+1))
        dmomshape = (2*uarg.shape[0]*(instrlag+1), theta.shape[0])

        self.assertEqual(moment.shape, momshape)
        self.assertEqual(dmoment.shape, dmomshape)

    def test_momcond_ret(self):
        """Test moment condition method."""
        uarg = np.array([-1, 0, 1])
        param = ARGparams()
        theta_ret = param.get_theta_ret()
        theta_vol = param.get_theta_vol()
        argmodel = ARG()

        nsim, nobs = 1, 10
        instrlag = 2
        vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param).flatten()
        argmodel.load_data(vol=vol)
        ret = argmodel.rsim(param=param)
        argmodel.load_data(ret=ret)

        moment, dmoment = argmodel.momcond_ret(theta_ret, theta_vol=theta_vol,
                                               uarg=uarg, zlag=instrlag)

        momshape = (vol.shape[0]-instrlag, 2*uarg.shape[0]*(instrlag+1))
        dmomshape = (2*uarg.shape[0]*(instrlag+1), theta_ret.shape[0])

        self.assertEqual(moment.shape, momshape)
        self.assertEqual(dmoment.shape, dmomshape)

    def test_momcond_joint(self):
        """Test moment condition method."""
        uarg = np.array([-1, 0, 1])
        param = ARGparams()
        argmodel = ARG()

        nsim, nobs = 1, 10
        instrlag = 2
        vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param).flatten()
        argmodel.load_data(vol=vol)
        ret = argmodel.rsim(param=param)
        argmodel.load_data(ret=ret)

        moment, dmoment = argmodel.momcond_joint(param.get_theta(), uarg=uarg,
                                                 zlag=instrlag)

        momshape = (vol.shape[0]-instrlag, 4*uarg.shape[0]*(instrlag+1))
        dmomshape = (4*uarg.shape[0]*(instrlag+1), param.get_theta().shape[0])

        self.assertEqual(moment.shape, momshape)
        self.assertEqual(dmoment.shape, dmomshape)

    def test_gmmest(self):
        """Test GMM estimation."""
        param_true = ARGparams()
        argmodel = ARG()
        nsim, nobs = 1, 500
        vol = argmodel.vsim(nsim=nsim, nobs=nobs, param=param_true).flatten()
        argmodel.load_data(vol=vol)
        uarg = np.linspace(.1, 10, 3) * 1j

        param_final, results = argmodel.estimate_gmm(param_start=param_true,
                                                     uarg=uarg, zlag=2)

        self.assertIsInstance(results, Results)
        self.assertIsInstance(param_final.get_theta_vol(), np.ndarray)
        self.assertEqual(results.theta.shape[0],
                         param_true.get_theta_vol().shape[0])


if __name__ == '__main__':
    ut.main()
