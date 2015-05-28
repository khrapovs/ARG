#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ARG class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from argamma import ARG, ARGparams


class ARGTestCase(ut.TestCase):
    """Test ARG model class."""

    def test_conversion_to_q(self):
        """Test conversion to Q measure."""
        mean, rho, delta = .1**2, .9, 1.1
        phi, price_vol, price_ret = -.5, -1, .5
        param = ARGparams(mean=mean, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret, price_vol=price_vol)
        scale = param.get_scale()
        argmodel = ARG()
        param_q = argmodel.convert_to_q(param)

        self.assertIsInstance(param_q, ARGparams)

        factor = 1/(1 + scale * (price_vol \
            + argmodel.alpha(price_ret, param)))
        scale = scale * factor
        beta = param.get_beta() * factor
        rho = scale * beta

        self.assertEqual(param_q.scale, scale)
        self.assertEqual(param_q.rho, rho)
        self.assertEqual(param_q.delta, delta)

    def test_uncond_moments(self):
        """Test unconditional moments of the ARG model."""

        mean, rho, delta = 1, .9, 3
        param = ARGparams(mean=mean, rho=rho, delta=delta)
        argmodel = ARG()

        self.assertIsInstance(argmodel.umean(param), float)
        self.assertIsInstance(argmodel.uvar(param), float)
        self.assertIsInstance(argmodel.ustd(param), float)

        # TODO : test using symbolic library
        # that these moments coincide with derivatives of a, b, c

    def test_abc_functions(self):
        """Test functions a, b, c of ARG model."""

        param = ARGparams()
        argmodel = ARG()
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, i)

            self.assertIsInstance(argmodel.afun(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.bfun(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.cfun(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.afun_q(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.bfun_q(uarg, param), np.ndarray)

            self.assertEqual(uarg.shape, argmodel.afun(uarg, param).shape)
            self.assertEqual(uarg.shape, argmodel.bfun(uarg, param).shape)
            self.assertEqual(uarg.shape, argmodel.cfun(uarg, param).shape)
            self.assertEqual(uarg.shape, argmodel.afun_q(uarg, param).shape)
            self.assertEqual(uarg.shape, argmodel.bfun_q(uarg, param).shape)

    def test_ret_functions(self):
        """Test functions alpha, beta, gamma of ARG model."""

        param = ARGparams()
        argmodel = ARG()
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, i)

            self.assertIsInstance(argmodel.alpha(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.beta(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.gamma(uarg, param), np.ndarray)

            self.assertEqual(uarg.shape, argmodel.alpha(uarg, param).shape)
            self.assertEqual(uarg.shape, argmodel.beta(uarg, param).shape)
            self.assertEqual(uarg.shape, argmodel.gamma(uarg, param).shape)

    def test_joint_functions(self):
        """Test functions lfun, gfun of ARG model."""
        param = ARGparams()
        argmodel = ARG()
        uarg, varg = 1., 2.

        lfun, gfun = argmodel.lgfun(uarg, varg, param)
        lfunq, gfunq = argmodel.lgfun_q(uarg, varg, param)

        self.assertIsInstance(lfun, float)
        self.assertIsInstance(gfun, float)
        self.assertIsInstance(lfunq, float)
        self.assertIsInstance(gfunq, float)

        uarg, varg = 1., np.arange(5)
        lfun, gfun = argmodel.lgfun(uarg, varg, param)
        lfunq, gfunq = argmodel.lgfun_q(uarg, varg, param)

        self.assertEqual(lfun.shape, varg.shape)
        self.assertEqual(gfun.shape, varg.shape)
        self.assertEqual(lfunq.shape, varg.shape)
        self.assertEqual(gfunq.shape, varg.shape)

    def test_joint_multiperiod_functions(self):
        """Test functions psin, upsn of ARG model."""
        param = ARGparams()
        varg = np.atleast_2d(1.)
        maturity = 5/365
        argmodel = ARG()
        argmodel.maturity = maturity

        psi, ups = argmodel.ch_fun_elements(varg, param)
        self.assertEqual(psi.shape, (1, 1))
        self.assertEqual(ups.shape, (1, 1))

        varg = np.arange(5)[:, np.newaxis]

        psi, ups = argmodel.ch_fun_elements(varg, param)
        self.assertEqual(psi.shape, varg.shape)
        self.assertEqual(ups.shape, varg.shape)

    def test_char_fun_ret_q(self):
        """Test risk-neutral return charcteristic function."""
        param = ARGparams()
        nobs = 5
        vol = np.arange(nobs)
        maturity = np.ones(nobs) * 5/365
        riskfree = 0.
        argmodel = ARG()
        argmodel.riskfree = riskfree
        argmodel.maturity = maturity
        argmodel.load_data(vol=vol)
        varg = np.atleast_2d(1.)

        self.assertEqual(argmodel.char_fun_ret_q(varg, param).shape, (1, nobs))

        narg = 10
        varg = np.ones((narg, 1))
        size = (narg, nobs)
        self.assertEqual(argmodel.char_fun_ret_q(varg, param).shape, size)

        param = ARGparams()
        vol = 1.
        argmodel = ARG()
        argmodel.maturity = 5/365
        argmodel.riskfree = 0.
        argmodel.load_data(vol=vol)
        varg = np.atleast_2d(1.)

        cfun = argmodel.char_fun_ret_q(varg, param)

        self.assertIsInstance(cfun[0, 0], complex)

    def test_charfun(self):
        """Test risk-neutral return charcteristic function."""
        param = ARGparams()
        vol = 1.
        maturity = 5/365
        riskfree = 0.
        argmodel = ARG(param=param)
        argmodel.riskfree = riskfree
        argmodel.maturity = maturity
        argmodel.load_data(vol=vol)
        narg = 5
        varg = np.arange(narg)[:, np.newaxis]
        cfun = argmodel.charfun(varg)

        self.assertIsInstance(cfun, np.ndarray)
        self.assertEqual(argmodel.charfun(varg).size, narg)

        nobs = 10
        vol = np.ones(nobs)
        maturity = maturity * np.ones(nobs)
        argmodel = ARG(param=param)
        argmodel.riskfree = riskfree
        argmodel.maturity = maturity
        argmodel.load_data(vol=vol)
        self.assertEqual(argmodel.charfun(varg).shape, (narg, nobs))

    def test_cos_restriction(self):
        """Test cos_restriction method."""
        param = ARGparams()
        vol = 1.
        maturity = 5/365
        riskfree = 0.
        argmodel = ARG(param=param)
        argmodel.riskfree = riskfree
        argmodel.maturity = maturity

        argmodel.load_data(vol=vol)
        alim, blim = argmodel.cos_restriction()

        self.assertIsInstance(alim[0], float)
        self.assertIsInstance(blim[0], float)

        nobs = 10
        vol = np.ones(nobs)
        argmodel = ARG(param=param)
        argmodel.riskfree = riskfree
        argmodel.maturity = maturity
        argmodel.load_data(vol=vol)
        alim, blim = argmodel.cos_restriction()

        self.assertEqual(alim.size, nobs)
        self.assertEqual(blim.size, nobs)

    def test_abc_derivatives(self):
        """Test derivatives of functions a, b, c of ARG model."""

        param = ARGparams()
        argmodel = ARG()
        for i in [1, 10]:
            uarg = np.linspace(-50, 100, i)

            self.assertIsInstance(argmodel.dafun(uarg, param), np.ndarray)
            self.assertIsInstance(argmodel.dbfun(uarg, param), np.ndarray)

            self.assertEqual(argmodel.dafun(uarg, param).shape,
                             (3, uarg.shape[0]))
            self.assertEqual(argmodel.dbfun(uarg, param).shape,
                             (3, uarg.shape[0]))

    def test_load_data(self):
        """Test load data into the model object."""
        argmodel = ARG()

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

        param = ARGparams()
        argmodel = ARG()

        self.assertIsInstance(argmodel.vsim(param=param), np.ndarray)
        self.assertIsInstance(argmodel.vsim2(param=param), np.ndarray)

        vol = argmodel.vsim2(param=param)
        argmodel.load_data(vol=vol)

        self.assertIsInstance(argmodel.rsim(param=param), np.ndarray)

        shapes = []
        nsim, nobs = 1, 1
        shapes.append((nobs, nsim))
        nsim, nobs = 2, 2
        shapes.append((nobs, nsim))
        nsim, nobs = int(1e3), int(1e3)
        shapes.append((nobs, nsim))

        for shape in shapes:
            nsim, nobs = shape

            self.assertEqual(argmodel.vsim(nsim=nsim, nobs=nobs,
                                           param=param).shape, shape)
            vol = argmodel.vsim2(nsim=nsim, nobs=nobs, param=param)
            argmodel.load_data(vol=vol)
            self.assertEqual(vol.shape, shape)
            self.assertEqual(argmodel.rsim(param=param).shape, shape)

            self.assertEqual(argmodel.vsim_last(nsim=nsim, nobs=nobs,
                                                param=param).shape, (nsim, ))
            self.assertGreater(argmodel.vsim(nsim=nsim, nobs=nobs,
                                             param=param).min(), 0)
            self.assertGreater(argmodel.vsim2(nsim=nsim, nobs=nobs,
                                              param=param).min(), 0)

    def test_option_premium(self):
        """Test option pricing for the model."""
        price, strike = 100, 90
        riskfree, maturity = 0, 5/365
        moneyness = np.log(strike/price) - riskfree * maturity
        vol = .2**2/365
        call = True

        rho = .55
        delta = .75
        mean = .2**2/365
        phi = -.0
        price_vol = -16.0
        price_ret = 20.95

        param = ARGparams(mean=mean, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret, price_vol=price_vol)
        argmodel = ARG(param=param)

        premium = argmodel.option_premium(vol=vol, moneyness=moneyness,
                                          maturity=maturity, riskfree=riskfree,
                                          call=call)
        self.assertEqual(premium.shape, (1,))

        nobs = 10
        strike = np.exp(np.linspace(-.1, .1, nobs))
        moneyness = np.log(strike/price) - riskfree * maturity
        maturity = 5/365 * np.ones(10)

        premium = argmodel.option_premium(vol=vol, moneyness=moneyness,
                                          maturity=maturity, riskfree=riskfree,
                                          call=call)
        self.assertEqual(premium.shape, strike.shape)

        data = {'vol': vol, 'moneyness': moneyness, 'maturity': maturity,
                'riskfree': riskfree, 'call': call}
        premium = argmodel.option_premium(data=data)
        self.assertEqual(premium.shape, strike.shape)


if __name__ == '__main__':
    ut.main()
