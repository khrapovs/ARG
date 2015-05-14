#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for ARGparams class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from argamma import ARGparams


class ARGparamsTestCase(ut.TestCase):
    """Test ARGparams class."""

    def test_param_class_vol(self):
        """Test parameter class for volatility parameters."""

        param = ARGparams()

        self.assertIsInstance(param.mean, float)
        self.assertIsInstance(param.rho, float)
        self.assertIsInstance(param.delta, float)
        self.assertIsInstance(param.get_beta(), float)
        self.assertIsInstance(param.get_scale(), float)

        mean, rho, delta = .1, .9, 1.1
        theta_true = [mean, rho, delta]
        param = ARGparams(mean=mean, rho=rho, delta=delta)

        self.assertIsInstance(param.get_theta_vol(), np.ndarray)
        self.assertIsInstance(param.get_scale(), float)
        self.assertIsInstance(param.get_mean(), float)
        np.testing.assert_array_equal(param.get_theta_vol(), theta_true)

        mean, rho, delta = .01, .5, 3
        theta_true = [mean, rho, delta]
        param.update(theta_vol=theta_true)

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.rho, rho)
        self.assertEqual(param.delta, delta)

    def test_param_class_ret(self):
        """Test parameter class for return parameters."""

        param = ARGparams()

        self.assertIsInstance(param.phi, float)
        self.assertIsInstance(param.price_ret, float)
        self.assertIsInstance(param.price_vol, float)

        phi, price_ret = -.5, 5
        theta_true = [phi, price_ret]
        param = ARGparams(phi=phi, price_ret=price_ret)

        self.assertIsInstance(param.get_theta_ret(), np.ndarray)
        np.testing.assert_array_equal(param.get_theta_ret(), theta_true)

        phi, price_ret = .1, 3
        theta_true = [phi, price_ret]
        param.update(theta_ret=theta_true)

        self.assertEqual(param.phi, phi)
        self.assertEqual(param.price_ret, price_ret)

    def test_param_class_joint(self):
        """Test parameter class for joint parameters."""

        mean, rho, delta = .1**2, .9, 1.1
        phi, price_ret = -.5, 1
        param = ARGparams(mean=mean, rho=rho, delta=delta,
                          phi=phi, price_ret=price_ret)

        theta_true = [mean, rho, delta, phi, price_ret]

        self.assertIsInstance(param.get_theta(), np.ndarray)
        np.testing.assert_array_equal(param.get_theta(), theta_true)

        theta_true = np.array(theta_true) / 10
        param.update(theta=theta_true)

        np.testing.assert_array_equal(param.get_theta(), theta_true)


if __name__ == '__main__':
    ut.main()
