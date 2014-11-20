#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import unittest as ut
import numpy as np

import ARG.arg as arg


class ARGTestCase(ut.TestCase):

    def test_param_class(self):
        """Test parameter class."""

        param = arg.ARGparams()

        self.assertIsInstance(param.scale, float)
        self.assertIsInstance(param.rho, float)
        self.assertIsInstance(param.delta, float)
        self.assertIsInstance(param.beta, float)

    def test_uncond_moments(self):
        """Test unconditional moments of the ARG model."""

        argmodel = arg.ARG()

        self.assertIsInstance(argmodel.umean(), float)
        self.assertIsInstance(argmodel.uvar(), float)
        self.assertIsInstance(argmodel.ustd(), float)

        # TODO : test using symbolic library
        # that these moments coincide with derivatives of a, b, c


    def test_abc_functions(self):
        """Test functions a, b, c of ARG model."""

        argmodel = arg.ARG()
        uarg = np.linspace(-50, 100, 100)

        self.assertIsInstance(argmodel.afun(uarg), np.ndarray)
        self.assertIsInstance(argmodel.bfun(uarg), np.ndarray)
        self.assertIsInstance(argmodel.cfun(uarg), np.ndarray)

        self.assertEqual(uarg.shape, argmodel.afun(uarg).shape)
        self.assertEqual(uarg.shape, argmodel.bfun(uarg).shape)
        self.assertEqual(uarg.shape, argmodel.cfun(uarg).shape)


if __name__ == '__main__':
    ut.main()