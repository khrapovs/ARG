#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import unittest as ut
import numpy as np

import ARG.arg as arg


class ARGTestCase(ut.TestCase):

    def test_truisms(self):
        """Test parameter class"""
        param = arg.ARGparams()
        self.assertTrue(isinstance(param.scale, float))
        self.assertTrue(isinstance(param.rho, float))
        self.assertTrue(isinstance(param.delta, float))
        self.assertTrue(isinstance(param.beta(), float))


if __name__ == '__main__':
    ut.main()