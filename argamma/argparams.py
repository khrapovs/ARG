#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARG parameters class

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['ARGparams']


class ARGparams(object):
    """Class for ARG model parameters.

    Attributes
    ----------
    scale : float
    rho : float
    delta : float
    beta : float
    theta : array

    Raises
    ------
    AssertionError

    """
    def __init__(self, scale=.001, rho=.9, delta=1.1, theta=None):
        """Initialize the class instance.

        """
        if not theta is None:
            assert len(theta) == 3, "Wrong number of parameters in theta!"
            [scale, rho, delta] = theta
        self.scale = scale
        self.rho = rho
        self.delta = delta
        assert scale > 0, "Scale must be greater than zero!"
        self.beta = self.rho / self.scale
        self.theta = np.array([scale, rho, delta])

    # def __repr__(self):
    #     """This is what is shown when you interactively explore the instance.

    #     """
    #     params = (self.scale, self.rho, self.delta)
    #     string = "scale = %.2f, rho = %.2f, delta = %.2f" % params
    #     return string

    def __str__(self):
        """This is what is shown when you print() the instance.

        """
        params = (self.scale, self.rho, self.delta)
        string = "scale = %.2f, rho = %.2f, delta = %.2f" % params
        return string
