#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARG parameters class

"""
from __future__ import print_function, division


class ARGparams(object):
    """Class for ARG model parameters.

    Attributes
    ----------
    scale : float
    rho : float
    delta : float

    Methods
    -------
    convert_to_theta
        Convert parameters to the vector

    """
    def __init__(self, scale=.001, rho=.9, delta=1.1, theta=None):
        """Initialize the class instance.

        """
        if theta:
            assert len(theta) == 3, "Wrong number of parameters in theta!"
            [scale, rho, delta] = theta
        self.scale = scale
        self.rho = rho
        self.delta = delta
        assert scale > 0, "Scale must be greater than zero!"
        self.beta = self.rho / self.scale
        self.theta = [scale, rho, delta]

    def __repr__(self):
        """This is what is shown when you interactively explore the instance.

        """
        params = (self.scale, self.rho, self.delta)
        string = "scale = %.2f, rho = %.2f, delta = %.2f" % params
        return string

    def __str__(self):
        """This is what is shown when you print() the instance.

        """
        return self.__repr__()
