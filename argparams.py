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

    """
    def __init__(self, scale=.001, rho=.9, delta=1.1):
        """Initialize the class instance.

        """
        self.scale = scale
        self.rho = rho
        self.delta = delta
        self.beta = self.rho / self.scale

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
