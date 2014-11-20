#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARG model class

References
----------
.. [1] Stanislav Khrapov and Eric Renault (2014)
    "Affine Option Pricing Model in Discrete Time",
    working paper, New Economic School.
    <https://sites.google.com/site/khrapovs/research
    /Renault-Khrapov-2012-Affine-Option-Pricing.pdf>

.. [2] Christian Gourieroux and Joann Jasiak (2006)
    "Autoregressive Gamma Processes",
    2006, Journal of Forecasting, 25(2), 129–152. doi:10.1002/for.978

.. [3] Serge Darolles, Christian Gourieroux, and Joann Jasiak (2006)
    "Structural Laplace Transform and Compound Autoregressive Models"
    Journal of Time Series Analysis, 27(4), 477–503.
    doi:10.1111/j.1467-9892.2006.00479.x

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class ARGparams(object):
    """Class for ARG model parameters.

    Attributes
    ----------
    scale : float
    rho : float
    delta : float

    """
    def __init__(self, scale=.01, rho=.9, delta=1.1):
        """Initialize the class instance.

        """
        self.scale = scale
        self.rho = rho
        self.delta = delta

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

    def beta(self):
        """Compute beta parameter of the model.

        Returns
        -------
        beta : float

        """
        return self.rho / self.scale


class ARG(object):
    """Class for ARG model.

    Attributes
    ----------
    param : ARGparams instance
        Parameters of the model

    """
    def __init__(self, param=ARGparams()):
        """INitialize class instance.

        """
        self.param = param

    def afun(self, uarg):
        """Function a().

        .. math::
            a\left(u\right)=\frac{\rho u}{1+cu}

        Parameters
        ----------
        uarg : array

        Returns
        -------
        a(u) : array

        """
        return self.param.rho * uarg / (1 + self.param.scale * uarg)

    def bfun(self, uarg):
        """Function b().

        .. math::
            b\left(u\right)=\delta\log\left(1+cu\right)

        Parameters
        ----------
        uarg : array

        Returns
        -------
        b(u) : array

        """
        return self.param.delta * np.log(1 + self.param.scale * uarg)

    def cfun(self, uarg):
        """Function c().

        .. math::
            c\left(u\right)=\delta\log\left\{ 1+\frac{cu}{1-\rho}\right\}

        Parameters
        ----------
        uarg : array

        Returns
        -------
        c(u) : array

        """
        return self.param.delta \
            * np.log(1 + self.param.scale * uarg / (1-self.param.rho))

    def plot_abfun(self, uarg):
        """Plot a() and b() functions on the same plot.

        """
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 3, 1)
        plt.plot(uarg, self.afun(uarg))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$a(u)$')
        plt.xlabel('$u$')

        plt.subplot(1, 3, 2)
        plt.plot(uarg, self.bfun(uarg))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$b(u)$')
        plt.xlabel('$u$')

        plt.subplot(1, 3, 3)
        plt.plot(uarg, self.cfun(uarg))
        plt.axhline(0)
        plt.axvline(0)
        plt.ylabel('$c(u)$')
        plt.xlabel('$u$')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from ARG.usage_example import test_arg
    test_arg()