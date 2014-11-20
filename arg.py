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

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


class ARG(object):

    def __init__(self):
        pass


if __name__ == '__main__':
    from ARG.usage_example import test_arg
    test_arg()