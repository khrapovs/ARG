#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions.

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['days_in_year', 'periods_from_maturity']


def days_in_year():
    """Return number of days in a year.

    Returns
    -------
    int
        Number of days in a year

    """
    return 365


def periods_from_maturity(maturity):
    """Return number of days in a year.

    Parameters
    ----------
    maturity : float
        Fraction of a year, i.e. 365

    Returns
    -------
    int
        Number of days in a year

    """
    return np.atleast_1d(maturity * days_in_year()).astype(int)


if __name__ == '__main__':

    pass
