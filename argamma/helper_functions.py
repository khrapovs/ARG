#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions.

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['days_in_year', 'periods_from_maturity', 'get_minmax_periods']


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
    return np.around(maturity * days_in_year(), decimals=1)


def get_minmax_periods(periods):
    """Get minimum and maximum periods.

    Returns
    -------
    int
        Minimum periods
    int
        Maximum periods

    """
    return np.min(periods), np.max(periods)


if __name__ == '__main__':

    pass
