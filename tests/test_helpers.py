#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for helper functions.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from argamma import days_in_year, days_from_maturity


class HelpersTestCase(ut.TestCase):
    """Test for helper functions."""

    def test_days_in_year(self):
        """Test number of days in a year."""

        days = days_in_year()
        self.assertIsInstance(days, int)

    def test_periods_maturity(self):
        """Test conversion between days and maturity."""

        days = 30
        maturity = days / days_in_year()
        periods = days_from_maturity(maturity)

        np.testing.assert_array_equal(np.array(days), periods)

        days = np.arange(10, 20, 2)
        maturity = days / days_in_year()
        periods = days_from_maturity(maturity)

        np.testing.assert_array_equal(np.array(days), periods)


if __name__ == '__main__':

    ut.main()
