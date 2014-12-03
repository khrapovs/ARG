#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage example.

"""
from __future__ import print_function, division

import numpy as np

from ARG import ARG, ARGparams

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


def play_with_arg():
    param = ARGparams()
    print(param)

    argmodel = ARG()
    uarg = np.linspace(-50, 100, 100)
    argmodel.plot_abc(uarg)

    argmodel.plot_vsim()

    vol = argmodel.vsim_last(nsim=10)
    print(vol.shape)

    argmodel.plot_vlast_density(nsim=1000)

def estimate_mle():
    """Try MLE estimator."""
    param_true = ARGparams()
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
    argmodel.load_data(vol=vol)
    param_final, results = argmodel.estimate_mle(param_start=param_true)

    print('True parameter:', param_true)
    print('Final parameter: ', param_final)
    print(type(results))

def estimate_gmm():
    """Try GMM estimator."""
    param_true = ARGparams()
    argmodel = ARG(param=param_true)
    nsim, nobs = 1, 500
    vol = argmodel.vsim(nsim=nsim, nobs=nobs).flatten()
    argmodel.load_data(vol=vol)
    uarg = np.linspace(.1, 10, 3) * 1j
    results = argmodel.gmmest(param_true.theta, uarg=uarg, instrlag=2)

    print('True parameter:', param_true)
    print('Final parameter: ', ARGparams(theta=results.theta))
    print('Std: ', results.tstat)
    results.print_results()


if __name__ == '__main__':
    #play_with_arg()
    #estimate_mle()
    estimate_gmm()