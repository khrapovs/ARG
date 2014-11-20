#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np

import ARG.arg as arg

def test_arg():
    param = arg.ARGparams()
    print(param)

    argmodel = arg.ARG()
    uarg = np.linspace(-1, 1, 10)
    afun = argmodel.afun(uarg)
    bfun = argmodel.bfun(uarg)
    print(afun, bfun)

if __name__ == '__main__':
    test_arg()