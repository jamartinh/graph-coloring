# -*- coding: utf-8 -*-
"""
Created on Sat Apr 06 10:09:21 2013

@author: jamh
"""

import numpy as np
from numba import double
from numba.decorators import jit

@jit(arg_types=[double[:,:], double[:,:]])
def pairwise_numba(X, D):
    M = X.shape[0]
    N = X.shape[1]
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
