"""
symmetric_toeplitz_entropy.py
=============================
Basic functions and charts for exploring the differential entropy of
multivariate Gaussian distributions with Toeplitz and Circulant covariance
matrices.
"""

import numpy as np
import scipy.linalg


def exponential_decay(ndim, decay_factor=0., circular=False):
    """Returns 1D float array of length ndim containing values that decrease by
    a constant decay_factor with each increasing index i, where array[0] = 1,
    with optional circular symmetry.
    """
    if not circular:
        
        if decay_factor > 0.:
            return (1. / decay_factor) * np.full(ndim, decay_factor, dtype=float).cumprod()

        else:
            _rarr = np.zeros(ndim, dtype=float)
            _rarr[0] = 1.
            return _rarr

    else:
        _rarr = np.zeros(ndim, dtype=float)
        _first_half = lagged_correlation(1 + ndim//2, decay_factor=decay_factor, circular=False)
        _rarr[:(1 + ndim//2)] = _first_half
        _rarr[(1 + ndim//2):] = _first_half[1:(ndim + 1)//2][::-1]
        return _rarr
