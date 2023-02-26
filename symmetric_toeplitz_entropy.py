"""
symmetric_toeplitz_entropy.py
=============================
Basic functions and charts for exploring the differential entropy of
multivariate Gaussian distributions with Toeplitz and Circulant covariance
matrices.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def exponential_decay(ndim, decay_factor=0., circular=False, oscillatory=False):
    """Returns 1D float array of length ndim containing values that decrease by
    a constant decay_factor with each increasing index i, where array[0] = 1,
    with optional circular symmetry.
    """
    _rarr = np.zeros(ndim, dtype=float)
    if not circular:
        _rarr[0] = 1.
        if decay_factor > 0.:
            _rarr[1:] = np.full(ndim - 1, decay_factor, dtype=float).cumprod()
        if oscillatory:
            _rarr *= (-1.)**(np.arange(ndim) % 2)

    else:
        _first_half = exponential_decay(1 + ndim//2, decay_factor=decay_factor, circular=False)
        _rarr[:(1 + ndim//2)] = _first_half
        _rarr[(1 + ndim//2):] = _first_half[1:(1 + ndim)//2][::-1]

    return _rarr
