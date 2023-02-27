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


# Params for main script & plotting
# ---------------------------------
NDIM = 100
DECAY_FACTORS = np.linspace(0., 0.95, num=50)


# Functions
# ---------

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

def exponential_toeplitz_corr(ndim, decay_factor=0., circulant=False, oscillatory=False):
    """Returns Toeplitz correlation matrix as a 2D float array of shape
    (ndim, ndim) with the specified exponential decay in band structure.
    """
    if circulant:
        return scipy.linalg.circulant(
            exponential_decay(
                ndim, decay_factor=decay_factor, circular=True, oscillatory=oscillatory)
        )

    else:
        return scipy.linalg.toeplitz(
            exponential_decay(
                ndim, decay_factor=decay_factor, circular=False, oscillatory=oscillatory)
        )


if __name__ == "__main__":

    # Calculate full Toeplitz and circulant approximation determinants via their eigenvalue products
    dets_toeplitz = np.asarray(
        [
            np.product(
                np.linalg.eigvalsh(
                    exponential_toeplitz_corr(NDIM, _dfac, circulant=False, oscillatory=False)
                )
            )
            for _dfac in DECAY_FACTORS
        ],
        dtype=float,
    )
    dets_circulant = np.asarray(
        [
            np.product(
                np.linalg.eigvalsh(
                    exponential_toeplitz_corr(NDIM, _dfac, circulant=True, oscillatory=False)
                )
            )
            for _dfac in DECAY_FACTORS
        ],
        dtype=float,
    )

    # Plot versus decay factors
    plt.plot(DECAY_FACTORS, dets_toeplitz, "k-")
    plt.plot(DECAY_FACTORS, dets_circulant, "k--", label="Circulant approximation")
    plt.yscale("log")
    plt.yticks(10.**(-10. * np.arange(11)[::-1]))
    plt.ylim(1.e-103, 1000.)
    plt.minorticks_on()
    plt.grid(True, which="both")
    plt.title(r"$\det{\mathbf{P}}$")
    plt.xlabel("Decay factor")
    plt.legend()
    plt.tight_layout()
    plt.show()
