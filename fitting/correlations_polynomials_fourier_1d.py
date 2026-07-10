"""
no_patterns/fitting/correlations_polynomials_fourier_1d.py
==========================================================

Performs a large sample analysis of average correlations in residuals from
regression of in one dimension, described in the paper "No patterns in
regression residuals," with Chebyshev polynomial and Fourier series models.
Correlation functions and correlation matrices for residuals, for data ordered
along the abcissa, are output.

Uses the regression simulation parameters and functions defined in
fitting/polynomials_fourier_1d.py
"""

import functools
import multiprocessing
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import polynomials_fourier_1d
from polynomials_fourier_1d import FIT_DEGREES, XARRS, NOISE_SIGMA, COEFF_SIGNAL_TO_NOISE


# Parameters
# ==========

# Number of simulated regression, out-of-sample datasets
NRUNS = 100000

# Number of cores to use in multiprocessing the regresssion - I find that on modern python
# environments a number rather fewer than the number of actual cores on your machine (6 for my
# laptop) works best, perhaps due to some under-the-hood parallelization
NCORES = 2

# Pickle cache file location setup
PKLDIR = os.path.join(".", "pickles")
if not os.path.isdir(PKLDIR):
    os.mkdir(PKLDIR)
PICKLE_CACHE = os.path.join(PKLDIR, f"correlations_regressions_1d_n{NRUNS}.pkl")
CLOBBER = False  # overwrite any existing pickle cache


def _fit_predict(data, design_matrix):
    """Perform OLS regression on input data using the input design_matrix,
    returning predictions
    """
    coeffs = np.linalg.lstsq(design_matrix, data, rcond=None)[0].T
    return design_matrix.dot(coeffs)


def build_regression_sample(
    rng,
    families=("cheb", "sinu"),
    nruns=NRUNS,
    fit_degrees=FIT_DEGREES,
    xarrs=XARRS,
    noise_sigma=NOISE_SIGMA,
    coeff_signal_to_noise=COEFF_SIGNAL_TO_NOISE,
):
    """Run full large sample analysis and return results in a dictionary, keyed
    by the curve family keys in the second layer of the fit_degrees
    [default: ("cheb", "sinu")]
    """
    design_matrices = polynomials_fourier_1d.features(xarrs=xarrs, fit_degrees=fit_degrees)
    for _matrices in design_matrices.values():
        assert set(_matrices.keys()) == set(families)
    output = {_fam: {} for _fam in families}

    # Ideal model coefficients and corresponding images on the coordinate grid
    for _fam in families:
        print(f"Generating {_fam} ideal model coefficients")
        output[_fam]["ctrue"] = rng.normal(
            loc=0.,
            scale=coeff_signal_to_noise * noise_sigma,
            size=(nruns, design_matrices["true"][_fam].shape[-1]),
        )
        output[_fam]["ytrue"] = (
            np.matmul(design_matrices["true"][_fam], output[_fam]["ctrue"].T).T
        ).reshape((nruns, len(xarrs[_fam])), order="C")

        # Generate the errors we will add to create simulated data
        print(f"Generating {_fam} errors, data")
        output[_fam]["errors"] = rng.normal(
            loc=0., scale=noise_sigma, size=(nruns, len(xarrs[_fam]))
        )
        output[_fam]["ydata"] = output[_fam]["ytrue"] + output[_fam]["errors"]

        # Perform too low, matching, too high, and very much too high degree regressions on data
        output[_fam]["predictions"] = {}
        output[_fam]["residuals"] = {}
        for _d in fit_degrees:
            _design_matrix = design_matrices[_d][_fam]
            _pfunc = functools.partial(_fit_predict, design_matrix=_design_matrix)
            print(f"Regressing {nruns} {_d} {_fam} runs using {NCORES=}")
            with multiprocessing.Pool(NCORES) as p:
                output[_fam]["predictions"][_d] = np.asarray(
                    p.map(_pfunc, [_zf for _zf in output[_fam]["ydata"]]), dtype=float
                )
            output[_fam]["residuals"][_d] = output[_fam]["ydata"] - output[_fam]["predictions"][_d]

    return output


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()

    if not CLOBBER and os.path.isfile(PICKLE_CACHE):
        print(f"Loading from {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "rb") as funit:
            regressions = pickle.load(funit)
    else:
        t0 = time.time()
        regressions = build_regression_sample(
            rng=rng,
            fit_degrees=FIT_DEGREES,
            xarrs=XARRS,
            noise_sigma=NOISE_SIGMA,
            coeff_signal_to_noise=COEFF_SIGNAL_TO_NOISE,
        )
        t1 = time.time()
        print(f"Wall time: {(t1 - t0):.2f}s")
        print(f"Saving results to {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "wb") as fout:
            pickle.dump(regressions, fout)
