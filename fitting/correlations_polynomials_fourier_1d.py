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
import numpy.polynomial.chebyshev
import pandas as pd

import polynomials_fourier_1d


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
PICKLE_CACHE = os.path.join(PKLDIR, f"correlations_n{NRUNS}.pkl")
CLOBBER = False  # overwrite any existing pickle cache


def _fit_predict(data, design_matrix=None):
    """Perform OLS regression on input data using design_matrix, returning prediction
    """
    coeffs = np.linalg.lstsq(design_matrix, data, rcond=None)[0].T
    return design_matrix.dot(coeffs)


def build_regression_sample(rng, nruns=NRUNS, families=("cheb", "sinu")):
    """Run full large sample analysis and return results in a dictionary, keyed
    by curve family ("cheb", "sinu")
    """
    design_matrices = polynomials_fourier_1d.features()
    output = {_family: {} for _family in families}

    # Ideal model coefficients and corresponding images on the coordinate grid
    for _family in families:
        print(f"Generating {_family} ideal model coefficients")
        output[_family]["ctrue"] = rng.normal(
            loc=0.,
            scale=polynomials_fourier_1d.coeff_signal_to_noise * polynomials_fourier_1d.noise_sigma,
            size=(nruns, design_matrices["true"][_family].shape[-1]),
        )
        output[_family]["ztrue"] = (
            np.matmul(design_matrices["true"][_family], output[_family]["ctrue"].T).T
        ).reshape((nruns, polynomials_fourier_1d.nx), order="C")

        # Generate the errors we will add to create simulated data
        print(f"Generating {_family} errors, data")
        output[_family]["errors"] = rng.normal(
            loc=0.,
            scale=polynomials_fourier_1d.noise_sigma,
            size=(nruns, polynomials_fourier_1d.nx),
        )
        output[_family]["zdata"] = output[_family]["ztrue"] + output[_family]["errors"]

        # Perform too low, matching, too high, and very much too high degree regressions on data
        output[_family]["predictions"] = {}
        for _d in polynomials_fourier_1d.fit_degrees:
            _design_matrix = design_matrices[_d][_family]
            _pfunc = functools.partial(_fit_predict, design_matrix=_design_matrix)
            print(f"Regressing {nruns} {_d} {_family} runs using multiprocessing with {NCORES=}")
            with multiprocessing.Pool(NCORES) as p:
                output[_family]["predictions"][_d] = np.asarray(
                    p.map(_pfunc, [_zf for _zf in output[_family]["zdata"]]), dtype=float
                )

    return output


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()

    if not CLOBBER and os.path.isfile(PICKLE_CACHE):
        print(f"Loading from {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "rb") as funit:
            results = pickle.load(funit)
    else:
        t0 = time.time()
        results = build_regression_sample(rng=rng, nruns=NRUNS, families=("cheb", "sinu"))
        t1 = time.time()
        print(f"Wall time: {(t1 - t0):.2f}s")
        print(f"Saving results to {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "wb") as fout:
            pickle.dump(results, fout)
