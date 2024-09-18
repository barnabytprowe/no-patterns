"""
oos_rss_polynomials_2d.py
=========================

Analysis of out-of-sample (OOS) residual sums of squares (RSS) of regression
predictions from fitting_polynomials_2d.py, described in the paper "No patterns
in regression residuals."
"""

import os
import functools
import multiprocessing
import pickle

import numpy as np

import fitting_polynomials_2d
from further_analysis_polynomials_2d import DEGREES, DEGREE_STRS


# Parameters
# ==========

# LaTeX display strings for supported NRUNS options
NRUNS_STRS = {
    300: r"$300$",
    1000: r"$10^3$",
    10000: r"$10^4$",
    100000: r"$10^5$",
    1000000: r"$10^6$",
}

# Number of simulated regression datasets
NRUNS = 1000
if NRUNS not in NRUNS_STRS:
    raise ValueError(f"User parameter NRUNS must be one of {set(NRUNS_STRS.keys())}")

# Number of cores to use in multiprocessing the regresssion - I find that on modern python
# environments a number rather less than the number of actual cores on your machine (6 for my
# laptop) works best, perhaps due to some under-the-hood parallelization
NCORES = 2

# Gather module scope degrees into an array for convenience, use values from fitting_polynomials_2d
_dvals = (
    fitting_polynomials_2d.fit_degree_lo,
    fitting_polynomials_2d.fit_degree_true,
    fitting_polynomials_2d.fit_degree_hi,
    fitting_polynomials_2d.fit_degree_vhi,
)
DEGREE_VALS = {
    _d: _dval for _d, _dv in zip(DEGREES, _dvals)
}
DEGREE_TITLES = {
    _d: DEGREE_STRS[_d].title()+" (G="+str(_dval)+")" for _d, _dv in zip(DEGREES, _dvals)
}


# Functions
# =========

def _fit_predict(data_flat, design_matrix=None, nx=fitting_polynomials_2d.nx, order="C"):
    """Perform regression on input fitting_polynomial_2d.py-style dataset using
    input features, returning regression prediction
    """
    coeffs = np.linalg.lstsq(design_matrix, data_flat, rcond=None)[0].T
    return design_matrix.dot(coeffs).reshape((nx, nx), order=order)


def build_regression_sample(
    rng, degree_vals=DEGREE_VALS, nx=fitting_polynomials_2d.nx, nruns=NRUNS
):
    """Run full large sample analysis and return results in a dictionary"""
    output = {}
    feature_labels = {_d: f"features_{_d}" for _d in degree_vals}

    # Prepare two independent variables on a grid
    xvals = np.linspace(x0x1_min, x0x1_max, num=nx, endpoint=True)
    x0, x1 = np.meshgrid(xvals, xvals)
    x0 = x0.flatten(order="C")
    x1 = x1.flatten(order="C")

    # Design matrices
    for _d in degree_vals:

        output[feature_labels[_d]] = fitting_polynomials_2d.chebyshev_design_matrix(
            x0, x1, degree=degree_values[_d])

    # Ideal model coefficients and corresponding images on the coordinate grid
    print("Generating ideal model coefficients")
    output["ctrue"] = rng.normal(
        loc=0., scale=coeff_signal_to_noise, size=(nruns, output["features_true"].shape[-1]))
    output["ztrue"] = (
        np.matmul(output["features_true"], output["ctrue"].T).T).reshape((nruns, nx, nx), order="C")

    # Generate the errors we will add to create simulated data
    print("Generating data errors")
    output["errors"] = rng.normal(loc=0., scale=noise_sigma, size=(nruns, nx, nx))
    output["zdata"] = output["ztrue"] + output["errors"]

    oshape = output["zdata"].shape
    # Perform too low, matching, too high, and very much too high degree regressions on data
    zdata_flat = output["zdata"].reshape(oshape[0], oshape[1] * oshape[2], order="C")
    output["predictions"] = {}
    for _d in degree_vals:

        _design_matrix = output[feature_labels[_d]]
        _pfunc = functools.partial(_fit_predict, design_matrix=_design_matrix, nx=nx, order="C")
        print(f"Regressing {nruns} {_d} runs using {NCORES} cores")
        with multiprocessing.Pool(NCORES) as p:
            output["predictions"][_d] = np.asarray(
                p.map(_pfunc, [_zf for _zf in zdata_flat]), dtype=float)

    # Generate a new set of errors from which to calculate the Mean Squared Error of Prediction
    print("Generating new data errors")
    output["errors_new"] = rng.normal(loc=0., scale=noise_sigma, size=(nruns, nx, nx))
    output["zdata_new"] = output["ztrue"] + output["errors_new"]
    return output
