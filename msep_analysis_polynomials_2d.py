"""
msep_analysis_polynomials_2d.py
===============================

Mean Squared Errors of Prediction analysis.
"""

import os
import functools
import multiprocessing
import pickle

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing

import fitting_polynomials_2d


# Parameters
# ==========

# Number of simulated regression datasets
NRUNS = 10000

# Number of cores to use in multiprocessing the regresssion - I find that on modern python
# environments a number rather less than the number of actual cores on your machine (6 for my
# laptop) works best, I think due to some under-the-hood parallelization
NCORES = 2


# Module scope linear regression class
regr = sklearn.linear_model.LinearRegression()  # uses LAPACK via np leastsq under the hood


# Functions
# =========

def _fit_predict(_dataset, _features=None):
    """Perform regression on input fitting_polynomial_2d.py-style dataset using
    input features, returning regression prediction
    """
    _dataset = _dataset.flatten(order="C")
    regr.fit(_features, _dataset)
    return regr.predict(_features).reshape(
        (fitting_polynomials_2d.nx, fitting_polynomials_2d.nx), order="C")


def run_msep(rng):
    """Run full msep analysis and return results in a dictionary"""
    output = {}

    # Design matrices
    output["features_lo"] = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_lo,
        order="C",
    )
    output["features_true"] = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_true,
        order="C",
    )
    output["features_hi"] = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_hi,
        order="C",
    )
    output["features_vhi"] = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_vhi,
        order="C",
    )

    # Ideal model coefficients and corresponding images on the coordinate grid
    print("Generating ideal model coefficients")
    output["ctrue"] = rng.normal(
        loc=0.,
        scale=fitting_polynomials_2d.coeff_signal_to_noise,
        size=(NRUNS, output["features_true"].shape[-1]),
    )
    output["ztrue"] = (
        np.matmul(output["features_true"], output["ctrue"].T).T
    ).reshape(NRUNS, fitting_polynomials_2d.nx, fitting_polynomials_2d.nx, order="C")

    # Generate the errors we will add to create simulated data
    print("Generating data errors")
    output["errors"] = rng.normal(
        loc=0.,
        scale=fitting_polynomials_2d.noise_sigma,
        size=(NRUNS, fitting_polynomials_2d.nx, fitting_polynomials_2d.nx),
    )
    output["zdata"] = output["ztrue"] + output["errors"]

    # Perform too low, matching, too high, and very much too high degree regressions on data
    output["predictions"] = {}
    for _key, features in zip(
                ("lo", "true", "hi", "vhi"),
                (
                    output["features_lo"],
                    output["features_true"],
                    output["features_hi"],
                    output["features_vhi"]
                ),
            ):

        _pfunc = functools.partial(_fit_predict, _features=features)
        print(f"Regressing {NRUNS} {_key} runs using {NCORES} cores")
        with multiprocessing.Pool(NCORES) as p:
            output["predictions"][_key] = np.asarray(
                p.map(_pfunc, [_zf for _zf in output["zdata"]]), dtype=float)

    # Generate a new set of errors from which to calculate the Mean Squared Error of Prediction
    print("Generating new data errors")
    output["errors_new"] = rng.normal(
        loc=0.,
        scale=fitting_polynomials_2d.noise_sigma,
        size=(NRUNS, fitting_polynomials_2d.nx, fitting_polynomials_2d.nx),
    )
    output["zdata_new"] = output["ztrue"] + output["errors_new"]
    return output


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()

    filename = f"msep_n{NRUNS}.pkl"
    if os.path.isfile(filename):
        print(f"Loading from {filename}")
        with open(filename, "rb") as funit:
            results = pickle.load(funit)
    else:
        results = run_msep(rng)
        print(f"Saving results to {filename}")
        with open(filename, "wb") as fout:
            pickle.dump(results, fout)
