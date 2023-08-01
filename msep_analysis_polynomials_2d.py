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
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing

import fitting_polynomials_2d
from further_analysis_polynomials_2d import DEGREES


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

# Module scope degrees array
degrees = (
    fitting_polynomials_2d.fit_degree_lo,
    fitting_polynomials_2d.fit_degree_true,
    fitting_polynomials_2d.fit_degree_hi,
    fitting_polynomials_2d.fit_degree_vhi,
)


# Functions
# =========

def _fit_predict(_dataset, _features=None):
    """Perform regression on input fitting_polynomial_2d.py-style dataset using
    input features, returning regression prediction
    """
    _dataset = _dataset.flatten(order="C")  # having flattening and then reshaping in the inner
                                            # loop is not ideal, but will postpone optimization
                                            # until performance limited
    regr.fit(_features, _dataset)
    return regr.predict(_features).reshape(
        (fitting_polynomials_2d.nx, fitting_polynomials_2d.nx), order="C")


def run_msep(rng):
    """Run full msep analysis and return results in a dictionary"""
    output = {}
    feature_labels = [f"features_{_d}" for _d in DEGREES]

    # Design matrices
    for _id, _flab in enumerate(feature_labels):

        output[_flab] = fitting_polynomials_2d.polynomial_design_matrix(
            square_dimension=fitting_polynomials_2d.side_dim,
            n_side=fitting_polynomials_2d.nx,
            degree=degrees[_id],
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
    for _dstr, _design_matrix in zip(DEGREES, [output[_flab] for _flab in feature_labels]):

        _pfunc = functools.partial(_fit_predict, _features=_design_matrix)
        print(f"Regressing {NRUNS} {_dstr} runs using {NCORES} cores")
        with multiprocessing.Pool(NCORES) as p:
            output["predictions"][_dstr] = np.asarray(
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

    print("Calculating key stats")
    residuals = {_d: results["zdata"] - results["predictions"][_d] for _d in DEGREES}
    errors_of_prediction = {_d: results["zdata_new"] - results["predictions"][_d] for _d in DEGREES}
    ideal_error = {_d: results["predictions"][_d] - results["ztrue"] for _d in DEGREES}

    msr_all = {_d: (residuals[_d]**2).mean(axis=(-2,-1)) for _d in DEGREES}
    msep_all = {_d: (errors_of_prediction[_d]**2).mean(axis=(-2,-1)) for _d in DEGREES}
    msie_all = {_d: (ideal_error[_d]**2).mean(axis=(-2,-1)) for _d in DEGREES}

    msr = pd.Series({_d: msr_all[_d].mean() for _d in DEGREES}, name="MSR")
    msep = pd.Series({_d: msep_all[_d].mean() for _d in DEGREES}, name="MSEP")
    msie = pd.Series({_d: msie_all[_d].mean() for _d in DEGREES}, name="MSIE")

    results = pd.concat([msr, msep, msie], axis=1)
    results.index = degrees

    print("Plotting")
    ax = results.plot(logy=True)
    ax.grid()
    plt.show()
