"""
oos_rss_polynomials_2d.py
=========================

Analysis of out-of-sample (OOS) residual sums of squares (RSS) of regression
predictions from fitting_polynomials_2d.py, described in the paper "No patterns
in regression residuals."
"""

import functools
import multiprocessing
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fitting_polynomials_2d
from further_analysis_polynomials_2d import DEGREES, DEGREE_STRS


# Parameters
# ==========

# Number of simulated regression, out-of-sample datasets
NRUNS = 1000

# Number of cores to use in multiprocessing the regresssion - I find that on modern python
# environments a number rather fewer than the number of actual cores on your machine (6 for my
# laptop) works best, perhaps due to some under-the-hood parallelization
NCORES = 1

# Gather module scope degrees into an array for convenience, use values from fitting_polynomials_2d
_dvals = (
    fitting_polynomials_2d.fit_degree_lo,
    fitting_polynomials_2d.fit_degree_true,
    fitting_polynomials_2d.fit_degree_hi,
    fitting_polynomials_2d.fit_degree_vhi,
)
DEGREE_VALS = {
    _d: _dval for _d, _dval in zip(DEGREES, _dvals)
}
DEGREE_TITLES = {
    _d: DEGREE_STRS[_d].title()+" (G="+str(_dval)+")" for _d, _dval in DEGREE_VALS.items()
}

# Pickle cache file location
PICKLE_CACHE = f"oos_rss_n{NRUNS}.pkl"
CLOBBER = False  # overwrite any existing pickle cache

# Figure size
FIGSIZE = (6, 7)

# Show plot figures on screen?
SHOW_PLOTS = True

# Output plot files
INS_OUTFILE = os.path.join(fitting_polynomials_2d.PROJDIR, f"in-sample_rss_n{NRUNS}.pdf")
OOS_OUTFILE = os.path.join(fitting_polynomials_2d.PROJDIR, f"out-of-sample_rss_n{NRUNS}.pdf")
MR_OUTFILE = os.path.join(fitting_polynomials_2d.PROJDIR, f"matched-relative_rss_n{NRUNS}.pdf")


# Functions
# =========

def _fit_predict(data_flat, design_matrix=None, nx=fitting_polynomials_2d.nx, order="C"):
    """Perform regression on input fitting_polynomial_2d.py-style dataset using
    input features, returning regression prediction
    """
    coeffs = np.linalg.lstsq(design_matrix, data_flat, rcond=None)[0].T
    return design_matrix.dot(coeffs).reshape((nx, nx), order=order)


def build_regression_sample(
    rng,
    degree_vals=DEGREE_VALS,
    nx=fitting_polynomials_2d.nx,
    nruns=NRUNS,
    x0x1_min=fitting_polynomials_2d.x0x1_min,
    x0x1_max=fitting_polynomials_2d.x0x1_max,
    coeff_signal_to_noise=fitting_polynomials_2d.coeff_signal_to_noise,
    noise_sigma=fitting_polynomials_2d.noise_sigma,
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
            x0, x1, degree=degree_vals[_d])

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
        print(f"Regressing {nruns} {_d} runs using multiprocessing with {NCORES=}")
        with multiprocessing.Pool(NCORES) as p:
            output["predictions"][_d] = np.asarray(
                p.map(_pfunc, [_zf for _zf in zdata_flat]), dtype=float)

    # Generate a new set of errors from which to calculate the Mean Squared Error of Prediction
    print("Generating new data errors")
    output["errors_new"] = rng.normal(loc=0., scale=noise_sigma, size=(nruns, nx, nx))
    output["zdata_new"] = output["ztrue"] + output["errors_new"]
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
        results = build_regression_sample(
            rng,
            degree_vals=DEGREE_VALS,
            nx=fitting_polynomials_2d.nx,
            nruns=NRUNS,
            x0x1_min=fitting_polynomials_2d.x0x1_min,
            x0x1_max=fitting_polynomials_2d.x0x1_max,
            coeff_signal_to_noise=fitting_polynomials_2d.coeff_signal_to_noise,
            noise_sigma=fitting_polynomials_2d.noise_sigma,
        )
        t1 = time.time()
        print(f"Wall time: {(t1 - t0):.2f}s")
        print(f"Saving results to {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "wb") as fout:
            pickle.dump(results, fout)

    print("Calculating in-sample, out-of-sample residuals and summary statistics")
    ins_residuals = {_d: results["zdata"] - results["predictions"][_d] for _d in DEGREES}
    oos_residuals = {_d: results["zdata_new"] - results["predictions"][_d] for _d in DEGREES}
    ins_rssn = pd.DataFrame({_d: (ins_residuals[_d]**2).mean(axis=(-2, -1)) for _d in DEGREES})
    oos_rssn = pd.DataFrame({_d: (oos_residuals[_d]**2).mean(axis=(-2, -1)) for _d in DEGREES})
    relative_oos_rss = (oos_rssn.T / oos_rssn["true"].T).T

    # Plot in-sample RSS for simulation runs
    fig = plt.figure(figsize=FIGSIZE)
    plt.boxplot(ins_rssn, whis=[0, 100], sym="x")
    plt.yscale("log")
    plt.grid(which="both")
    plt.ylabel(r"RSS$/N$", size="large")
    plt.xticks([1, 2, 3, 4], DEGREE_TITLES.values())
    plt.title(r"In-sample RSS$/N$")
    plt.tight_layout()
    print(f"Saving to {INS_OUTFILE}")
    plt.savefig(INS_OUTFILE)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=FIGSIZE)
    plt.boxplot(oos_rssn, whis=[0, 100], sym="x")
    plt.yscale("log")
    plt.grid(which="both")
    plt.ylabel(r"RSS$/N$", size="large")
    plt.xticks([1, 2, 3, 4], DEGREE_TITLES.values())
    plt.title(r"Out-of-sample RSS$/N$")
    plt.tight_layout()
    print(f"Saving to {OOS_OUTFILE}")
    plt.savefig(OOS_OUTFILE)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=FIGSIZE)
    plt.boxplot(relative_oos_rss[["lo", "hi", "vhi"]], whis=[0, 100], sym="x")
    plt.yscale("log")
    plt.grid(which="both")
    plt.ylabel(r"Out-of-sample RSS$_G$ / RSS$_{G^*}$", size="large")
    plt.ylim((0.85, 14.))
    plt.axhline(1., color="k", lw="1.2", ls="--")
    plt.xticks([1, 2, 3], np.asarray(list(DEGREE_TITLES.values()))[[True, False, True, True]])
    plt.title(r"Matched-relative out-of-sample RSS")
    plt.tight_layout()
    print(f"Saving to {MR_OUTFILE}")
    plt.savefig(MR_OUTFILE)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
