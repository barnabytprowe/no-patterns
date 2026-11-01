"""
no_patterns/fitting/large_sample_polynomials_2d.py
==================================================

Analysis of a large sample of polynomial regressions in two dimensions,
described in the paper "No patterns in regression residuals."
"""

import functools
import multiprocessing
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import sklearn.preprocessing

import polynomials_2d
from polynomials_2d import nx, x0x1_min, x0x1_max, coeff_signal_to_noise, noise_sigma
from further_analysis_polynomials_2d import DEGREES, DEGREE_STRS


# Parameters
# ==========

# LaTeX display strings for 10^x NRUNS
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
# laptop) works best, I think due to some under-the-hood parallelization
NCORES = 2


# Gather module scope degrees into an array for convenience, use values from fitting.polynomials_2d
degree_values = (
    polynomials_2d.fit_degree_lo,
    polynomials_2d.fit_degree_true,
    polynomials_2d.fit_degree_hi,
    polynomials_2d.fit_degree_vhi,
)
degree_titles = {
    _d: DEGREE_STRS[_d].title()+" (G="+str(_dv)+")" for _d, _dv in zip(DEGREES, degree_values)}


# Functions
# =========

def _fit_predict(data_flat, design_matrix=None):
    """Perform regression on input fitting/polynomial_2d.py-style dataset using
    input features, returning regression prediction
    """
    coeffs = np.linalg.lstsq(design_matrix, data_flat, rcond=None)[0].T
    return design_matrix.dot(coeffs).reshape((nx, nx), order="C")


def run_large_sample_analysis(rng):
    """Run full large sample analysis and return results in a dictionary"""
    output = {}
    feature_labels = [f"features_{_d}" for _d in DEGREES]

    # Prepare two independent variables on a grid
    xvals = np.linspace(x0x1_min, x0x1_max, num=nx, endpoint=True)
    x0, x1 = np.meshgrid(xvals, xvals)
    x0 = x0.flatten(order="C")
    x1 = x1.flatten(order="C")

    # Design matrices
    for _id, _flab in enumerate(feature_labels):

        output[_flab] = fitting_polynomials_2d.chebyshev_design_matrix(
            x0, x1, degree=degree_values[_id])

    # Ideal model coefficients and corresponding images on the coordinate grid
    print("Generating ideal model coefficients")
    output["ctrue"] = rng.normal(
        loc=0., scale=coeff_signal_to_noise, size=(NRUNS, output["features_true"].shape[-1]))
    output["ztrue"] = (
        np.matmul(output["features_true"], output["ctrue"].T).T).reshape((NRUNS, nx, nx), order="C")

    # Generate the errors we will add to create simulated data
    print("Generating data errors")
    output["errors"] = rng.normal(loc=0., scale=noise_sigma, size=(NRUNS, nx, nx))
    output["zdata"] = output["ztrue"] + output["errors"]

    oshape = output["zdata"].shape
    # Perform too low, matching, too high, and very much too high degree regressions on data
    zdata_flat = output["zdata"].reshape(oshape[0], oshape[1] * oshape[2], order="C")
    output["predictions"] = {}
    for _dstr, _design_matrix in zip(DEGREES, [output[_flab] for _flab in feature_labels]):

        _pfunc = functools.partial(_fit_predict, design_matrix=_design_matrix)
        print(f"Regressing {NRUNS} {_dstr} runs using {NCORES} cores")
        with multiprocessing.Pool(NCORES) as p:
            output["predictions"][_dstr] = np.asarray(
                p.map(_pfunc, [_zf for _zf in zdata_flat]), dtype=float)

    # Generate a new set of errors from which to calculate the Mean Squared Error of Prediction
    print("Generating new data errors")
    output["errors_new"] = rng.normal(loc=0., scale=noise_sigma, size=(NRUNS, nx, nx))
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
        results = run_large_sample_analysis(rng)
        print(f"Saving results to {filename}")
        with open(filename, "wb") as fout:
            pickle.dump(results, fout)

    print("Calculating residuals")
    residuals = {_d: results["zdata"] - results["predictions"][_d] for _d in DEGREES}
    print("Calculating cross validation residuals")
    cross_validation = {_d: results["zdata_new"] - results["predictions"][_d] for _d in DEGREES}
    print("Calculating ideal discrepancies")
    ideal_discrepancy = {_d: results["predictions"][_d] - results["ztrue"] for _d in DEGREES}

    print("Calculating summary statistics")
    # Calculate RSS / N, mean square cross validation "error", and mean square ideal discrepancy
    # per regression for each degree
    rssn_all = pd.DataFrame({_d: (residuals[_d]**2).mean(axis=(-2, -1)) for _d in DEGREES})
    xval_all = pd.DataFrame({_d: (cross_validation[_d]**2).mean(axis=(-2, -1)) for _d in DEGREES})
    msid_all = pd.DataFrame({_d: (ideal_discrepancy[_d]**2).mean(axis=(-2, -1)) for _d in DEGREES})

    # Calculate the overfitting parameters
    psi_all = pd.DataFrame(
        {
            _d: (
                (ideal_discrepancy[_d]**2).sum(axis=(-2, -1))
            ) / (results["errors"]**2).sum(axis=(-2, -1))
            for _d in DEGREES
        }
    )
    psi_stats = psi_all.describe()
    psi_stats.columns = ("Low degree", "Matching degree", "High degree", "Very high degree")
    psi_stats.index = (
        "Count", "Mean", "Standard deviation", "Minimum", r"$25%$", r"$50%$", r"$75%$", "Maximum")
    psi_stats.loc["Count"] = NRUNS_STRS[NRUNS]
    psi_styler = psi_stats.style
    psi_styler.format(precision=3)
    psi_styler.format(subset=("Count", psi_stats.describe().columns), precision=0)
    print()
    print("psi:")
    print(psi_styler.to_latex())

    omega_all = pd.DataFrame(
        {
            _d: -1. + 2. * (
                (ideal_discrepancy[_d] * results["errors"]).sum(axis=(-2, -1))
            ) / (ideal_discrepancy[_d]**2).sum(axis=(-2, -1))
            for _d in DEGREES
        }
    )

    omega_stats = omega_all.describe()
    omega_stats.columns = psi_stats.columns
    omega_stats.index = psi_stats.index
    omega_stats.loc["Count"] = NRUNS_STRS[NRUNS]
    #omega_stats.loc["Standard deviation", ["Overspecified", "Highly overspecified"]] = (
    #    [r"$1 \times 10^{-11}$", r"$4 \times 10^{-11}$"])
    omega_styler = omega_stats.style
    omega_styler.format(precision=3)
    omega_styler.format(subset=("Count", omega_stats.describe().columns), precision=0)
    #omega_styler.format(
    #    subset=("Standard deviation", ["Overspecified", "Highly overspecified"]), precision=13)
    print()
    print("omega:")
    print(omega_styler.to_latex())

    1/0

    rssn_mean = pd.Series(
        {_d: rssn_all[_d].mean() for _d in DEGREES}, name="RSS / N")
    xval_mean = pd.Series(
        {_d: xval_all[_d].mean() for _d in DEGREES}, name="Mean square cross validation error")
    msid_mean = pd.Series(
        {_d: msid_all[_d].mean() for _d in DEGREES}, name="Mean square ideal discrepancy")
    psi_mean = pd.Series({_d: psi_all[_d].mean() for _d in DEGREES}, name=r"Mean $\psi$")
    omega_mean = pd.Series({_d: omega_all[_d].mean() for _d in DEGREES}, name=r"Mean $\omega$")

    mean_results = pd.concat([rssn_mean, xval_mean, msid_mean], axis=1)
    mean_results.index = degrees

    # MSR violin charts
    print("Plotting RSS / N")
    fig, ax = plt.subplots(figsize=fitting_polynomials_2d.FIGSIZE)
    ax.axhline(0., color="k", ls=":")
    sns.violinplot(
        np.log10(msr_all).rename(columns=degree_titles),
        palette="Greys",
        scale="area",
        cut=0.,
        ax=ax,
    )
    ax.set_ylabel(r"$\log_{10}\left[ {\rm RSS} \, / \, N \right]$")
    ax.set_xlabel("Degree")
    ax.set_title(
        "Distribution of "
        r"$\log_{10}\left[ {\rm RSS} \, / \, N \right] = \log_{10}\left[ \frac{1}{N} \sum_n r^2_n \right]$"
        " with degree"
    )
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
    ax.yaxis.grid(which="both")
    ax.xaxis.grid(which="major")
    ax.set_ylim((-3.6, 1.4))
    plt.tight_layout()
    outfile = os.path.join(fitting_polynomials_2d.PROJDIR, f"msr_poly2d_n{NRUNS}.pdf")
    fig.savefig(outfile)
    plt.show()

    import ipdb; ipdb.set_trace()

    # MSEP violin charts
    print("Plotting MSEP")
    fig, ax = plt.subplots(figsize=fitting_polynomials_2d.FIGSIZE)
    ax.axhline(0., color="k", ls=":")
    sns.violinplot(
        np.log10(msep_all).rename(columns=degree_titles),
        palette="Greys",
        scale="area",
        cut=0.,
        ax=ax,
    )
    ax.set_ylabel(r"$\log_{10}{\rm MSEP}$")
    ax.set_xlabel("Degree")
    ax.set_title("Distribution of "+r"$\log_{10}{\rm MSEP}$"+" with polynomial degree")
    ax.grid()
    ax.set_ylim((-0.89, 1.35))
    plt.tight_layout()
    outfile = os.path.join(fitting_polynomials_2d.PROJDIR, f"msep_poly2d_n{NRUNS}.pdf")
    fig.savefig(outfile)
    plt.show()

    # MSIE violin charts
    print("Plotting MSIE")
    fig, ax = plt.subplots(figsize=fitting_polynomials_2d.FIGSIZE)
    ax.axhline(0., color="k", ls=":")
    sns.violinplot(
        np.log10(msie_all).rename(columns=degree_titles),
        palette="Greys",
        scale="area",
        cut=0.,
        ax=ax,
    )
    ax.set_ylabel(r"$\log_{10}{\rm MSIE}$")
    ax.set_xlabel("Degree")
    ax.set_title("Distribution of "+r"$\log_{10}{\rm MSIE}$"+" with polynomial degree")
    ax.grid()
    ax.set_ylim((-0.89, 1.35))
    plt.tight_layout()
    outfile = os.path.join(fitting_polynomials_2d.PROJDIR, f"msie_poly2d_n{NRUNS}.pdf")
    fig.savefig(outfile)
    plt.show()

    # Line chart of mean of MSR, MSEP, MSIE versus regression model set degree
    print("Plotting means")
    ax = mean_results.plot(
        logy=True, marker="s", color=[mpl.colormaps["gray"](_x) for _x in (.75, .4, .05)])
    ax.axhline(1., color="k", ls=":")
    ax.grid(which="both")
    ax.set_xlabel("Degree")
    ax.set_title("Sample mean of "+NRUNS_STRS[NRUNS]+" regressions")
    plt.tight_layout()
    outfile = os.path.join(
        fitting_polynomials_2d.PROJDIR, f"mean_msr_msep_msie_poly2d_n{NRUNS}.pdf")
    plt.savefig(outfile)
    plt.show()
