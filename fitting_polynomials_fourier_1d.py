"""
fitting_polynomials_fourier_1d.py
=================================

Examples of regression of in one dimension, described in the paper "No patterns
in regression residuals," illustrating underspecified, correctly specified, and
overfitting scenarios.  Chebyshev polynomial of the first kind series models,
and Fourier series models, are combined with additive iid Gaussian noise and
subject to ordinary linear regression.

Saves output into a uniquely timestamped subfolder of
./plots/polynomials_fourier_1d/.
"""

import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.polynomial.chebyshev
import pandas as pd

import fitting_polynomials_2d
from fitting_polynomials_2d import PLTDIR


# Parameters
# ==========

# Number of data points
nx = 100

# Sigma of iid pixel noise
noise_sigma = 1.

# Settings for the ideal model, underspecified, overspecified and highly overspecified
# series model degrees to use as model sets for the ideal model and fitting
fit_degree = {}

# real signal (ideal model) degree in the simulations (1D polynomial / Fourier series) will also
# be used as a model set for regression
fit_degree["true"] = {"cheb": 8, "sinu": 4}

fit_degree["lo"] = {"cheb": 4, "sinu": 2}  # underspecified model sets
fit_degree["hi"] = {"cheb": 16, "sinu": 8}  # overspecified model sets
fit_degree["vhi"] = {"cheb": 32, "sinu": 16}  # added to illustrate more extreme behaviour clearly

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 1.

# Define x coordinates as linearly spaced points on the some interval, e.g. [0, 1), [-1, 1)
x = {
    "cheb": np.linspace(-1., 1., num=nx, endpoint=False),
    "sinu": np.linspace(0., 1., num=nx, endpoint=False),
}


# Plot settings
FIGSIZE = (10, 4)
FIGSIZE_RESIDUALS = (10, 1.25)
CLIM = [-2.5, 2.5]
CMAP = "Greys_r"
TITLE_SIZE = "x-large"

# Title display strings for plots
FIT_DISPLAY = {
    "lo": "Low degree",
    "true": "Matching degree",
    "hi": "High degree",
    "vhi": "Very high degree",
}
CURVE_FAMILY_DISPLAY = {"cheb": "polynomial", "sinu": "Fourier"}

# Periodogram chart settings
PERIODOGRAM_YTICKS = 10**np.linspace(-32., 4., num=10, dtype=float)
PERIODOGRAM_YLIM = 10**np.asarray([-32, 4.], dtype=float)

# Autocorrelation function chart settings
ACF_MAX_LAG = 10

# Output folder structure: project dir
PROJDIR = os.path.join(PLTDIR, "polynomials_fourier_1d")

# Output file types
OUTFILE_EXTENSIONS = (".png", ".pdf")


# Functions
# =========

def sinusoid_design_matrix(x, degree):
    """Returns the sinusoid [cosx, sinx] design matrix up to input degree"""
    sinx = np.asarray([np.sin(2. * np.pi * float(j) * x) for j in range(0, 1 + degree)]).T
    cosx = np.asarray([np.cos(2. * np.pi * float(j) * x) for j in range(0, 1 + degree)]).T
    return np.hstack([cosx, sinx])


def chebyshev_design_matrix(x, degree):
    """Returns the Chebyshev polynomial design matrix up to input degree"""
    i1n = np.eye(1 + degree)
    return np.asarray([numpy.polynomial.chebyshev.chebval(x, _row) for _row in i1n]).T


def plot_regressions(xarr, yarrs, curve_family_display, tstmp, outdir, show=True):
    """Makes and saves scatter and line plots of 1D regressions.

    Args:
        xarr:
            numpy array-like containing x coordinates shared by all arrays in
            yarrs
        yarrs:
            list of 6 array-likes containing the following values in the
            dependent variable y, in order:
            - ideal model
            - data (= ideal model + iid errors)
            - Low degree model set OLS prediction
            - Matching degree model set set OLS prediction
            - High degree model set set OLS prediction
            - Very high degree model set set OLS prediction
        curve_family_display: one of {'polynomial', 'Fourier'}
        tstmp: timestamp used in folder structure
        outdir: output folder
        show: plt.show()?
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(
        f"{curve_family_display.title()} series regression simulation", size=TITLE_SIZE)
    ax.plot(xarr, yarrs[0], color="k", ls="-", linewidth=2, label="Ideal model")
    ax.plot(xarr, yarrs[1], "k+", markersize=15, label="Data")
    ax.plot(xarr, yarrs[2], color="red", ls="--", linewidth=1, label=FIT_DISPLAY["lo"])
    ax.plot(xarr, yarrs[3], color="k", ls="-", linewidth=1, label=FIT_DISPLAY["true"])
    ax.plot(xarr, yarrs[4], color="blue", ls="-.", linewidth=1, label=FIT_DISPLAY["hi"])
    ax.plot(xarr, yarrs[5], color="purple", ls=":", linewidth=1.25, label=FIT_DISPLAY["vhi"])
    ax.set_xlabel(r"$x$")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir, f"curves_{curve_family_display.lower().replace(' ', '_')}_{tstmp}{_suffix}")
        print(f"Saving to {outfile}")
        fig.savefig(outfile)

    if show:
        plt.show()
    plt.close(fig)


def plot_residuals(residuals, fit_display, curve_family_display, tstmp, outdir, show=True):
    """Makes and saves pcolor images plots of residuals in 1D regressions.

    Args:
        residuals: np.array-like
        fit_display: display str for fit, e.g. Low degree, Matching degree etc.
        curve_family_display: one of {'polynomial', 'Fourier'}
        tstmp: timestamp used in folder structure
        outdir: output folder
        show: plt.show()?
    """
    fig = plt.figure(figsize=FIGSIZE_RESIDUALS)
    ax = fig.add_axes([0.075, 0.3, 0.855, 0.45])
    im = ax.pcolor(residuals.reshape((1, len(residuals))), cmap=CMAP, clim=CLIM)
    ax.set_yticklabels([])
    ax.set_title(
        f"{fit_display} {curve_family_display} series regression residuals", size=TITLE_SIZE)

    # See https://stackoverflow.com/a/39938019 for colormap handling
    divider = make_axes_locatable(ax)
    cax = fig.add_axes([0.945, 0.3, 0.01, 0.45])
    fig.colorbar(im, cax=cax, orientation='vertical')
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir, (
                f"residuals_{fit_display.lower().replace(' ', '_')}_"
                f"{curve_family_display.lower().replace(' ', '_')}_{tstmp}{_suffix}"
            ),
        )
        print(f"Saving to {outfile}")
        fig.savefig(outfile)
        if show:
            plt.show()

    plt.close(fig)
    return


def plot_periodograms(periodograms, nfull, curve_family_display, tstmp, outdir, show=True):
    """Makes and saves plots of error and residual periodograms from 1D
    regressions.

    Args:
        periodograms:
            list of 5 array-likes containing the following 1d periodograms (in
            order):
            - iid errors
            - Low degree model set residuals
            - Matching degree model set residuals
            - High degree model set residuals
            - Very high degree model set residuals
        nfull:
            int full size of original dataset, such that
            len(p) = nfull // 2 + 1 for each for each p in the periodograms
        curve_family_display: one of {'polynomial', 'Fourier'}
        tstmp: timestamp used in folder structure
        outdir: output folder
        show: plt.show()?
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(
        curve_family_display.title()+" series regression residual periodograms", size=TITLE_SIZE)

    ax.plot(
        np.arange(len(periodograms[0])) / nfull, periodograms[0], color="k", ls="--",
        linewidth=1, label="iid errors",
    )
    ax.plot(
        np.arange(len(periodograms[1])) / nfull, periodograms[1], color="red", ls="--",
        linewidth=1.5, label=FIT_DISPLAY["lo"],
    )
    ax.plot(
        np.arange(len(periodograms[2])) / nfull, periodograms[2], color="k", ls="-",
        linewidth=1.5, label=FIT_DISPLAY["true"],
    )
    ax.plot(
        np.arange(len(periodograms[3])) / nfull, periodograms[3], color="blue", ls="-.",
        linewidth=1.5, label=FIT_DISPLAY["hi"],
    )
    ax.plot(
        np.arange(len(periodograms[4])) / nfull, periodograms[4], color="purple", ls=":",
        linewidth=1.5, label=FIT_DISPLAY["vhi"],
    )

    ax.set_yscale("log")
    ax.set_yticks(PERIODOGRAM_YTICKS)
    ax.set_ylim(PERIODOGRAM_YLIM)
    ax.set_xlabel("Frequency")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir,
            f"periodograms_{curve_family_display.lower().replace(' ', '_')}_{tstmp}{_suffix}",
        )
        print(f"Saving to {outfile}")
        fig.savefig(outfile)

    if show:
        plt.show()
    plt.close(fig)


def plot_acfs(acfs, nfull, curve_family_display, tstmp, outdir, show=True):
    """Makes and saves plots of error and residual autocorrelation functions
    from 1D regressions.

    Args:
        acfs:
            list of 5 array-likes containing the following 1d autocorrelation
            functions (in order):
            - iid errors
            - Low degree model set residuals
            - Matching degree model set residuals
            - High degree model set residuals
            - Very high degree model set residuals
        nfull:
            int full size of original dataset, such that
            len(acf) = nfull // 2 + 1 for each for each acf in the acfs
        curve_family_display: one of {'polynomial', 'Fourier'}
        tstmp: timestamp used in folder structure
        outdir: output folder
        show: plt.show()?
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(
        curve_family_display.title()+" series regression residual autocorrelation functions",
        size=TITLE_SIZE,
    )

    offset = 0.
    ax.plot(
        np.arange(len(acfs[0])), acfs[0], color="k", ls="--", linewidth=1, label="iid errors")
    ax.plot(
        1 * offset + np.arange(len(acfs[1])), acfs[1],
        marker="o", color="red", ls="--", linewidth=1.5, label=FIT_DISPLAY["lo"],
    )
    ax.plot(
        2 * offset + np.arange(len(acfs[2])), acfs[2],
        marker="x", color="k", ls="-", linewidth=1.5, label=FIT_DISPLAY["true"],
    )
    ax.plot(
        3 * offset + np.arange(len(acfs[3])), acfs[3],
        marker="+", color="blue", ls="-.", linewidth=1.5, label=FIT_DISPLAY["hi"],
    )
    ax.plot(
        4 * offset + np.arange(len(acfs[4])), acfs[4],
        marker=".", color="purple", ls=":", linewidth=1.5, label=FIT_DISPLAY["vhi"],
    )

    ax.axhline(-2. / np.sqrt(nfull), ls=":", linewidth=1.2, color="k")
    ax.axhline(-1. / np.sqrt(nfull), ls=":", linewidth=1.2, color="k")
    ax.axhline(+0., ls="-", linewidth=1, color="k")
    ax.axhline(+1. / np.sqrt(nfull), ls=":", linewidth=1.2, color="k")
    ax.axhline(+2. / np.sqrt(nfull), ls=":", linewidth=1.2, color="k")
    ax.set_xlabel("Lag")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir,
            f"acfs_{curve_family_display.lower().replace(' ', '_')}_{tstmp}{_suffix}",
        )
        print(f"Saving to {outfile}")
        fig.savefig(outfile)

    if show:
        plt.show()
    plt.close(fig)


# Main script
# ===========

if __name__ == "__main__":

    # Current timestamp, used in I/0
    tstmp = pd.Timestamp.now().isoformat().replace(":", "")
    outdir = fitting_polynomials_2d.build_output_folder_structure(tstmp, project_dir=PROJDIR)

    # Output dict - will be pickled
    output = {}

    # Design matrices for the different model sets
    features = {
        _degree: {
            _k: _f(x=x[_k], degree=fit_degree[_degree][_k])
            for _k, _f in (("cheb", chebyshev_design_matrix), ("sinu", sinusoid_design_matrix))
        }
        for _degree in ("lo", "true", "hi", "vhi")
    }

    for _cf in ("cheb", "sinu"):  # Big outer loop over curve family

        # Build the true 1d curve coefficients
        output[f"{_cf}_coeffs_true"] = coeff_signal_to_noise * np.random.randn(
            features["true"][_cf].shape[-1])
        # Build the true 1d curves from these coefficients
        output[f"ytrue_{_cf}"] = np.matmul(features["true"][_cf], output[f"{_cf}_coeffs_true"])
        # Add random Gaussian iid errors to generate our simulation dataset y values
        output[f"e_{_cf}"] = noise_sigma * np.random.randn(nx)
        output[f"y_{_cf}"] = output[f"ytrue_{_cf}"] + output[f"e_{_cf}"]

        # Plot scatter plots of data, ideal model and predictions
        # First perform regression at different degrees to generate predictions
        for _fit in ("lo", "true", "hi", "vhi"):

            _design_matrix = features[_fit][_cf]
            _coeffs = np.linalg.lstsq(_design_matrix, output[f"y_{_cf}"], rcond=None)[0]
            _yfit = _design_matrix.dot(_coeffs.T)
            output[f"ypred_{_cf}_{_fit}"] = _yfit

        # Plot ideal model, data, and ordinary least squares regression predictions
        plot_regressions(
            xarr=x[_cf],
            yarrs=[
                output[f"ytrue_{_cf}"],  # ideal model
                output[f"y_{_cf}"],  # data
                output[f"ypred_{_cf}_lo"],
                output[f"ypred_{_cf}_true"],
                output[f"ypred_{_cf}_hi"],
                output[f"ypred_{_cf}_vhi"]
            ],
            curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
            tstmp=tstmp,
            outdir=outdir,
            show=True,
        )

        # Now plot residuals, but using imaging to bring out patterns
        for _fit in ("lo", "true", "hi", "vhi"):

            # Residuals = data - model
            _res = output[f"y_{_cf}"] - output[f"ypred_{_cf}_{_fit}"]
            output[f"res_{_cf}_{_fit}"] = _res.copy()  # store residuals
            plot_residuals(
                residuals=_res,
                fit_display=FIT_DISPLAY[_fit],
                curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
                tstmp=tstmp,
                outdir=outdir,
                show=True,
            )
            # Calculate residual periodogram via FFT and store
            output[f"rp_{_cf}_{_fit}"] = np.abs(np.fft.rfft(_res))**2 / len(_res)

        # Calculate periodograms of just the errors for plotting
        output[f"ep_{_cf}"] = np.abs(np.fft.rfft(output[f"e_{_cf}"]))**2 / len(output[f"e_{_cf}"])

        # Now we plot error and residual periodograms
        plot_periodograms(
            [
                output[f"ep_{_cf}"],  # iid errors periodogram for comparison
                output[f"rp_{_cf}_lo"],
                output[f"rp_{_cf}_true"],
                output[f"rp_{_cf}_hi"],
                output[f"rp_{_cf}_vhi"],
            ],
            nfull=nx,
            curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
            tstmp=tstmp,
            outdir=outdir,
            show=True,
        )

        # Calculate (circular) autocorrelation functions via inverse FFT of residual periodograms
        for _fit in ("lo", "true", "hi", "vhi"):

            output[f"racf_{_cf}_{_fit}"] = np.fft.irfft(output[f"rp_{_cf}_{_fit}"])
            output[f"racf_{_cf}_{_fit}"] /= output[f"racf_{_cf}_{_fit}"][0]  # variance normalize
            output[f"racf_{_cf}_{_fit}"] = (  # take only first n // 2 + 1 elements due to symmetry
                output[f"racf_{_cf}_{_fit}"][:len(output[f"rp_{_cf}_{_fit}"])])

        # Calculate (circular) autocorrelation function of just the errors for plotting
        output[f"eacf_{_cf}"] = np.fft.irfft(output[f"ep_{_cf}"])
        output[f"eacf_{_cf}"] /= output[f"eacf_{_cf}"][0]  # variance normalize
        # take only first n // 2 + 1 elements due to symmetry
        output[f"eacf_{_cf}"] = output[f"eacf_{_cf}"][:len(output[f"ep_{_cf}"])]

        # Now plot autocorrelation functions
        plot_acfs(
            [
                output[f"eacf_{_cf}"][:(1 + ACF_MAX_LAG)],  # iid errors periodogram for comparison
                output[f"racf_{_cf}_lo"][:(1 + ACF_MAX_LAG)],
                output[f"racf_{_cf}_true"][:(1 + ACF_MAX_LAG)],
                output[f"racf_{_cf}_hi"][:(1 + ACF_MAX_LAG)],
                output[f"racf_{_cf}_vhi"][:(1 + ACF_MAX_LAG)],
            ],
            nfull=nx,
            curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
            tstmp=tstmp,
            outdir=outdir,
            show=True,
        )
