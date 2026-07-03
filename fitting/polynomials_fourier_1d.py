"""
no_patterns/fitting/polynomials_fourier_1d.py
=============================================

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
import numpy as np
import numpy.polynomial.chebyshev
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from polynomials_2d import PLTDIR, build_output_folder_structure


# Parameters
# ==========

# Number of data points
NX = 100

# Sigma of iid pixel noise
NOISE_SIGMA = 1.

# Settings for the ideal model, underspecified, overspecified and highly overspecified
# series model degrees to use as model sets for the ideal model and fitting
FIT_DEGREES = {}

FIT_DEGREES["lo"] = {"cheb": 2, "sinu": 1}  # underspecified model sets

# Real signal (ideal model) degree in the simulations (1D polynomial / Fourier series) will also
# be used as a model set for regression for each curve family
FIT_DEGREES["true"] = {"cheb": 8, "sinu": 4}

FIT_DEGREES["hi"] = {"cheb": 16, "sinu": 8}  # overspecified model sets
FIT_DEGREES["vhi"] = {"cheb": 32, "sinu": 16}  # added to illustrate more extreme behaviour clearly

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to NOISE_SIGMA
COEFF_SIGNAL_TO_NOISE = 1.

# Define x coordinates as linearly spaced points on the some interval, e.g. [0, 1), [-1, 1)
# depending on curve family
XMINS = {"cheb": -1., "sinu": 0.}
XMAXS = {"cheb":  1., "sinu": 1.}
XARRS = {
    _family: np.linspace(XMINS[_family], XMAXS[_family], num=NX, endpoint=False)
    for _family in ("cheb", "sinu")
}

VERBOSE = False

# Plot settings
FIGSIZE = (10, 4)
FIGSIZE_PERIODOGRAMS = (10, 6)
FIGSIZE_RESIDUALS = (10, 1.25)
CLIM = [-2.5, 2.5]
CMAP = "Greys_r"
# set plotting XLIMS depending on XARRS
XLIMS = {_cf: (XARRS[_cf].min() - 0.02, XARRS[_cf].max() + 0.02) for _cf in ("cheb", "sinu")}
TITLE_SIZE = "x-large"
LABEL_SIZE = "large"

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
ACF_MAX_LAG = 12

# Output folder structure: project dir
PROJDIR = os.path.join(PLTDIR, "polynomials_fourier_1d")

# Output file types
OUTFILE_EXTENSIONS = (".png", ".pdf")


# Functions
# =========

def sinusoid_design_matrix(xarr, degree):
    """Returns the sinusoid [cosx, sinx] design matrix up to input degree"""
    sinx = np.asarray([np.sin(2. * np.pi * float(j) * xarr) for j in range(0, 1 + degree)]).T
    cosx = np.asarray([np.cos(2. * np.pi * float(j) * xarr) for j in range(0, 1 + degree)]).T
    return np.hstack([cosx, sinx])


def chebyshev_design_matrix(xarr, degree):
    """Returns the Chebyshev polynomial design matrix up to input degree"""
    i1n = np.eye(1 + degree)
    return np.asarray([numpy.polynomial.chebyshev.chebval(xarr, _row) for _row in i1n]).T


def features(xarrs=XARRS, fit_degrees=FIT_DEGREES):
    """Returns a dict containing the Sinusoid and Chebyshev design matrices for
    all series degrees in the input fit_degrees dict.
    """
    return {
        _degree_label: {
            "cheb": chebyshev_design_matrix(xarr=xarrs["cheb"], degree=_degree_dict["cheb"]),
            "sinu": sinusoid_design_matrix(xarr=xarrs["sinu"], degree=_degree_dict["sinu"]),
        }
        for _degree_label, _degree_dict in fit_degrees.items()
    }


def plot_regressions(xarr, yarrs, xlim, curve_family_display, tstmp, outdir, show=True):
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
        xlim: length-2 iterable for input to plot axis set_xlim
        curve_family_display: one of {'polynomial', 'Fourier'}
        tstmp: timestamp used in folder structure
        outdir: output folder
        show: plt.show()?
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(
        f"{curve_family_display.title()} series regression", size=TITLE_SIZE)
    ax.plot(xarr, yarrs[0], color="k", ls="-", linewidth=2, label="Ideal model")
    ax.plot(xarr, yarrs[1], "k+", markersize=15, label="Data")
    ax.plot(xarr, yarrs[2], color="red", ls="--", linewidth=1, label=FIT_DISPLAY["lo"])
    ax.plot(xarr, yarrs[3], color="k", ls="-", linewidth=1, label=FIT_DISPLAY["true"])
    ax.plot(xarr, yarrs[4], color="blue", ls="-.", linewidth=1, label=FIT_DISPLAY["hi"])
    ax.plot(xarr, yarrs[5], color="purple", ls=":", linewidth=1.25, label=FIT_DISPLAY["vhi"])
    ax.set_xlabel(r"$x$", size=LABEL_SIZE)
    ax.set_ylabel(r"$y$", size=LABEL_SIZE)
    ax.grid()
    ax.legend()
    ax.set_xlim(xlim)
    fig.tight_layout()
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir,
            curve_family_display.lower(),
            f"curves_{curve_family_display.lower().replace(' ', '_')}_{tstmp}{_suffix}")
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
            outdir,
            curve_family_display.lower(),
            (
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
    fig, (ax0, ax1) = plt.subplots(2, figsize=FIGSIZE_PERIODOGRAMS)
    fig.suptitle(
        curve_family_display.title()+" series regression residual periodograms", size=TITLE_SIZE)

    for _ax, _method in zip((ax0, ax1), ("semilogy", "plot")):
        _plt = getattr(_ax, _method)
        _plt(
            np.arange(len(periodograms[0])) / nfull, periodograms[0], color="k", ls="--",
            linewidth=1, label="iid errors"
        )
        _plt(
            np.arange(len(periodograms[1])) / nfull, periodograms[1], color="red", ls="--",
            linewidth=1.5, label=FIT_DISPLAY["lo"],
        )
        _plt(
            np.arange(len(periodograms[2])) / nfull, periodograms[2], color="k", ls="-",
            linewidth=1.5, label=FIT_DISPLAY["true"],
        )
        _plt(
            np.arange(len(periodograms[3])) / nfull, periodograms[3], color="blue", ls="-.",
            linewidth=1.5, label=FIT_DISPLAY["hi"],
        )
        _plt(
            np.arange(len(periodograms[4])) / nfull, periodograms[4], color="purple", ls=":",
            linewidth=1.5, label=FIT_DISPLAY["vhi"],
        )
        if _method == "plot":
            _ax.set_yscale("log")
            _ax.set_yticks(PERIODOGRAM_YTICKS)
            _ax.set_ylim(PERIODOGRAM_YLIM)
            _ax.set_xlabel(r"Frequency $k/N$", size=LABEL_SIZE)
            _ax.legend()
        else:
            _ax.set_ylim((1.e-3, 1.e3))

        _ax.set_ylabel(f"$\iota[k]$", size=LABEL_SIZE)
        _ax.grid()

    fig.tight_layout()
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir,
            curve_family_display.lower(),
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
    ax.set_xlabel(r"Lag $\ell$", size=LABEL_SIZE)
    ax.set_ylabel(r"$\left. r[\ell] ~ \middle/ ~ r[0] \right. $", size=LABEL_SIZE)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    for _suffix in OUTFILE_EXTENSIONS:

        outfile = os.path.join(
            outdir,
            curve_family_display.lower(),
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

    design_matrices = features()

    # Current timestamp, used in I/0
    tstmp = pd.Timestamp.now().isoformat().replace(":", "")
    outdir = build_output_folder_structure(tstmp, project_dir=PROJDIR)

    # Output dict - will be pickled
    output = {}

    for _cf in ("cheb", "sinu"):  # Big outer loop over curve family
        # Build the true 1d curve coefficients
        output[f"{_cf}_coeffs_true"] = COEFF_SIGNAL_TO_NOISE * NOISE_SIGMA * np.random.randn(
            design_matrices["true"][_cf].shape[-1]
        )
        # Build the true 1d curves from these coefficients
        output[f"ytrue_{_cf}"] = np.matmul(
            design_matrices["true"][_cf],
            output[f"{_cf}_coeffs_true"],
        )
        # Add random Gaussian iid errors to generate our simulation dataset y values
        output[f"e_{_cf}"] = NOISE_SIGMA * np.random.randn(NX)
        output[f"y_{_cf}"] = output[f"ytrue_{_cf}"] + output[f"e_{_cf}"]

        # Plot scatter plots of data, ideal model and predictions
        # First perform regression at different degrees to generate predictions
        for _degree_label in FIT_DEGREES:
            _design_matrix = design_matrices[_degree_label][_cf]
            _coeffs = np.linalg.lstsq(_design_matrix, output[f"y_{_cf}"], rcond=None)[0]
            if VERBOSE:
                print(
                    # note the sinusoidal design matrix contains an inactive feature for sin(0 * x)
                    # (but this is handled without issue to machine eps by the SVD leastsq solution)
                    f"{_cf} {_design_matrix} n_coeffs "
                    f"= {_design_matrix.shape[1] - (1 if _cf == 'sinu' else 0)}"
                )
                print(_coeffs)

            _yfit = _design_matrix.dot(_coeffs.T)
            output[f"ypred_{_cf}_{_degree_label}"] = _yfit

        # Prep output folder
        if not os.path.isdir(os.path.join(outdir, CURVE_FAMILY_DISPLAY[_cf].lower())):
            os.mkdir(os.path.join(outdir, CURVE_FAMILY_DISPLAY[_cf].lower()))

        # Plot ideal model, data, and ordinary least squares regression predictions
        plot_regressions(
            xarr=XARRS[_cf],
            yarrs=[
                output[f"ytrue_{_cf}"],  # ideal model
                output[f"y_{_cf}"],  # data
                output[f"ypred_{_cf}_lo"],
                output[f"ypred_{_cf}_true"],
                output[f"ypred_{_cf}_hi"],
                output[f"ypred_{_cf}_vhi"]
            ],
            xlim=XLIMS[_cf],
            curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
            tstmp=tstmp,
            outdir=outdir,
            show=True,
        )

        # Now plot residuals, but using imaging to bring out patterns
        for _degree_label in FIT_DEGREES:
            # Residuals = data - model
            _res = output[f"y_{_cf}"] - output[f"ypred_{_cf}_{_degree_label}"]
            output[f"res_{_cf}_{_degree_label}"] = _res.copy()  # store residuals
            plot_residuals(
                residuals=_res,
                fit_display=FIT_DISPLAY[_degree_label],
                curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
                tstmp=tstmp,
                outdir=outdir,
                show=True,
            )
            # Calculate residual periodogram via FFT and store
            output[f"rp_{_cf}_{_degree_label}"] = np.abs(np.fft.rfft(_res))**2 / NX
            assert len(output[f"rp_{_cf}_{_degree_label}"]) == (NX // 2 + 1)
            # Residuals from OLS should be ~ 0 anyhow so we whether to mean subtract is moot, but
            # for comparison (and a more standard_ ACF calculation (e.g. Box et al 15, also
            # Wikipedia) we will pad residuals with zeros to 2x length then calculate and store the
            # resulting periodogram; in practice the two ACF definitions will be ~close at lag << nx
            _rmean = _res.mean()
            assert np.isclose(_rmean, 0, atol=1.e-14, rtol=0.)
            _zpres = np.zeros(2 * NX, dtype=float)  # zero-padded storage array
            _zpres[:NX] = _res  - _rmean
            output[f"zprp_{_cf}_{_degree_label}"] = np.abs(np.fft.rfft(_zpres))**2 / (2 * NX)

        # Calculate periodograms of just the errors for plotting
        output[f"ep_{_cf}"] = np.abs(np.fft.rfft(output[f"e_{_cf}"]))**2 / NX
        # also calc 2x length zero padded, mean subtracted errors periodogram
        _zperrs = np.zeros(2 * NX, dtype=float)  # zero-padded storage array
        _zperrs[:NX] = output[f"e_{_cf}"] - output[f"e_{_cf}"].mean()  # mean subtract
        output[f"zpep_{_cf}"] = np.abs(np.fft.rfft(_zperrs))**2 / (2 * NX)

        # Now we plot error and residual periodograms
        plot_periodograms(
            [
                output[f"ep_{_cf}"],  # iid errors periodogram for comparison
                output[f"rp_{_cf}_lo"],
                output[f"rp_{_cf}_true"],
                output[f"rp_{_cf}_hi"],
                output[f"rp_{_cf}_vhi"],
            ],
            nfull=NX,
            curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
            tstmp=tstmp,
            outdir=outdir,
            show=True,
        )

        # Calculate (circular) autocorrelation functions via inverse FFT of residual periodograms
        for _degree_label in FIT_DEGREES:
            _nhalf = len(output[f"rp_{_cf}_{_degree_label}"])  # only first n//2 + 1 elements matter by symmetry
            output[f"racf_{_cf}_{_degree_label}"] = np.fft.irfft(
                output[f"rp_{_cf}_{_degree_label}"]
            )
            if VERBOSE:
                print(f"racf_{_cf}_{_degree_label}[0] = {output[f'racf_{_cf}_{_degree_label}'][0]}")
            output[f"racf_{_cf}_{_degree_label}"] /= output[f"racf_{_cf}_{_degree_label}"][0]  # variance normalize
            # take non-redundant first _nhalf elements only
            output[f"racf_{_cf}_{_degree_label}"] = output[f"racf_{_cf}_{_degree_label}"][:_nhalf]
            # then calculate the unbiased (non-circular) equivalent for comparison from the
            # zero-padded periodograms
            output[f"uracf_{_cf}_{_degree_label}"] = np.fft.irfft(
                output[f"zprp_{_cf}_{_degree_label}"]
            )
            output[f"uracf_{_cf}_{_degree_label}"] /= output[f"uracf_{_cf}_{_degree_label}"][0]  # variance normalize
            # take non-redundant first _nhalf elements only, apply the debiasing factor
            output[f"uracf_{_cf}_{_degree_label}"] = (
                output[f"uracf_{_cf}_{_degree_label}"][:_nhalf] * NX
                / (NX - np.arange(_nhalf, dtype=float))
            )
            if VERBOSE:
                print(
                    f"biased - unbiased difference for racf_{_cf}_{_degree_label} "
                    "up to {ACF_MAX_LAG=}:"
                )
                _difference = (
                    output[f"uracf_{_cf}_{_degree_label}"] - output[f"racf_{_cf}_{_degree_label}"]
                )[1:(1 + ACF_MAX_LAG)]
                print(_difference)
                print(pd.Series(_difference).describe())

        # Calculate circular autocorrelation and unbiased (e.g. Smith 2007) equivalent function of
        # just the errors, for plotting
        _nhalf = len(output[f"ep_{_cf}"])
        output[f"eacf_{_cf}"] = np.fft.irfft(output[f"ep_{_cf}"])
        output[f"eacf_{_cf}"] /= output[f"eacf_{_cf}"][0]  # variance normalize
        output[f"eacf_{_cf}"] = output[f"eacf_{_cf}"][:_nhalf]
        output[f"ueacf_{_cf}"] = np.fft.irfft(output[f"zpep_{_cf}"])
        output[f"ueacf_{_cf}"] /= output[f"ueacf_{_cf}"][0]  # variance normalize
        # take non-redundant first _nhalf elements only, apply the debiasing factor
        output[f"ueacf_{_cf}"] = (
            output[f"ueacf_{_cf}"][:_nhalf] * NX / (NX - np.arange(_nhalf, dtype=float))
        )

        # Now plot autocorrelation functions
        plot_acfs(
            [
                output[f"eacf_{_cf}"][:(1 + ACF_MAX_LAG)],  # iid errors ACF for comparison
                output[f"racf_{_cf}_lo"][:(1 + ACF_MAX_LAG)],
                output[f"racf_{_cf}_true"][:(1 + ACF_MAX_LAG)],
                output[f"racf_{_cf}_hi"][:(1 + ACF_MAX_LAG)],
                output[f"racf_{_cf}_vhi"][:(1 + ACF_MAX_LAG)],
            ],
            nfull=NX,
            curve_family_display=CURVE_FAMILY_DISPLAY[_cf],
            tstmp=tstmp,
            outdir=outdir,
            show=True,
        )
