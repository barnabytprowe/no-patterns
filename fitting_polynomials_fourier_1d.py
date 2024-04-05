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

# True, low (insufficient) and high (overfitting) polynomial order to use when fitting
fit_degree_true = 8  # the real signal in the simulations will be a 1D sinusoidal / polnoymial
                     # series up to this order
fit_degree_lo = 2
fit_degree_hi = 16
fit_degree_vhi = 32  # Added to illustrate the more extreme behaviour more clearly

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 1.

# Title display strings for plots
fit_display = {
    "lo": "Low degree",
    "true": "Matching",
    "hi": "High degree",
    "vhi": "Very high degree",
}
curve_family_display = {"sinu": "Fourier", "cheb": "polynomial"}


# Plotting settings
FIGSIZE = (10, 4)
FIGSIZE_RESIDUALS = (10, 1.25)
CLIM = [-2.5, 2.5]
CMAP = "Greys_r"
TITLE_SIZE = "x-large"

# Periodogram chart settings
PERIODOGRAM_YTICKS = 10**np.linspace(-32., 4., num=10, dtype=float)
PERIODOGRAM_YLIM = 10**np.asarray([-32, 4.], dtype=float)

# Output folder structure: project dir
PROJDIR = os.path.join(PLTDIR, "polynomials_fourier_1d")

# Output file types
OUTFILE_EXTENSIONS = (".png", ".pdf")


# Functions
# =========

def sinusoid_design_matrix(x, degree):
    """Returns the sinusoid [cosx, sinx] design matrix up to input degree"""
    sinx = np.asarray([np.sin(2. * np.pi * float(j) * x) for j in range(0, degree)]).T
    cosx = np.asarray([np.cos(2. * np.pi * float(j) * x) for j in range(0, degree)]).T
    return np.hstack([cosx, sinx])


def chebyshev_design_matrix(x, degree):
    """Returns the Chebyshev polynomial design matrix up to input degree"""
    i1n = np.eye(degree)
    return np.asarray([numpy.polynomial.chebyshev.chebval(x, _row) for _row in i1n]).T


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
    ax.set_title(f"{fit_display} {curve_family_display} residual map", size=TITLE_SIZE)

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


# Main script
# ===========

if __name__ == "__main__":

    # Current timestamp, used in I/0
    tstmp =  pd.Timestamp.now().isoformat().replace(":", "")
    outdir = fitting_polynomials_2d.build_output_folder_structure(tstmp, project_dir=PROJDIR)

    # Output dict - will be pickled
    output = {}

    # Define x coordinates as linearly spaced points on the unit interval
    x_sinu = np.linspace(0., 1., num=nx, endpoint=False)
    x_cheb = np.linspace(-1., 1., num=nx, endpoint=False)

    # Design matrices for the true, too low and too high cases
    features_lo = {
        "sinu": sinusoid_design_matrix(x=x_sinu, degree=fit_degree_lo),
        "cheb": chebyshev_design_matrix(x=x_cheb, degree=(1 + fit_degree_lo)),
    }
    features_true = {
        "sinu": sinusoid_design_matrix(x=x_sinu, degree=fit_degree_true),
        "cheb": chebyshev_design_matrix(x=x_cheb, degree=(1 + fit_degree_true)),
    }
    features_hi = {
        "sinu": sinusoid_design_matrix(x=x_sinu, degree=fit_degree_hi),
        "cheb": chebyshev_design_matrix(x=x_cheb, degree=(1 + fit_degree_hi)),
    }
    features_vhi = {
        "sinu": sinusoid_design_matrix(x=x_sinu, degree=fit_degree_vhi),
        "cheb": chebyshev_design_matrix(x=x_cheb, degree=(1 + fit_degree_vhi)),
    }

    # Build the true 1d curve coefficients
    sinu_coeffs_true = np.random.randn(features_true["sinu"].shape[-1]) * coeff_signal_to_noise
    cheb_coeffs_true = np.random.randn(features_true["cheb"].shape[-1]) * coeff_signal_to_noise
    output["sinu_coeffs_true"] = sinu_coeffs_true
    output["cheb_coeffs_true"] = cheb_coeffs_true

    # Build the true 1d curves from these coefficients
    ytrue_sinu = np.matmul(features_true["sinu"], sinu_coeffs_true)
    ytrue_cheb = np.matmul(features_true["cheb"], cheb_coeffs_true)
    output["ytrue_sinu"] = ytrue_sinu
    output["ytrue_cheb"] = ytrue_cheb

    # Add random Gaussian iid errors to generate our simulation dataset y values
    e_sinu = noise_sigma * np.random.randn(nx)
    e_cheb = noise_sigma * np.random.randn(nx)
    y_sinu = ytrue_sinu + e_sinu
    y_cheb = ytrue_cheb + e_cheb
    output["e_sinu"] = e_sinu
    output["e_cheb"] = e_cheb
    output["y_sinu"] = y_sinu
    output["y_cheb"] = y_cheb

    # Fit the sinusoidal and polynomial features
    features_dict = {
        "lo": features_lo,
        "true": features_true,
        "hi": features_hi,
        "vhi": features_vhi,
    }
    for _curve_family in ("sinu", "cheb"):

        # Calculate periodograms of just the errors; will be plotted later
        output[f"ep_{_curve_family}"] = np.abs(
            np.fft.rfft(output[f"e_{_curve_family}"]))**2 / len(output[f"e_{_curve_family}"])

        # Perform regression at different degrees
        for _fit in ("lo", "true", "hi", "vhi"):

            _design_matrix = features_dict[_fit][_curve_family]
            _coeffs = np.linalg.lstsq(_design_matrix, output[f"y_{_curve_family}"], rcond=None)[0]
            _yfit = _design_matrix.dot(_coeffs.T)
            output[f"ypred_{_curve_family}_{_fit}"] = _yfit

        # Then plot data versus x and regression predictions as curves of y versus x
        # First we plot the lines in a conventional x, y graph format
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title(
            curve_family_display[_curve_family].title()+" series regression in one dimension",
            size=TITLE_SIZE,
        )
        _x = {"sinu": x_sinu, "cheb": x_cheb}[_curve_family]
        ax.plot(
            _x, output[f"ytrue_{_curve_family}"],
            color="k", ls="-", linewidth=2, label="Ideal model",
        )
        ax.plot(_x, output[f"y_{_curve_family}"], "k+", markersize=15, label="Data")
        ax.plot(
            _x, output[f"ypred_{_curve_family}_lo"],
            color="red", ls="--", linewidth=1, label=fit_display["lo"],
        )
        ax.plot(
            _x, output[f"ypred_{_curve_family}_true"],
            color="k", ls="-", linewidth=1, label=fit_display["true"],
        )
        ax.plot(
            _x, output[f"ypred_{_curve_family}_hi"],
            color="blue", ls="-.", linewidth=1, label=fit_display["hi"],
        )
        ax.plot(
            _x, output[f"ypred_{_curve_family}_vhi"],
            color="purple", ls=":", linewidth=1.25, label=fit_display["vhi"],
        )
        ax.set_xlabel(r"$x$")
        ax.grid()
        ax.legend()
        fig.tight_layout()
        plt.show()
        for _suffix in OUTFILE_EXTENSIONS:

            outfile = os.path.join(outdir, f"curves_{_curve_family}_{tstmp}{_suffix}")
            print(f"Saving to {outfile}")
            fig.savefig(outfile)

        plt.close(fig)

        # Now we are going to look at the residuals, but imaging rather than plotting them
        # Define and store the residuals
        for _fit in ("lo", "true", "hi", "vhi"):

            # Residuals = data - model
            _res = output[f"y_{_curve_family}"] - output[f"ypred_{_curve_family}_{_fit}"]
            output[f"res_{_curve_family}_{_fit}"] = _res.copy()

            plot_residuals(
                residuals=_res,
                fit_display=fit_display[_fit],
                curve_family_display=curve_family_display[_curve_family],
                tstmp=tstmp,
                outdir=outdir,
                show=True,
            )

            # Calculate and store residual periodogram
            output[f"rp_{_curve_family}_{_fit}"] = np.abs(np.fft.rfft(_res))**2 / len(_res)

        # Now we're going to plot periodograms
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title(
            curve_family_display[_curve_family].title()+" series regression residual periodograms",
            size=TITLE_SIZE,
        )
        ax.plot(
            output[f"ep_{_curve_family}"],
            color="k", ls="--", linewidth=1, label="iid errors",
        )
        ax.plot(
            output[f"rp_{_curve_family}_lo"],
            color="red", ls="--", linewidth=1.5, label=fit_display["lo"],
        )
        ax.plot(
            output[f"rp_{_curve_family}_true"],
            color="k", ls="-", linewidth=1.5, label=fit_display["true"],
        )
        ax.plot(
            output[f"rp_{_curve_family}_hi"],
            color="blue", ls="-.", linewidth=1.5, label=fit_display["hi"],
        )
        ax.plot(
            output[f"rp_{_curve_family}_vhi"],
            color="purple", ls=":", linewidth=1.5, label=fit_display["vhi"],
        )
        ax.set_yscale("log")
        ax.set_yticks(PERIODOGRAM_YTICKS)
        ax.set_ylim(PERIODOGRAM_YLIM)
        ax.grid()
        ax.legend()
        fig.tight_layout()
        plt.show()
        for _suffix in OUTFILE_EXTENSIONS:

            outfile = os.path.join(outdir, f"periodograms_{_curve_family}_{tstmp}{_suffix}")
            print(f"Saving to {outfile}")
            fig.savefig(outfile)

        plt.close(fig)
