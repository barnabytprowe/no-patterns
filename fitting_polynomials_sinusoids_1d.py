"""
fitting_polynomials_sinusoids_1d.py
===================================
Script that generates examples of 1D curve fitting scenarios with additive iid
Gaussian noise, illustrating underfitting, correctly specified, and overfitting
scenarios.

Saves output into a uniquely timestamped subfolder of
./plots/polynomials_sinusoids_1d/.
"""

import os
import pickle

import matplotlib.pyplot as plt
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
fit_degree_true = 8  # the real signal in the simulations will be a 2D polnoymial of this order
fit_degree_lo = 2
fit_degree_hi = 16

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 1.

# Plotting settings
FIGSIZE = (15, 3.25)
CMAP = "Greys_r"
TITLE_SIZE = "x-large"

# Output folder structure: project dir
PROJDIR = os.path.join(PLTDIR, "polynomials_sinusoids_1d")


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

    # Add random noise to generate our simulation dataset y values
    y_sinu = ytrue_sinu + noise_sigma * np.random.randn(nx)
    y_cheb = ytrue_cheb + noise_sigma * np.random.randn(nx)
    output["y_sinu"] = y_sinu
    output["y_cheb"] = y_cheb

    # Fit the sinusoidal and polynomial features
    features_dict = {"lo": features_lo, "true": features_true, "hi": features_hi}
    curve_family_display = {"sinu": "sinusoidal", "cheb": "Chebyshev polynomial"}
    for _curve_family in ("sinu", "cheb"):

        for _fit in ("lo", "true", "hi"):

            _design_matrix = features_dict[_fit][_curve_family]
            _coeffs = np.linalg.lstsq(_design_matrix, output[f"y_{_curve_family}"], rcond=None)[0]
            _yfit = _design_matrix.dot(_coeffs.T)
            output[f"ypred_{_curve_family}_{_fit}"] = _yfit

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title(
            curve_family_display[_curve_family].title()+" curve fitting in one dimension",
            size=TITLE_SIZE,
        )
        _x = {"sinu": x_sinu, "cheb": x_cheb}[_curve_family]
        ax.plot(
            _x, output[f"ytrue_{_curve_family}"],
            color="k", ls="-", linewidth=2, label="Ideal model")
        ax.plot(_x, output[f"y_{_curve_family}"], "k+", markersize=15, label="Data")
        ax.plot(
            _x, output[f"ypred_{_curve_family}_lo"],
            color="red", ls="--", linewidth=0.75, label="Low (underfitting)",
        )
        ax.plot(
            _x, output[f"ypred_{_curve_family}_true"],
            color="k", ls="-", linewidth=0.75, label="Matching",
        )
        ax.plot(
            _x, output[f"ypred_{_curve_family}_hi"],
            color="blue", ls="-.", linewidth=0.75, label="High (overfitting)",
        )
        ax.set_xlabel(r"$x$")
        ax.grid()
        ax.legend()
        fig.tight_layout()
        plt.show()
        outfile = os.path.join(outdir, f"curves_{_curve_family}_{tstmp}.pdf")
        print(f"Saving to {outfile}")
        fig.savefig(outfile)
        plt.close(fig)
