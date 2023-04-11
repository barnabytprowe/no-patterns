"""
fitting_polynomials_2d.py
=========================
Script that generates examples of 2D contour curve fitting scenarios with
additive iid Gaussian noise, illustrating unfitting, correctly specified, and
overfitting scenarios.

Saves output into a uniquely timestamped subfolder of ./plots/polynomials_2d/.
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.linear_model
import sklearn.preprocessing


# Parameters
# ==========

# Side of unit square
nx = 10

# Sigma of iid pixel noise
noise_sigma = 1.

# True, low (insufficient) and high (overfitting) polynomial order to use when fitting
fit_degree_true = 8  # the real signal in the simulations will be a 2D polnoymial of this order
fit_degree_lo = 2
fit_degree_hi = 16

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 8.

# Plotting settings
FIGSIZE = (6, 4.85)
CMAP = "Greys_r"
TITLE_SIZE = "x-large"

# Output folder structure
PLTDIR = os.path.join(".", "plots")
PROJDIR = os.path.join(".", "plots", "polynomials_2d")


# Functions
# =========

def polynomial_design_matrix(X, degree):
    """Returns polynomial design matrix for input coordinates and polynomial degree.

    Args:
        X:
            array-like of shape (n_samples, n_dimensions), the data to input to
            the polynomial features
        degree: maximum polynomial degree of features in all dimensions
    """
    poly_features = sklearn.preprocessing.PolynomialFeatures(degree=degree)
    return poly_features.fit_transform(X, y=None)


def build_output_folder_structure(timestamp):
    """Builds output folder structure using input timestamp and module scope
    PLTDIR, PROJDIR constant variables.  Returns output folder name.
    """
    for _dir in (PLTDIR, PROJDIR):

        if not os.path.isdir(_dir):
            os.mkdir(_dir)

    outdir = os.path.join(PROJDIR, timestamp)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    return outdir


# Main script
# ===========

if __name__ == "__main__":

    # Current timestamp, used in I/0
    tstmp =  pd.Timestamp.now().isoformat().replace(":", "")
    outdir = build_output_folder_structure(tstmp)

    # Output dict - will be pickled
    output = {}

    # Define x, y grid coords for centroids of a unit square spanning square grid
    # centred on the origin
    xvals = np.linspace(-.5 + 1./(2. * nx), .5 - 1./(2. * nx), num=nx, endpoint=True)
    Xxgrid, Xygrid = np.meshgrid(xvals, xvals)

    # Use sklearn PolynomialFeatures to model simple polynomials in 2D
    # x, y coordinates
    X = np.stack([Xxgrid.flatten(order="C"), Xygrid.flatten(order="C")], axis=1)
    # Design matrices for the true, too low and too high cases
    features_true = polynomial_design_matrix(X=X, degree=fit_degree_true)
    features_fit_lo = polynomial_design_matrix(X=X, degree=fit_degree_lo)
    features_fit_hi = polynomial_design_matrix(X=X, degree=fit_degree_hi)

    # Build the true 2D contour and plot
    ctrue = np.random.randn(features_true.shape[-1]) * coeff_signal_to_noise
    ztrue = (np.matmul(features_true, ctrue)).reshape((nx, nx), order="C")
    output["ctrue"] = ctrue
    output["ztrue"] = ztrue

    fig = plt.figure(figsize=FIGSIZE)
    plt.title("Ideal model curve", size=TITLE_SIZE)
    plt.pcolor(ztrue, cmap=CMAP); plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ideal_"+tstmp+".png"))
    plt.show()

    # Add the random noise to generate the dataset, and plot
    zdata = ztrue + noise_sigma * np.random.randn(*ztrue.shape)
    output["zdata"] = zdata

    fig = plt.figure(figsize=FIGSIZE)
    plt.title("Data", size=TITLE_SIZE)
    plt.pcolor(zdata, cmap=CMAP); plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "data_"+tstmp+".png"))
    plt.show()

    # Perform too low order, true and too high order regressions
    predictions = []
    zflat = zdata.flatten(order="C")
    for features in (features_fit_lo, features_true, features_fit_hi):

        regr = sklearn.linear_model.LinearRegression()
        regr.fit(features, zflat)
        predictions.append(regr.predict(features).reshape((nx, nx), order="C"))

    pred_lo, pred_true, pred_hi = tuple(predictions)
    output["pred_lo"] = pred_lo
    output["pred_true"] = pred_true
    output["pred_hi"] = pred_hi

    # Plot residuals
    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(zdata - pred_lo, cmap=CMAP); plt.colorbar(); plt.clim([-2.5, 2.5])
    plt.title("Low order polynomial fit residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lo_"+tstmp+".png"))
    plt.show()
    rlo = zdata - pred_lo
    output["rlo"] = rlo

    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(zdata - pred_true, cmap=CMAP); plt.colorbar(); plt.clim([-2.5, 2.5])
    plt.title("Matching order polynomial fit residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "matching_"+tstmp+".png"))
    plt.show()
    rtrue = zdata - pred_true
    output["rtrue"] = rtrue

    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(zdata - pred_hi, cmap=CMAP); plt.colorbar(); plt.clim([-2.5, 2.5])
    plt.title("High order polynomial fit residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hi_"+tstmp+".png"))
    plt.show()
    rhi = zdata - pred_hi
    output["rhi"] = rhi

    outfile = os.path.join(outdir, "output_"+tstmp+".pickle")
    print("Saving to "+outfile)
    with open(outfile, "wb") as fout:
        pickle.dump(output, fout)
