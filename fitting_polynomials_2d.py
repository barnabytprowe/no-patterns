"""
fitting_polynomials_2d.py
=========================

Examples of regression of in two dimensions, described in the paper "No
patterns in regression residuals," illustrating underspecified, correctly
specified, and overspecified regression of randomly-generated polynomial
surfaces on a regular 2D grid.

Saves output from each simulated regression into a uniquely timestamped
subfolder of ./plots/polynomials_2d/.
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

# Datapoints per side of coordinate grid
nx = 10
# Side dimension of coordinate grid
side_dim = np.sqrt(2.)

# Sigma of iid pixel noise
noise_sigma = 1.

# Low (underspecified), true / matching, high (overspecified) and very high (very overspecified)
# polynomial model set degree to use in simulated regressions
fit_degree_lo = 2
fit_degree_true = 8  # the actual signal curve will be a 2D polynomial series of this degree
fit_degree_hi = 12
fit_degree_vhi = 16

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 8.

# Plotting settings
FIGSIZE = (6, 4.9)  # this makes the pcolor plots approximately square
CLIM = [-2.5, 2.5]  # a reasonable balance to show features across the lo->vhi residual plots
CMAP = "Greys_r"
TITLE_SIZE = "x-large"

# Output folder structure
PLTDIR = os.path.join(".", "plots")
PROJDIR = os.path.join(PLTDIR, "polynomials_2d")


# Functions
# =========

def polynomial_design_matrix(square_dimension, n_side, degree, order="C"):
    """Returns a polynomial design matrix for a square coordinate grid centred
    on the origin, up to the given degree in total polynomial order.

    Args:
        square_dimension: side dimensions of the coordinate grid square
        n_side: number of grid points per side
        degree: maximum polynomial degree of features in all dimensions
        order: array row/column ordering to use in np.flatten, default: 'C'
    """
    # Define x, y grid coords for square of linear dimension side_dim centred on the origin
    xvals = np.linspace(-square_dimension / 2., square_dimension / 2., num=n_side, endpoint=True)
    Xxgrid, Xygrid = np.meshgrid(xvals, xvals)
    X = np.stack([Xxgrid.flatten(order=order), Xygrid.flatten(order=order)], axis=1)

    # Use sklearn PolynomialFeatures to model simple polynomials over x0, x1 coordinates
    poly_features = sklearn.preprocessing.PolynomialFeatures(degree=degree)
    return poly_features.fit_transform(X, y=None)


def build_output_folder_structure(timestamp, project_dir=PROJDIR):
    """Builds output folder structure using input timestamp and module scope
    PLTDIR, PROJDIR constant variables.  Returns output folder name.
    """
    for _dir in (PLTDIR, project_dir):

        if not os.path.isdir(_dir):
            os.mkdir(_dir)

    outdir = os.path.join(project_dir, timestamp)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    return outdir


# Main script
# ===========

if __name__ == "__main__":

    # Current timestamp, used in I/0
    tstmp =  pd.Timestamp.now().isoformat().replace(":", "")
    outdir = build_output_folder_structure(tstmp, project_dir=PROJDIR)

    # Output dict - will be pickled
    output = {}

    # Design matrices
    features_fit_lo = polynomial_design_matrix(
        square_dimension=side_dim, n_side=nx, degree=fit_degree_lo, order="C")
    features_true = polynomial_design_matrix(
        square_dimension=side_dim, n_side=nx, degree=fit_degree_true, order="C")
    features_fit_hi = polynomial_design_matrix(
        square_dimension=side_dim, n_side=nx, degree=fit_degree_hi, order="C")
    features_fit_vhi = polynomial_design_matrix(
        square_dimension=side_dim, n_side=nx, degree=fit_degree_vhi, order="C")

    # Build the true / ideal 2D contour and plot
    ctrue = np.random.randn(features_true.shape[-1]) * coeff_signal_to_noise
    ztrue = (np.matmul(features_true, ctrue)).reshape((nx, nx), order="C")
    output["ctrue"] = ctrue
    output["ztrue"] = ztrue

    fig = plt.figure(figsize=FIGSIZE)
    plt.title("Ideal model curve", size=TITLE_SIZE)
    plt.pcolor(ztrue, cmap=CMAP)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ideal_"+tstmp+".png"))
    plt.show()

    # Add the random noise to generate the dataset and plot
    zdata = ztrue + noise_sigma * np.random.randn(*ztrue.shape)
    output["zdata"] = zdata

    fig = plt.figure(figsize=FIGSIZE)
    plt.title("Data", size=TITLE_SIZE)
    plt.pcolor(zdata, cmap=CMAP)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "data_"+tstmp+".png"))
    plt.show()

    # Perform too low, matching, too high, and very much too high degree regressions on data
    regr = sklearn.linear_model.LinearRegression()  # uses LAPACK via np leastsq under the hood
    zflat = zdata.flatten(order="C")
    predictions = []
    for features in (features_fit_lo, features_true, features_fit_hi, features_fit_vhi):

        regr.fit(features, zflat)
        predictions.append(regr.predict(features).reshape((nx, nx), order="C"))

    pred_lo, pred_true, pred_hi, pred_vhi = tuple(predictions)
    output["pred_lo"] = pred_lo
    output["pred_true"] = pred_true
    output["pred_hi"] = pred_hi
    output["pred_vhi"] = pred_vhi

    # Residuals
    rlo = zdata - pred_lo
    rtrue = zdata - pred_true
    rhi = zdata - pred_hi
    rvhi = zdata - pred_vhi
    output["rlo"] = rlo
    output["rtrue"] = rtrue
    output["rhi"] = rhi
    output["rvhi"] = rvhi

    # Plot residuals
    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(rlo, cmap=CMAP)
    plt.colorbar()
    plt.clim(CLIM)
    plt.title("Low degree polynomial residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lo_"+tstmp+".png"))
    plt.show()

    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(rtrue, cmap=CMAP)
    plt.colorbar()
    plt.clim(CLIM)
    plt.title("Matching degree polynomial residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "matching_"+tstmp+".png"))
    plt.show()

    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(rhi, cmap=CMAP)
    plt.colorbar()
    plt.clim(CLIM)
    plt.title("High degree polynomial residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hi_"+tstmp+".png"))
    plt.show()

    fig = plt.figure(figsize=FIGSIZE)
    plt.pcolor(rvhi, cmap=CMAP)
    plt.colorbar()
    plt.clim(CLIM)
    plt.title("Very high degree polynomial residuals", size=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "vhi_"+tstmp+".png"))
    plt.show()

    # Save output for further analysis
    outfile = os.path.join(outdir, "output_"+tstmp+".pickle")
    print("Saving to "+outfile)
    with open(outfile, "wb") as fout:
        pickle.dump(output, fout)
