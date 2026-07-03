"""
no_patterns/fitting/polynomials_2d.py
=====================================

Examples of regression of in two dimensions, described in the paper "No
patterns in regression residuals," illustrating underspecified, "correctly
specified", and overspecified regression of randomly-generated polynomial
surfaces on a regular 2D grid.

Saves output from each simulated regression into a uniquely timestamped
subfolder of ./plots/polynomials_2d/.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Parameters
# ==========

# Datapoints per side of coordinate grid
NX = 28
# Extent of coordinate grid
X0X1_MIN = -.85
X0X1_MAX = +.85

# Sigma of iid pixel noise
NOISE_SIGMA = 1.

# Settings for the ideal model, underspecified, overspecified and highly overspecified
# series model degrees to use as model sets for the ideal model and fitting
FIT_DEGREES = {}

FIT_DEGREES["lo"] = 3  # underspecified model sets

# Real signal (ideal model) degree in the simulations (2D polynomial series), will also
# be used as a model set for regression for each curve family
FIT_DEGREES["true"] = 6

FIT_DEGREES["hi"] = 24  # overspecified model sets
FIT_DEGREES["vhi"] = 48  # added to illustrate more extreme behaviour clearly

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to NOISE_SIGMA
COEFF_SIGNAL_TO_NOISE = 1.

# Plotting settings
FIGSIZE = (6, 5)  # this makes the pcolor plots approximately square
CLIM = [-2.5, 2.5]  # a reasonable balance to show features across the lo->vhi residual plots
CMAP = "Greys_r"
TITLE_SIZE = "x-large"

# Output folder structure
PLTDIR = os.path.join(".", "plots")
PROJDIR = os.path.join(PLTDIR, "polynomials_2d")


# Functions
# =========

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


# Consistent functions for determining p, q indices labels for the kth coefficient
# also labelled theta_pq, used in the construction of the design matrix

def _dtri(k):
    """Total order (in both x0 and x1) of the kth column polynomial, minus one"""
    return np.floor(.5 * (np.sqrt(1. + 8. * k) - 3)).astype(int)


def _ktri(k):
    """Number of coefficients in the complete series of degree _dtri(k)"""
    return (1 + _dtri(k)) * (2 + _dtri(k)) // 2


def _ps(k):
    """p index label for the kth coefficient theta_pq"""
    return 1 + _dtri(k) + _ktri(k) - k


def _qs(k):
    """q index label for the kth coefficient theta_pq"""
    return k - _ktri(k)


def square_grid(xmin, xmax, nx=NX, endpoint=True, flatten_order="C"):
    """Returns numpy arrays x0, x1 containing the coordinates of a square grid,
    symmetric with respect to the line x0=x1, defined by input xmin and xmax
    coordinate values.
    """
    xvals = np.linspace(xmin, xmax, num=nx, endpoint=True)
    x0, x1 = np.meshgrid(xvals, xvals)
    if flatten_order is not None:
        x0 = x0.flatten(order=flatten_order)
        x1 = x1.flatten(order=flatten_order)
    return x0, x1


def chebyshev_design_matrix(x0, x1, degree):
    """Returns the Chebyshev polynomial design matrix up to input degree for two
    independent coordinate arrays x0, x1
    """
    if len(x0) != len(x1):
        raise ValueError("input coordinate arrays x0 and x1 unequal length")

    # Get the columns of the matrix corresponding to the x0, x1 coordinate grids
    i1n = np.eye(1 + degree)
    mp = np.asarray([np.polynomial.chebyshev.chebval(x0, _row) for _row in i1n]).T
    mq = np.asarray([np.polynomial.chebyshev.chebval(x1, _row) for _row in i1n]).T

    # Prepare final output matrix
    ncoeff = (degree + 1) * (degree + 2) // 2
    ps = _ps(np.arange(ncoeff, dtype=int))
    qs = _qs(np.arange(ncoeff, dtype=int))
    design_matrix = np.concatenate(
        [(mp[:, _p] * mq[:, _q]).reshape(len(x0), 1) for _p, _q in zip(ps, qs)],
        axis=1,
    )
    return design_matrix


def plot_image(data, title, filename=None, clim=None, show=True, tick_stride=7):
    """Quick image plot using module level constant settings"""

    fig = plt.figure(figsize=FIGSIZE)
    plt.title(title, size=TITLE_SIZE)
    plt.pcolor(data, cmap=CMAP)
    if clim is not None:
        plt.clim(clim)
    plt.colorbar()

    # Ensure integer ticks
    xticks = np.linspace(
        0, data.shape[0], num=1 + data.shape[0]//tick_stride, endpoint=True, dtype=int)
    yticks = np.linspace(
        0, data.shape[1], num=1 + data.shape[1]//tick_stride, endpoint=True, dtype=int)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    return


# Main script
# ===========

if __name__ == "__main__":

    # Current timestamp, used in I/0
    tstmp = pd.Timestamp.now().isoformat().replace(":", "")
    outdir = build_output_folder_structure(tstmp, project_dir=PROJDIR)

    # Output dict - will be pickled
    output = {}

    # Prepare two independent variables on a grid
    x0, x1 = square_grid(xmin=X0X1_MIN, xmax=X0X1_MAX, nx=NX, endpoint=True, flatten_order="C")

    design_matrices = {
        _degree_label: chebyshev_design_matrix(x0, x1, degree=_deg)
        for _degree_label, _deg in FIT_DEGREES.items()
    }

    # Build the true / ideal 2D contour, plot and save
    ctrue = np.random.randn(design_matrices["true"].shape[-1]) * COEFF_SIGNAL_TO_NOISE
    ztrue = (np.matmul(design_matrices["true"], ctrue)).reshape((NX, NX), order="C")

    plot_image(
        ztrue, "Ideal model", filename=os.path.join(outdir, "ideal_"+tstmp+".png"), show=True
    )
    output["ctrue"] = ctrue
    output["ztrue"] = ztrue

    # Add the random noise to generate the dataset, plot and save
    zdata = ztrue + NOISE_SIGMA * np.random.randn(*ztrue.shape)

    plot_image(zdata, "Data", filename=os.path.join(outdir, "data_"+tstmp+".png"), show=True)
    output["zdata"] = zdata

    # Perform too low, matching, too high, and very much too high degree regressions on data
    zflat = zdata.flatten(order="C")
    predictions = {}
    for _degree_label, _design_matrix in design_matrices.items():
        _coeffs = np.linalg.lstsq(_design_matrix, zflat, rcond=None)[0].T
        predictions[_degree_label] = _design_matrix.dot(_coeffs).reshape((NX, NX), order="C")
        output[f"pred_{_degree_label}"] = predictions[_degree_label]  # store in output pickle

    # Calculate and plot residuals
    rlo = zdata - predictions["lo"]
    print(f"Low degree polynomial n_coeffs = {design_matrices['lo'].shape[1]}")
    plot_image(
        rlo,
        "Low degree polynomial residuals",
        filename=os.path.join(outdir, "lo_"+tstmp+".png"),
        clim=CLIM,
    )
    rtrue = zdata - predictions["true"]
    print(f"Matching degree polynomial n_coeffs = {design_matrices['true'].shape[1]}")
    plot_image(
        rtrue,
        "Matching degree polynomial residuals",
        filename=os.path.join(outdir, "matching_"+tstmp+".png"),
        clim=CLIM,
    )
    rhi = zdata - predictions["hi"]
    print(f"High degree polynomial n_coeffs = {design_matrices['hi'].shape[1]}")
    plot_image(
        rhi,
        "High degree polynomial residuals",
        filename=os.path.join(outdir, "hi_"+tstmp+".png"),
        clim=CLIM,
    )
    rvhi = zdata - predictions["vhi"]
    print(f"Very high degree polynomial n_coeffs = {design_matrices['vhi'].shape[1]}")
    plot_image(
        rvhi,
        "Very high degree polynomial residuals",
        filename=os.path.join(outdir, "vhi_"+tstmp+".png"),
        clim=CLIM,
    )
    output["rlo"] = rlo
    output["rtrue"] = rtrue
    output["rhi"] = rhi
    output["rvhi"] = rvhi

    # Save output for further analysis
    outfile = os.path.join(outdir, "output_"+tstmp+".pickle")
    print("Saving to "+outfile)
    with open(outfile, "wb") as fout:
        pickle.dump(output, fout)
