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
nx = 28
# Extent of coordinate grid
x0x1_min = -.85
x0x1_max = +.85

# Sigma of iid pixel noise
noise_sigma = 1.

# Low (underspecified), true / matching, high (overspecified) and very high (very overspecified)
# polynomial model set degree to use in simulated regressions
fit_degree_lo = 3
fit_degree_true = 6  # the actual signal curve will be a 2D polynomial series of this degree
fit_degree_hi = 24
fit_degree_vhi = 48

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 1.

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


def square_grid(min_val=-1.0, max_val=+1.0, nside=100, endpoint=True, flatten_order="C"):
    """Returns numpy arrays x0, x1 containing the coordinates of a square grid,
    symmetric with respect to the line x0=x1, defined by input min and max
    coordinate values.
    """
    xvals = np.linspace(min_val, max_val, num=nx, endpoint=True)
    x0, x1 = np.meshgrid(xvals, xvals)
    if flatten_order is not None:
        x0 = x0.flatten(order=flatten_order)
        x1 = x1.flatten(order=flatten_order)
    return x0, x1


def chebyshev_design_matrix(x0, x1, degree):
    """Returns the Chebyshev polynomial design matrix up to input degree for two
    independent coordinates x0, x1
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
    x0, x1 = square_grid(
        min_val=x0x1_min, max_val=x0x1_max, nside=nx, endpoint=True, flatten_order="C")

    # Design matrices
    design_lo, design_true, design_hi, design_vhi = tuple(
        chebyshev_design_matrix(x0, x1, degree=_deg)
        for _deg in (fit_degree_lo, fit_degree_true, fit_degree_hi, fit_degree_vhi)
    )

    # Build the true / ideal 2D contour, plot and save
    ctrue = np.random.randn(design_true.shape[-1]) * coeff_signal_to_noise
    ztrue = (np.matmul(design_true, ctrue)).reshape((nx, nx), order="C")

    plot_image(
        ztrue, "Ideal model", filename=os.path.join(outdir, "ideal_"+tstmp+".png"), show=True)
    output["ctrue"] = ctrue
    output["ztrue"] = ztrue

    # Add the random noise to generate the dataset, plot and save
    zdata = ztrue + noise_sigma * np.random.randn(*ztrue.shape)

    plot_image(zdata, "Data", filename=os.path.join(outdir, "data_"+tstmp+".png"), show=True)
    output["zdata"] = zdata

    # Perform too low, matching, too high, and very much too high degree regressions on data
    zflat = zdata.flatten(order="C")
    predictions = []
    for _design_matrix in (design_lo, design_true, design_hi, design_vhi):

        _coeffs = np.linalg.lstsq(_design_matrix, zflat, rcond=None)[0].T
        _prediction = _design_matrix.dot(_coeffs).reshape((nx, nx), order="C")
        predictions.append(_prediction)

    pred_lo, pred_true, pred_hi, pred_vhi = tuple(predictions)
    output["pred_lo"] = pred_lo
    output["pred_true"] = pred_true
    output["pred_hi"] = pred_hi
    output["pred_vhi"] = pred_vhi

    # Calculate and plot residuals
    rlo = zdata - pred_lo
    plot_image(
        rlo,
        "Low degree polynomial residuals",
        filename=os.path.join(outdir, "lo_"+tstmp+".png"),
        clim=CLIM,
    )
    rtrue = zdata - pred_true
    plot_image(
        rtrue,
        "Matching degree polynomial residuals",
        filename=os.path.join(outdir, "matching_"+tstmp+".png"),
        clim=CLIM,
    )
    rhi = zdata - pred_hi
    plot_image(
        rhi,
        "High degree polynomial residuals",
        filename=os.path.join(outdir, "hi_"+tstmp+".png"),
        clim=CLIM,
    )
    rvhi = zdata - pred_vhi
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
