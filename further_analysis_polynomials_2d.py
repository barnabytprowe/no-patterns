"""
further_analysis_polynomials_2d.py
==================================

Further analysis and additional plots of regressions performed by
fitting_polynomials_2d.py, described in the paper "No patterns in regression
residuals."
"""

import collections
import glob
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fitting_polynomials_2d
from fitting_polynomials_2d import PROJDIR
from fitting_polynomials_2d import FIGSIZE
from fitting_polynomials_2d import CLIM
from fitting_polynomials_2d import CMAP
from fitting_polynomials_2d import TITLE_SIZE


# Parameters
# ==========

# Gather the timestamps of all regression results generated so far with the main script in
# fitting_polynomials_2d.py
TIMESTAMPS = [os.path.basename(_p) for _p in glob.glob(os.path.join(PROJDIR, "*-*-*T*"))]

# Degrees of the polynomial series model sets used in the regressions
DEGREES = ("lo", "true", "hi", "vhi")
DEGREE_STRS = {"lo": "low", "true": "matching", "hi": "high", "vhi": "very high"}  # used in display

HIST_RANGE = [-8, 8]
HIST_NBINS = 32
HIST_YLIM = [0, 50]

HIST_LABEL_FONTSIZE = "x-large"
HIST_TICK_FONTSIZE = "x-large"

# Number of runs of further data generation with which to estimate the RMS error of prediction
NRUNS = 1000000


# Functions
# =========

def pathfile(timestamp, projdir=PROJDIR):
    """Returns the folder and output pickle filename corresponding to a run
    of fitting_polynomials_2d.py for the given timestamp.
    """
    tfolder = os.path.join(projdir, timestamp)
    if not os.path.isdir(tfolder):
        raise FileNotFoundError(tfolder)

    return tfolder, os.path.join(tfolder, f"output_{timestamp}.pickle")


def get_data_stats(timestamp):
    """Gets data, then calculates and returns some second order stats from a
    fitting_polynomials_2d.py output pickle file for the given timestamp.
    """
    tfolder, tsfile = pathfile(timestamp, projdir=PROJDIR)
    print(f"Building stats from {tsfile}")
    with open(tsfile, "rb") as fin:
        data = pickle.load(fin)

    # For later convenience, add path information to the data dict
    data["file"] = tsfile
    data["folder"] = tfolder
    data["timestamp"] = timestamp

    rstats = {}
    for _degree in DEGREES:

        _res = data[f"r{_degree}"]
        rstats[_degree] = {
            "file": tsfile,
            "folder": tfolder,
            "timestamp": timestamp,
            "ctrue": data["ctrue"],
            "ztrue": data["ztrue"],
            "zdata": data["zdata"],
            "errors": data["zdata"] - data["ztrue"],
            "residuals": _res,
            "RSS": np.sum(_res**2),
            "rho": np.fft.ifft2(np.abs(np.fft.fft2(_res))**2).real / np.product(_res.shape),
            "cps": np.abs(np.fft.fft2(_res))**2,
        }

    return data, {_degree: rstats[_degree] for _degree in DEGREES}


def report_stats(timestamp):
    """Calls get_data_stats for an input timestamp, storing the returned stats
    dict to yaml format output in the timestamp folder, and returning the
    (data, stats) tuple
    """
    data, stats = get_data_stats(timestamp)
    print("RSS (lo, true, hi, vhi):")
    print(tuple(stats[_degree]["RSS"] for _degree in DEGREES))

    statfile = os.path.join(data["folder"], f"stats_{timestamp}.yaml")
    print(f"Writing to {statfile}")
    with open(statfile, "w") as fout:
        yaml.dump(stats, fout)

    return data, stats


def plot_shuffled_residuals(stats, rng=None):
    """Generates and saves (into the same folder as the stats) a randomly-shuffled
    image of the all the residuals from an input stats dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _degree in DEGREES:

        res = stats[_degree]["residuals"]
        # re-order residuals using shuffle
        res_flattened = res.flatten()
        rng.shuffle(res_flattened)
        shuffled_res = res_flattened.reshape(res.shape)
        # plot
        fig = plt.figure(figsize=FIGSIZE)
        plt.pcolor(shuffled_res, cmap=CMAP)
        plt.colorbar()
        plt.clim(CLIM)
        title_str = f"Shuffled {DEGREE_STRS[_degree]} degree polynomial residuals"
        plt.title(title_str, size=TITLE_SIZE)
        plt.tight_layout()
        timestamp = stats[_degree]["timestamp"]
        if _degree == "true":  # annoyingly fitting_polynomials_2d saved images down as matching_*
            outfile = os.path.join(stats[_degree]["folder"], f"matching_shuffled_{timestamp}.png")
        else:
            outfile =  os.path.join(stats[_degree]["folder"], f"{_degree}_shuffled_{timestamp}.png")
        plt.savefig(outfile)
        plt.close(fig)


def plot_histogram_residuals(stats):
    """Generates and saves (into the same folder as the stats) a histogram of
    residual values from an input stats dict.
    """
    for _degree in DEGREES:

        print(f"Plotting {_degree} histogram")
        fig = plt.figure(figsize=FIGSIZE)
        _r = stats[_degree]["residuals"].flatten(s)
        _msr = (_r**2).mean()
        plt.hist(_r, bins=HIST_NBINS, range=HIST_RANGE, color="Gray")
        plt.ylim(*HIST_YLIM)
        plt.ylabel("Counts", fontsize=HIST_LABEL_FONTSIZE)
        plt.xlabel("Residual value", fontsize=HIST_LABEL_FONTSIZE)
        plt.xticks(fontsize=HIST_TICK_FONTSIZE)
        plt.yticks(fontsize=HIST_TICK_FONTSIZE)
        plt.grid()
        title_str = f"Histogram of {DEGREE_STRS[_degree]} degree polynomial residuals"
        plt.title(title_str, size=TITLE_SIZE)
        plt.figtext(x=.65, y=.8, s=(r"RSS$/N$ = "+f"{_msr:5.3f}"), fontsize="xx-large")
        plt.tight_layout()
        timestamp = stats[_degree]["timestamp"]
        if _degree == "true":  # annoyingly fitting_polynomials_2d saved images down as matching_*
            outfile = os.path.join(stats[_degree]["folder"], f"hist_matching_{timestamp}.png")
        else:
            outfile = os.path.join(stats[_degree]["folder"], f"hist_{_degree}_{timestamp}.png")
        plt.savefig(outfile)
        plt.close(fig)
        _mse =  (stats[_degree]["errors"].flatten()**2).mean()
        print(f"{_degree} mean square residual, error = {_msr}, {_mse}")


def plot_predictions(data):
    """Generates plots of predictions from fitting_polynomials_2d.py output
    and saves it into the same folder for the given timestamp.
    """
    for _degree in DEGREES:

        print(f"Plotting {_degree} prediction")
        pred = data["pred_"+_degree]
        # plot
        fig = plt.figure(figsize=FIGSIZE)
        plt.pcolor(pred, cmap=CMAP)
        plt.colorbar()
        title_str = f"{DEGREE_STRS[_degree].title()} degree polynomial prediction"
        plt.title(title_str, size=TITLE_SIZE)
        plt.tight_layout()
        timestamp = data["timestamp"]
        if _degree == "true":  # annoyingly fitting_polynomials_2d saved images down as matching_*
            plt.savefig(os.path.join(data["folder"], f"matching_prediction_{timestamp}.png"))
        else:
            plt.savefig(os.path.join(data["folder"], f"{_degree}_prediction_{timestamp}.png"))
        plt.close(fig)


def mean_squared_cross_validation_residual(data, rng=None):
    """Calculates the mean squared cross validation residual, the difference
    between predictions and new datasets generated with the same ideal model
    but with new additive, iid errors.
    """
    if rng is None:
        rng = np.random.default_rng()

    timestamp = data["timestamp"]
    print(f"Generating {NRUNS} new datasets for {timestamp}")
    imshape = data["ztrue"].shape
    new_errors = rng.normal(
        loc=0.,
        scale=fitting_polynomials_2d.noise_sigma,
        size=(NRUNS, imshape[0], imshape[1]),
    )
    new_datasets = data["ztrue"] + new_errors

    mean_squared_xvr = {}
    mean_msxvr = {}
    std_msxvr = {}
    for _degree in DEGREES:

        pred = data["pred_"+_degree]
        _ssquared_xvr = ((new_datasets - pred)**2).sum(axis=(-2, -1))
        mean_squared_xvr[_degree] = _ssquared_xvr / np.product(imshape)
        mean_msxvr[_degree] = np.mean(mean_squared_xvr[_degree])
        std_msxvr[_degree] = np.std(mean_squared_xvr[_degree])

    mean_msxvr = pd.Series(mean_msxvr)
    std_msxvr = pd.Series(std_msxvr)
    print(f"Mean Squared Cross-Validation Residual for {timestamp}:")
    print(
        pd.DataFrame({"Mean": mean_msxvr, "Std": std_msxvr, "StdErr": std_msxvr / np.sqrt(NRUNS)}))
    return mean_squared_xvr


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()
    for _timestamp in TIMESTAMPS:

        data, stats = report_stats(_timestamp)
        plot_shuffled_residuals(stats, rng=rng)
        plot_histogram_residuals(stats)
        plot_predictions(data)
        mean_squared_cross_validation_residual(data, rng=rng)
