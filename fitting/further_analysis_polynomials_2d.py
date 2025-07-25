"""
no_patterns/fitting/further_analysis_polynomials_2d.py
======================================================

Further analysis and additional plots of regressions performed by
fitting/polynomials_2d.py, described in the paper "No patterns in regression
residuals."

Uses glob to locate all timestamped output generated so far by
fitting/polynomials_2d.py, then generates histograms (including shuffled) of
residuals for each regression simulation suite, along with images of predictions
and other statistics.
"""

import collections
import glob
import os
import pickle
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import polynomials_2d
from polynomials_2d import PROJDIR, FIGSIZE, CLIM, CMAP, TITLE_SIZE


# Parameters
# ==========

# Gather the timestamps of all regression results generated so far with the main script in
# fitting_polynomials_2d.py
TIMESTAMPS = [os.path.basename(_p) for _p in glob.glob(os.path.join(PROJDIR, "*-*-*T*"))]

# Degrees of the polynomial series model sets used in the regressions
DEGREES = ("lo", "true", "hi", "vhi")
DEGREE_STRS = {"lo": "low", "true": "matching", "hi": "high", "vhi": "very high"}  # used in display

HIST_RANGE = [-8, 8]
HIST_NBINS = 80
HIST_YLIM = [0, 100]

HIST_LABEL_FONTSIZE = "x-large"
HIST_TICK_FONTSIZE = "x-large"

# Number of runs of further data generation with which to estimate the RMS error of prediction
NRUNS = 1000


# Functions
# =========

def pathfile(timestamp, projdir=PROJDIR):
    """Returns the folder and output pickle filename corresponding to a run
    of fitting/polynomials_2d.py for the given timestamp.
    """
    tfolder = os.path.join(projdir, timestamp)
    if not os.path.isdir(tfolder):
        raise FileNotFoundError(tfolder)

    return tfolder, os.path.join(tfolder, f"output_{timestamp}.pickle")


def get_data_stats(timestamp):
    """Gets data, then calculates and returns some second order stats from a
    fitting/polynomials_2d.py output pickle file for the given timestamp.
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

        _res = stats[_degree]["residuals"]
        # re-order residuals using shuffle
        _res_flattened = _res.flatten(order="C")
        rng.shuffle(_res_flattened)
        _shuffled_res = _res_flattened.reshape(_res.shape, order="C")

        _timestamp = stats[_degree]["timestamp"]
        if _degree == "true":  # annoyingly fitting/polynomials_2d saved images down as matching_*
            _outfile = os.path.join(
                stats[_degree]["folder"], f"matching_shuffled_{_timestamp}.png")
        else:
            _outfile = os.path.join(
                stats[_degree]["folder"], f"{_degree}_shuffled_{_timestamp}.png")

        polynomials_2d.plot_image(
            _shuffled_res,
            title=f"Shuffled {DEGREE_STRS[_degree]} degree polynomial residuals",
            filename=_outfile,
            clim=CLIM,
            show=False,
            tick_stride=7
        )


def plot_histogram_residuals(stats):
    """Generates and saves (into the same folder as the stats) a histogram of
    residual values from an input stats dict.
    """
    for _degree in DEGREES:

        print(f"Plotting {_degree} histogram")
        fig = plt.figure(figsize=FIGSIZE)
        _r = stats[_degree]["residuals"].flatten()
        _msr = (_r**2).mean()

        plt.hist(_r, bins=HIST_NBINS, range=HIST_RANGE, color="Gray")
        if _degree != "vhi":
            plt.ylim(*HIST_YLIM)
        plt.ylabel("Counts", fontsize=HIST_LABEL_FONTSIZE)
        plt.xlabel("Residual value", fontsize=HIST_LABEL_FONTSIZE)
        plt.xticks(fontsize=HIST_TICK_FONTSIZE)
        plt.yticks(fontsize=HIST_TICK_FONTSIZE)
        plt.grid()
        title_str = f"Histogram: {DEGREE_STRS[_degree]} degree polynomial residuals"
        plt.title(title_str, size=TITLE_SIZE)
        plt.figtext(x=.65, y=.8, s=(r"RSS$/N$ = "+f"{_msr:5.3f}"), fontsize="xx-large")
        plt.tight_layout()

        timestamp = stats[_degree]["timestamp"]
        if _degree == "true":  # annoyingly fitting/polynomials_2d saved images down as matching_*
            outfile = os.path.join(stats[_degree]["folder"], f"hist_matching_{timestamp}.png")
        else:
            outfile = os.path.join(stats[_degree]["folder"], f"hist_{_degree}_{timestamp}.png")
        plt.savefig(outfile)
        plt.close(fig)
        _mse = (stats[_degree]["errors"].flatten()**2).mean()
        print(f"{_degree} mean square residual, error = {_msr}, {_mse}")


def plot_predictions(data):
    """Generates plots of predictions from fitting_polynomials_2d.py output
    and saves it into the same folder for the given timestamp.
    """

    timestamp = data["timestamp"]
    for _degree in DEGREES:

        print(f"Plotting {_degree} prediction")
        _prediction = data["pred_"+_degree]

        if _degree == "true":  # annoyingly fitting_polynomials_2d saved images down as matching_*
            _outfile = os.path.join(data["folder"], f"matching_prediction_{timestamp}.png")
        else:
            _outfile = os.path.join(data["folder"], f"{_degree}_prediction_{timestamp}.png")
        polynomials_2d.plot_image(
            _prediction,
            title=f"{DEGREE_STRS[_degree].title()} degree polynomial prediction",
            filename=_outfile,
            clim=None,
            show=False,
            tick_stride=7
        )


def mean_squared_cross_validation_residual(data, rng=None, nruns=NRUNS):
    """Calculates the mean squared cross validation residual, the difference
    between predictions and new datasets generated with the same ideal model
    but with new additive, iid errors.
    """
    if rng is None:
        rng = np.random.default_rng()

    timestamp = data["timestamp"]
    print(f"Generating {nruns} new datasets for {timestamp}")
    imshape = data["ztrue"].shape
    new_errors = rng.normal(
        loc=0., scale=polynomials_2d.noise_sigma, size=(nruns, imshape[0], imshape[1]))
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
        pd.DataFrame({"Mean": mean_msxvr, "Std": std_msxvr, "StdErr": std_msxvr / np.sqrt(nruns)}))
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
