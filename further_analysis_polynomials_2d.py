"""
further_analysis_polynomials_2d.py
==================================

A python script that performs further analysis and generates additional plots
based on the results of the regressions performed by fitting_polynomials_2d.py,
described in the paper "No patterns in regression residuals."
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
HIST_NBINS = 80
HIST_YLIM = [0, 50]


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
        plt.hist(
            stats[_degree]["residuals"].flatten(), bins=HIST_NBINS, range=HIST_RANGE, color="Gray")
        plt.ylim(*HIST_YLIM)
        plt.ylabel("Counts")
        plt.xlabel("Residual value")
        plt.grid()
        title_str = f"Histogram of {DEGREE_STRS[_degree]} degree polynomial residuals"
        plt.title(title_str, size=TITLE_SIZE)
        plt.tight_layout()
        timestamp = stats[_degree]["timestamp"]
        if _degree == "true":  # annoyingly fitting_polynomials_2d saved images down as matching_*
            outfile = os.path.join(stats[_degree]["folder"], f"hist_matching_{timestamp}.png")
        else:
            outfile = os.path.join(stats[_degree]["folder"], f"hist_{_degree}_{timestamp}.png")
        plt.savefig(outfile)
        plt.close(fig)


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


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()
    for _timestamp in TIMESTAMPS:

        data, stats = report_stats(_timestamp)
        plot_shuffled_residuals(stats, rng=rng)
        plot_histogram_residuals(stats)
