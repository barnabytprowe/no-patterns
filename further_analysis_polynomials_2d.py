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
DEGREES = ("lo", "true", "hi")
DEGREE_STRS = {"lo": "low", "true": "matching", "hi": "high"}  # used in display


# Functions
# =========

def pathfile(timestamp, projdir=PROJDIR):
    """Returns the folder and output pickle filename corresponding to a run
    of fitting_polynomials_2d.py for the given timestamp.
    """
    _tfolder = os.path.join(projdir, timestamp)
    if not os.path.isdir(_tfolder):
        raise FileNotFoundError(_tfolder)

    return _tfolder, os.path.join(_tfolder, f"output_{timestamp}.pickle")


def get_stats(timestamp):
    """Calculates and returns key stats from a fitting_polynomials_2d.py output
    pickle file for the given timestamp.
    """
    _tfolder, _tsfile = pathfile(timestamp, projdir=PROJDIR)
    print(f"Building stats from {_tsfile}")
    with open(_tsfile, "rb") as fin:
        data = pickle.load(fin)

    rstats = {}
    for _degree in DEGREES:

        _res = data[f"r{_degree}"]
        rstats[_degree] = {
            "file": _tsfile,
            "timestamp": timestamp,
            "ctrue": data["ctrue"],
            "ztrue": data["ztrue"],
            "zdata": data["zdata"],
            "residuals": _res,
            "RSS": np.sum(_res**2),
            "rho": np.fft.ifft2(np.abs(np.fft.fft2(_res))**2).real / np.product(_res.shape),
            "cps": np.abs(np.fft.fft2(_res))**2,
        }

    return {_degree: rstats[_degree] for _degree in DEGREES}


def report_stats(timestamp):
    """Calls get_stats for an input timestamp, storing the returned dict to yaml
    format output in the timestamp folder.
    """
    _tfolder, _tsfile = pathfile(timestamp, projdir=PROJDIR)
    stats = get_stats(timestamp)
    print("RSS (lo, true, hi):")
    print(tuple(stats[_degree]["RSS"] for _degree in ("lo", "true", "hi")))

    statfile = os.path.join(_tfolder, f"stats_{timestamp}.yaml")
    print(f"Writing to {statfile}")
    with open(statfile, "w") as fout:
        yaml.dump(stats, fout)

    return stats


def plot_shuffled_residuals(stats, rng=None):
    """Generates and saves (into the same folder as the stats) a randomly-shuffled
    image of the all the residuals from an input stats dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _degree in DEGREES:

        timestamp = stats[_degree]["timestamp"]
        _tfolder, _ = pathfile(timestamp, projdir=PROJDIR)
        res = stats[_degree]["residuals"]
        # re-order residuals using shuffle
        res_flattened = res.flatten()
        rng.shuffle(res_flattened)
        shuffled_res = res_flattened.reshape(res.shape)
        # plot
        fig = plt.figure(figsize=FIGSIZE)
        plt.pcolor(shuffled_res, cmap=CMAP); plt.colorbar()
        plt.clim(CLIM)
        title_str = f"Shuffled {DEGREE_STRS[_degree]} degree polynomial residuals"
        plt.title(title_str, size=TITLE_SIZE)
        plt.tight_layout()
        if _degree == "true":  # annoyingly fitting_polynomials_2d saved images down as matching_*
            plt.savefig(os.path.join(_tfolder, f"matching_shuffled_{timestamp}.png"))
        else:
            plt.savefig(os.path.join(_tfolder, f"{_degree}_shuffled_{timestamp}.png"))
        plt.close(fig)


def plot_histogram_residuals(stats):
    """Generates and saves (into the same folder as the stats) a histogram of
    residual values from an input stats dict.
    """
    for _degree in DEGREES:

        plt.hist(stats[_degree]["residuals"].flatten(), bins=24, range=[-4, 4], color="Gray")
        plt.ylim(0, 50)
        plt.ylabel("Counts")
        plt.xlabel("Residual value")
        plt.grid()
        title_str = f"Histogram of {DEGREE_STRS[_degree]} degree polynomial residuals"
        plt.title(title_str, size=TITLE_SIZE)
        plt.tight_layout()
        plt.show()


# Main script
# ===========

if __name__ == "__main__":

    # Loop through timestamps and report stats into folders
    rng = np.random.default_rng()
    for _timestamp in TIMESTAMPS:

        stats = report_stats(_timestamp)
        plot_shuffled_residuals(stats, rng=rng)
        plot_histogram_residuals(stats)
