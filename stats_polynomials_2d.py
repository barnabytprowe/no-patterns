"""
stats_polynomials_2d.py
=======================

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
from fitting_polynomials_2d import CMAP
from fitting_polynomials_2d import TITLE_SIZE


# Params
# ------

TIMESTAMPS = [os.path.basename(_p) for _p in glob.glob(os.path.join(PROJDIR, "*-*-*T*"))]


# Functions
# ---------

def pathfile(timestamp, projdir=PROJDIR):
    """Returns the folder and output pickle filename corresponding to a run
    of fitting_polynomials_2d.py for the given timestamp.
    """
    _tfolder = os.path.join(projdir, timestamp)
    if not os.path.isdir(_tfolder):
        raise FileNotFoundError(_tfolder)

    return _tfolder, os.path.join(_tfolder, f"output_{timestamp}.pickle")


def get_stats(timestamp, rng=None):
    """Retrieves, calculates and returns key stats from a
    fitting_polynomials_2d.py output pickle file for a given timestamp, and
    generates randomized (shuffled) versions of the residual plots.

    Also generages a PNG image of a shuffled (by input rng if given) residual
    grid.
    """
    if rng is None:
        rng = np.random.default_rng()

    _tfolder, _tsfile = pathfile(timestamp, projdir=PROJDIR)
    print(f"Building stats from {_tsfile}")
    with open(_tsfile, "rb") as fin:
        data = pickle.load(fin)

    rstats = {}
    order_strs = {"lo": "low", "true": "matching", "hi": "high"}
    for _order in ("lo", "true", "hi"):

        _res = data[f"r{_order}"]
        rstats[_order] = {
            "file": _tsfile,
            "ctrue": data["ctrue"],
            "ztrue": data["ztrue"],
            "zdata": data["zdata"],
            "residuals": _res,
            "RSS": np.sum(_res**2),
            "rho": np.fft.ifft2(np.abs(np.fft.fft2(_res))**2).real / np.product(_res.shape),
            "cps": np.abs(np.fft.fft2(_res))**2,
        }
        _res_flattened = _res.flatten()
        rng.shuffle(_res_flattened)
        _shuffled_res = _res_flattened.reshape(_res.shape)
        fig = plt.figure(figsize=FIGSIZE)
        plt.pcolor(_shuffled_res, cmap=CMAP); plt.colorbar()
        plt.clim([-2.5, 2.5])
        _title_str = f"Shuffled {order_strs[_order]} order polynomial residuals"
        plt.title(_title_str, size=TITLE_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(_tfolder, f"{_order}_shuffled_{timestamp}.png"))
        plt.close(fig)

    return tuple(rstats[_o] for _o in ("lo", "true", "hi"))


def report_stats(timestamp, rng=None):
    """Gets stats for a timestamp, generates additional plots, and stores all
    useful data to yaml format output in the timstamp folder.
    """
    _tfolder, _tsfile = pathfile(timestamp, projdir=PROJDIR)
    _lo, _true, _hi = get_stats(timestamp, rng=rng)
    stats = {"lo": _lo, "true": _true, "hi": _hi}
    print("RSS (lo, true, hi):")
    print(tuple(stats[_order]["RSS"] for _order in ("lo", "true", "hi")))
    statfile = os.path.join(_tfolder, f"stats_{timestamp}.yaml")
    print(f"Writing to {statfile}")
    with open(statfile, "w") as fout:
        yaml.dump(stats, fout)
    return


if __name__ == "__main__":

    # Loop through timestamps and report stats into folders
    rng = np.random.default_rng()
    for _timestamp in TIMESTAMPS:

        report_stats(_timestamp, rng=rng)
