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


# Params
# ------

DEFAULT_ROOTDIR = os.path.join(".", "plots")
TIMESTAMPS = [os.path.basename(_p) for _p in glob.glob(os.path.join(DEFAULT_ROOTDIR, "*-*-*T*"))]


# Functions
# ---------

def pathfile(timestamp, rootdir=DEFAULT_ROOTDIR):
    _tfolder = os.path.join(rootdir, timestamp)
    if not os.path.isdir(_tfolder):
        raise FileNotFoundError(_tfolder)

    return _tfolder, os.path.join(_tfolder, f"output_{timestamp}.pickle")


def get_stats(file):
    with open(file, "rb") as fin:
        data = pickle.load(fin)

    rstats = {}
    for _order in ("lo", "true", "hi"):

        _res = data[f"r{_order}"]
        rstats[_order] = {
            "file": file,
            "ctrue": data["ctrue"],
            "ztrue": data["ztrue"],
            "zdata": data["zdata"],
            "residuals": _res,
            "RSS": np.sum(_res**2),
            "rho": np.fft.ifft2(np.abs(np.fft.fft2(_res))**2).real / np.product(_res.shape),
            "cps": np.abs(np.fft.fft2(_res))**2,
        }

    return tuple(rstats[_o] for _o in ("lo", "true", "hi"))


def report_stats(timestamp):
    _tfolder, _tsfile = pathfile(timestamp, rootdir=DEFAULT_ROOTDIR)
    print(f"Building stats from {_tsfile}")
    _lo, _true, _hi = get_stats(_tsfile)
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
    for _timestamp in TIMESTAMPS:

        report_stats(_timestamp)
