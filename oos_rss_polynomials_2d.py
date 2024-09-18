"""
oos_rss_polynomials_2d.py
=========================

Analysis of out-of-sample (OOS) residual sums of squares (RSS) of regression
predictions from fitting_polynomials_2d.py, described in the paper "No patterns
in regression residuals."
"""

import os
import functools
import multiprocessing
import pickle

import numpy as np

import fitting_polynomials_2d
from further_analysis_polynomials_2d import DEGREES, DEGREE_STRS


# Parameters
# ==========

# LaTeX display strings for supported NRUNS options
NRUNS_STRS = {
    300: r"$300$",
    1000: r"$10^3$",
    10000: r"$10^4$",
    100000: r"$10^5$",
    1000000: r"$10^6$",
}

# Number of simulated regression datasets
NRUNS = 1000
if NRUNS not in NRUNS_STRS:
    raise ValueError(f"User parameter NRUNS must be one of {set(NRUNS_STRS.keys())}")

# Number of cores to use in multiprocessing the regresssion - I find that on modern python
# environments a number rather less than the number of actual cores on your machine (6 for my
# laptop) works best, perhaps due to some under-the-hood parallelization
NCORES = 2

# Gather module scope degrees into an array for convenience, use values from fitting_polynomials_2d
degree_values = (
    fitting_polynomials_2d.fit_degree_lo,
    fitting_polynomials_2d.fit_degree_true,
    fitting_polynomials_2d.fit_degree_hi,
    fitting_polynomials_2d.fit_degree_vhi,
)
degree_titles = {
    _d: DEGREE_STRS[_d].title()+" (G="+str(_dv)+")" for _d, _dv in zip(DEGREES, degree_values)
}
