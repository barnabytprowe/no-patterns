"""
msep_analysis_polynomials_2d.py
===============================

Mean Squared Errors of Prediction analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing

import fitting_polynomials_2d


# Parameters
# ==========

NRUNS = 1000


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()

    # Design matrices
    features_lo = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_lo,
        order="C",
    )
    features_true = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_true,
        order="C",
    )
    features_hi = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_hi,
        order="C",
    )
    features_vhi = fitting_polynomials_2d.polynomial_design_matrix(
        square_dimension=fitting_polynomials_2d.side_dim,
        n_side=fitting_polynomials_2d.nx,
        degree=fitting_polynomials_2d.fit_degree_vhi,
        order="C",
    )

    # Ideal model coefficients and corresponding images on the coordinate grid
    ctrue = rng.normal(
        loc=0.,
        scale=fitting_polynomials_2d.coeff_signal_to_noise,
        size=(NRUNS, features_true.shape[-1]),
    )
    ztrue = (
        np.matmul(features_true, ctrue.T).T
    ).reshape(NRUNS, fitting_polynomials_2d.nx, fitting_polynomials_2d.nx, order="C")
