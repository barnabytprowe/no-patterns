"""
symmetric_toeplitz_entropy.py
=============================
Basic functions and charts for exploring the differential entropy of
multivariate Gaussian distributions with Toeplitz and Circulant covariance
matrices.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


# Params for main script & plotting
# ---------------------------------

NDIM = 100
DECAY_FACTORS = np.linspace(0., 0.30, num=301)
DECAY_FACTORS_EXTENDED = np.linspace(0., 0.95, num=951)

# Parameters for example matrix display using imshow
EXAMPLE_DECAY_FACTOR = 0.3
CMAP = "gray_r"
VMIN = -EXAMPLE_DECAY_FACTOR
VMAX = 1.
ZOOM_NDIM = 10

# Output
outdir = os.path.join(".", "plots")
outfile = os.path.join(outdir, "determinants_toeplitz.pdf")
outfile_extended = os.path.join(outdir, "determinants_toeplitz_extended.pdf")

toeplitz_matrix_example_prefix = "toeplitz_example_"
toeplitz_matrix_example_zoom_prefix = "toeplitz_example_zoom_"
circulant_matrix_example_prefix = "circulant_example_"


# Functions
# ---------

def exponential_decay(ndim, decay_factor=0., circular=False, oscillatory=False):
    """Returns 1D float array of length ndim containing values that decrease by
    a constant decay_factor with each increasing index i, where array[0] = 1,
    with optional circular symmetry.
    """
    _rarr = np.zeros(ndim, dtype=float)
    if not circular:
        _rarr[0] = 1.
        if decay_factor > 0.:
            _rarr[1:] = np.full(ndim - 1, decay_factor, dtype=float).cumprod()
        if oscillatory:
            _rarr *= (-1.)**(np.arange(ndim) % 2)

    else:
        _first_half = exponential_decay(
            1 + ndim//2, decay_factor=decay_factor, circular=False, oscillatory=oscillatory)
        _rarr[:(1 + ndim//2)] = _first_half
        _rarr[(1 + ndim//2):] = _first_half[1:(1 + ndim)//2][::-1]

    return _rarr


def exponential_toeplitz_corr(ndim, decay_factor=0., circulant=False, oscillatory=False):
    """Returns Toeplitz correlation matrix as a 2D float array of shape
    (ndim, ndim) with the specified exponential decay in band structure.
    """
    if circulant:
        return scipy.linalg.circulant(
            exponential_decay(
                ndim, decay_factor=decay_factor, circular=True, oscillatory=oscillatory)
        )

    else:
        return scipy.linalg.toeplitz(
            exponential_decay(
                ndim, decay_factor=decay_factor, circular=False, oscillatory=oscillatory)
        )


if __name__ == "__main__":

    # Make plots of Toeplitz matrix, Toeplitz matrix inset, and circulant approximation
    oscstr = {True: "oscillatory", False: "non-oscillatory"}
    for _osc in (True, False):

        toeplitz_example = exponential_toeplitz_corr(
            NDIM, EXAMPLE_DECAY_FACTOR, circulant=False, oscillatory=_osc)
        toeplitz_example_zoom = toeplitz_example[:ZOOM_NDIM, :ZOOM_NDIM]
        circulant_example = exponential_toeplitz_corr(
            NDIM, EXAMPLE_DECAY_FACTOR, circulant=True, oscillatory=_osc)

        # First show the Toeplitz
        fig, ax = plt.subplots(figsize=(16, 12))
        im = ax.imshow(toeplitz_example, cmap=CMAP, vmin=VMIN, vmax=VMAX)
        cbar = fig.colorbar(im)
        ax.set_title(
            r"$\mathbf{P}$ with decay factor $\lambda$ = "+str(EXAMPLE_DECAY_FACTOR)+
            " ("+oscstr[_osc]+")",
            size=20,
        )
        ax.tick_params(axis="both", labelsize=16)
        cbar.ax.tick_params(axis="both", labelsize=16)
        fig.tight_layout()
        _outfile = os.path.join(
            outdir,
            toeplitz_matrix_example_prefix+oscstr[_osc]+f"_lambda_{EXAMPLE_DECAY_FACTOR}.pdf"
        )
        print(f"Saving figure to {_outfile}")
        fig.savefig(_outfile)

        # Then the circulant approximation to the Toeplitz
        fig, ax = plt.subplots(figsize=(16, 12))
        im = ax.imshow(circulant_example, cmap=CMAP, vmin=VMIN, vmax=VMAX)
        cbar = fig.colorbar(im)
        ax.set_title(
            r"$\mathbf{P}$ with decay factor $\lambda$ = "+str(EXAMPLE_DECAY_FACTOR)+
            " ("+oscstr[_osc]+", circulant approximation)",
            size=20,
        )
        ax.tick_params(axis="both", labelsize=16)
        cbar.ax.tick_params(axis="both", labelsize=16)
        fig.tight_layout()
        _outfile = os.path.join(
            outdir,
            circulant_matrix_example_prefix+oscstr[_osc]+f"_lambda_{EXAMPLE_DECAY_FACTOR}.pdf"
        )
        print(f"Saving figure to {_outfile}")
        fig.savefig(_outfile)

        # Now we generate the chart of the zoomed in diagonal region (only do for Toeplitz)
        fig, ax = plt.subplots()
        im = ax.imshow(toeplitz_example_zoom, cmap=CMAP, vmin=VMIN, vmax=VMAX)
        cbar = fig.colorbar(im)
        ax.set_title(
            r"Diagonal block from $\mathbf{P}$ with decay factor $\lambda$ = "+
            str(EXAMPLE_DECAY_FACTOR)+" ("+oscstr[_osc]+")",
        )
        fig.tight_layout()
        _outfile = os.path.join(
            outdir,
            toeplitz_matrix_example_zoom_prefix+oscstr[_osc]+f"_lambda_{EXAMPLE_DECAY_FACTOR}.pdf"
        )
        print(f"Saving figure to {_outfile}")
        fig.savefig(_outfile)

    # Plot determinants of the correlation matrix (scales with entropy) as function of decay factor
    dets_toeplitz = {}
    dets_circulant = {}
    for i, _decay_factors in enumerate((DECAY_FACTORS, DECAY_FACTORS_EXTENDED)):

        # Calculate Toeplitz and circulant approximation determinants via their eigenvalue products
        dets_toeplitz[i] = np.asarray(
            [
                np.product(
                    np.linalg.eigvalsh(
                        exponential_toeplitz_corr(NDIM, _dfac, circulant=False, oscillatory=False)
                    )
                )
                for _dfac in _decay_factors
            ],
            dtype=float,
        )
        dets_circulant[i] = np.asarray(
            [
                np.product(
                    np.linalg.eigvalsh(
                        exponential_toeplitz_corr(NDIM, _dfac, circulant=True, oscillatory=False)
                    )
                )
                for _dfac in _decay_factors
            ],
            dtype=float,
        )

        fig, ax = plt.subplots()
        ax.plot(_decay_factors, dets_toeplitz[i], "k-")
        ax.plot(_decay_factors, dets_circulant[i], "k--", label="Circulant approximation")
        ax.set_yscale("log")
        if i == 1:  # extended decay factors case need to give pyplot help with ticks
            ax.set_yticks(10.**(-10. * np.arange(11)[::-1]))  # uncomment for extended plots
        ax.grid(True, which="both")
        ax.set_title(r"$\det{\mathbf{P}}$")
        ax.set_xlabel(r"Decay factor $\lambda$")
        ax.legend()
        fig.tight_layout()
        if i == 1:
            _outfile = outfile_extended
        else:
            _outfile = outfile
        print(f"Saving figure to {_outfile}")
        fig.savefig(_outfile)
