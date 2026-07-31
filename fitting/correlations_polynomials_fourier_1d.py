"""
no_patterns/fitting/correlations_polynomials_fourier_1d.py
==========================================================

Performs a large sample analysis of average correlations in residuals from
regression of in one dimension, described in the paper "No patterns in
regression residuals," with Chebyshev polynomial and Fourier series models.
Correlation functions and correlation matrices for residuals, for data ordered
along the abcissa, are output.

Uses the regression simulation parameters and functions defined in
fitting/polynomials_fourier_1d.py
"""

import functools
import multiprocessing
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import polynomials_fourier_1d
from polynomials_2d import PLTDIR
from polynomials_fourier_1d import (
    sample_spectrum,
    circular_acf,
    unbiased_acf,
    nhalf_dft,
    SUPPORTED_CURVE_FAMILIES,
    FIT_DEGREES,
    XARRS,
    NOISE_SIGMA,
    COEFF_SIGNAL_TO_NOISE,
    FIT_DISPLAY,
    CURVE_FAMILY_DISPLAY,
    LABEL_SIZE,
    TITLE_SIZE,
    OUTFILE_EXTENSIONS,
)


# Parameters
# ==========

# Number of simulated regression, out-of-sample datasets
NRUNS = 10**5

# Number of cores to use in multiprocessing the regresssion - I find that on modern python
# environments a number rather fewer than the number of actual cores on your machine (6 for my
# laptop) works best, perhaps due to some under-the-hood parallelization
NCORES = 2

# Pickle cache file location setup
PKLDIR = os.path.join(".", "pickles")
if not os.path.isdir(PKLDIR):
    os.mkdir(PKLDIR)
PICKLE_CACHE = os.path.join(PKLDIR, f"correlations_regressions_1d_n{NRUNS}.pkl")
CLOBBER = False  # overwrite any existing pickle cache

# Parameters for example matrix display using imshow
CMAP = "magma" # "gray_r"
VMIN = -1.
VMAX = 1.
ZOOM_NDIM = 10

# Output folder structure: project dir
PROJDIR = os.path.join(PLTDIR, "correlations_polynomials_fourier_1d")
os.makedirs(PROJDIR, exist_ok=True)


def _fit_predict(data, design_matrix):
    """Perform OLS regression on input data using the input design_matrix,
    returning predictions
    """
    coeffs = np.linalg.lstsq(design_matrix, data, rcond=None)[0].T
    return design_matrix.dot(coeffs)


def build_regression_sample(
    rng,
    nruns=NRUNS,
    fit_degrees=FIT_DEGREES,
    xarrs=XARRS,
    noise_sigma=NOISE_SIGMA,
    coeff_signal_to_noise=COEFF_SIGNAL_TO_NOISE,
):
    """Run full large sample analysis and return results in a dictionary, keyed
    by the curve family keys in the second layer of the fit_degrees, which must
    be a subset of {"cheb", "sinu"}.
    """
    design_matrices = polynomials_fourier_1d.features(xarrs=xarrs, fit_degrees=fit_degrees)

    output = {_fam: {} for _fam in xarrs}
    # Ideal model coefficients and corresponding images on the coordinate grid
    for _fam in xarrs:
        print(f"Generating {_fam} ideal model coefficients")
        output[_fam]["ctrue"] = rng.normal(
            loc=0.,
            scale=coeff_signal_to_noise * noise_sigma,
            size=(nruns, design_matrices[_fam]["true"].shape[-1]),
        )
        output[_fam]["ytrue"] = (
            np.matmul(design_matrices[_fam]["true"], output[_fam]["ctrue"].T).T
        ).reshape((nruns, len(xarrs[_fam])), order="C")

        # Generate the errors we will add to create simulated data
        print(f"Generating {_fam} errors, data")
        output[_fam]["errors"] = rng.normal(
            loc=0., scale=noise_sigma, size=(nruns, len(xarrs[_fam]))
        )
        output[_fam]["ydata"] = output[_fam]["ytrue"] + output[_fam]["errors"]

        # Perform too low, matching, too high, and very much too high degree regressions on data
        for _d in fit_degrees:
            output[_fam][_d] = {}
            _design_matrix = design_matrices[_fam][_d]
            _pfunc = functools.partial(_fit_predict, design_matrix=_design_matrix)
            print(f"Regressing {nruns} {_d} {_fam} runs using {NCORES=}")
            with multiprocessing.Pool(NCORES) as p:
                output[_fam][_d]["predictions"] = np.asarray(
                    p.map(_pfunc, [_zf for _zf in output[_fam]["ydata"]]), dtype=float
                )
            output[_fam][_d]["residuals"] = output[_fam]["ydata"] - output[_fam][_d]["predictions"]

    return output


def build_spectra_acfs(
    regressions,
    families=SUPPORTED_CURVE_FAMILIES,
    degree_labels=tuple(FIT_DEGREES),
):
    """
    """
    sample_spectra = {_family: {} for _family in families}
    circular_acfs = {_family: {} for _family in families}
    unbiased_acfs = {_family: {} for _family in families}
    for _family in families:
        for _degree_label in degree_labels:
            _residuals = regressions[_family][_degree_label]["residuals"]
            print(f"Running spectral analysis: {_family}, {_degree_label}")
            _ss = sample_spectrum(_residuals)
            sample_spectra[_family][_degree_label] = _ss
            circular_acfs[_family][_degree_label] = circular_acf(
                _residuals, real_sample_spectrum=_ss
            )
            unbiased_acfs[_family][_degree_label] = unbiased_acf(_residuals)

    return sample_spectra, circular_acfs, unbiased_acfs


def symmetric_extend(nfull, yhalf):
    """Symmetrically extends yhalf, an ndarray of trailing dimension
    nhalf_dft(nfull), to int nfull.

    Suited for when yhalf comes from symmetric spectral analysis output such as:
    sample_spectra, circular_acfs, unbiased_acfs
    """
    nhalf = nhalf_dft(nfull)
    assert yhalf.shape[-1] == nhalf

    # construct full shape vector and allocate output yfull
    full_shape = list(yhalf.shape)
    full_shape[-1] = nfull
    yfull = np.zeros(full_shape, dtype=float)

    # allocate first ~half of nfull, assigning yhalf
    yfull[..., :yhalf.shape[-1]] = yhalf.copy()

    # second half has a slightly different treatment based on the parity of nfull
    neven = ((nfull % 2) == 0)
    if neven:
        yhalf_complement = yhalf[..., 1:yhalf.shape[-1]-1][..., ::-1].copy()
        assert yfull[..., yhalf.shape[-1]:].shape[-1] == yhalf_complement.shape[-1]
    else:
        yhalf_complement = yhalf[..., 1:yhalf.shape[-1]][..., ::-1].copy()
        assert yfull[..., yhalf.shape[-1]:].shape[-1] == yhalf_complement.shape[-1]
        raise NotImplementedError("odd nfull not tested")
    yfull[..., yhalf.shape[-1]:] = yhalf_complement

    return yfull


# Main script
# ===========

if __name__ == "__main__":

    rng = np.random.default_rng()

    if not CLOBBER and os.path.isfile(PICKLE_CACHE):
        print(f"Loading from {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "rb") as funit:
            regressions = pickle.load(funit)
    else:
        t0 = time.time()
        regressions = build_regression_sample(
            rng=rng,
            fit_degrees=FIT_DEGREES,
            xarrs=XARRS,
            noise_sigma=NOISE_SIGMA,
            coeff_signal_to_noise=COEFF_SIGNAL_TO_NOISE,
        )
        t1 = time.time()
        print(f"Wall time: {(t1 - t0):.2f}s")
        print(f"Saving results to {PICKLE_CACHE=}")
        with open(PICKLE_CACHE, "wb") as fout:
            pickle.dump(regressions, fout)

    sample_spectra, circular_acfs, unbiased_acfs = build_spectra_acfs(
        regressions, families=SUPPORTED_CURVE_FAMILIES, degree_labels=tuple(FIT_DEGREES)
    )

    # Calculate mean values over the NRUNS samples
    mean_unbiased_acfs = {
        _family: {
            _degree_label: unbiased_acfs[_family][_degree_label].mean(axis=0)
            for _degree_label in FIT_DEGREES
        }
        for _family in SUPPORTED_CURVE_FAMILIES
    }
    mean_circular_acfs = {
        _family: {
            _degree_label: circular_acfs[_family][_degree_label].mean(axis=0)
            for _degree_label in FIT_DEGREES
        }
        for _family in SUPPORTED_CURVE_FAMILIES
    }
    # stderr_unbiased_acfs = {
    #     _family: {
    #         _degree_label: unbiased_acfs[_family][_degree_label].std(axis=0) / np.sqrt(NRUNS)
    #         for _degree_label in FIT_DEGREES
    #     }
    #     for _family in SUPPORTED_CURVE_FAMILIES
    # }
    # stderr_circular_acfs = {
    #     _family: {
    #         _degree_label: circular_acfs[_family][_degree_label].std(axis=0) / np.sqrt(NRUNS)
    #         for _degree_label in FIT_DEGREES
    #     }
    #     for _family in SUPPORTED_CURVE_FAMILIES
    # }

    # Plot mean acfs, both "unbiased" and circular
    linestyles = ["--", "-", "-.", ":"]
    colors = ["red", "k", "blue", "purple"]
    for _circularity, _acfs_dict in (("", mean_unbiased_acfs), (" circular", mean_circular_acfs)):
        for _family in SUPPORTED_CURVE_FAMILIES:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.set_title(
                (
                    f"Mean {CURVE_FAMILY_DISPLAY[_family].lower()} regression residual"
                    f"{_circularity} autocorrelation functions from {NRUNS} runs"
                ),
                size=TITLE_SIZE,
            )
            for i, _degree_label in enumerate(FIT_DEGREES):
                ax.plot(
                    _acfs_dict[_family][_degree_label] if _circularity == "" else (
                        symmetric_extend(len(XARRS[_family]), _acfs_dict[_family][_degree_label])
                    ),
                    color=colors[i],
                    ls=linestyles[i],
                    label=FIT_DISPLAY[_degree_label],
                )

            ax.set_xlabel(r"Lag $\ell$", size=LABEL_SIZE)
            ax.set_ylabel(r"$\left. r[\ell] ~ \middle/ ~ r[0] \right. $", size=LABEL_SIZE)
            ax.legend()
            ax.grid()
            fig.tight_layout()
            for _suffix in OUTFILE_EXTENSIONS:
                _outfile = os.path.join(
                    PROJDIR, f"mean{_circularity.replace(' ', '_')}_acf_{_family}_n{NRUNS}{_suffix}"
                )
                print(f"Saving to {_outfile}")
                fig.savefig(_outfile)

            plt.show()

    for _family in ("cheb",):  # SUPPORTED_CURVE_FAMILIES:
        for _degree_label in FIT_DEGREES:
            # Plot the symmetric Toeplitz unbiased ACF
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(
                scipy.linalg.toeplitz(mean_unbiased_acfs[_family][_degree_label]),
                cmap=CMAP,
                vmin=VMIN,
                vmax=VMAX,
            )
            cbar = fig.colorbar(im)
            nlog10_str = str(int(np.log10(NRUNS)))
            ax.set_title(
                (
                    f"Mean unbiased ACF from\n{FIT_DISPLAY[_degree_label].lower()} "
                    f"{CURVE_FAMILY_DISPLAY[_family]} regressions"
                ),
                size=12,
            )
            ax.tick_params(axis="both", labelsize=10)
            cbar.ax.tick_params(axis="both", labelsize=10)
            fig.tight_layout()
            plt.show()

            # Plot the symmetric circulant circular ACF
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(
                scipy.linalg.circulant(
                    symmetric_extend(
                        len(XARRS[_family]), mean_circular_acfs[_family][_degree_label]
                    )
                ),
                cmap=CMAP,
                vmin=VMIN,
                vmax=VMAX,
            )
            cbar = fig.colorbar(im)
            ax.set_title(
                (
                    f"Mean circular ACF from\n{FIT_DISPLAY[_degree_label].lower()} "
                    f"{CURVE_FAMILY_DISPLAY[_family]} regressions"
                ),
                size=12,
            )
            ax.tick_params(axis="both", labelsize=10)
            cbar.ax.tick_params(axis="both", labelsize=10)
            fig.tight_layout()
            plt.show()
