"""
fitting_polynomials_numeric_2d.py
=================================

Examples of regression of in two dimensions, described in the paper "No
patterns in regression residuals," illustrating underspecified, correctly
specified, and overspecified regression of randomly-generated polynomial
surfaces on a regular 2D grid.

Uses numeric gradient descent-based optimization schemes provided by pytorch
(e.g. batch gradient descent, Adam, L-BFGS) rather than solving for the exact
linear algebra solution.

Saves output from each simulated regression into a uniquely timestamped
subfolder of ./plots/polynomials_numeric_2d/.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import fitting_polynomials_2d
from fitting_polynomials_2d import (  # all parameters defined in core fitting module
    x0x1_min,
    x0x1_max,
    nx,
    fit_degree_lo,
    fit_degree_true,
    fit_degree_hi,
    fit_degree_vhi,
    coeff_signal_to_noise,
    noise_sigma,
)


# Parameters
# ==========

# Output folder structure
PLTDIR = os.path.join(".", "plots")
PROJDIR = os.path.join(PLTDIR, "polynomials_2d_numeric")

# Which under/overspecifications to run?
all_specifications = ("lo", "true", "hi", "vhi")
run_specifications = ("lo", "true", "hi", "vhi")

# Use existing timestamp?
tstamp_exists = False
if tstamp_exists:
    tstamp = ""
    tstamp_project_dir = os.path.join(PLTDIR, "polynomials_2d")

# Pytorch settings
n_epochs = int(1e5)
iter_stride = 1000
loss_function = torch.nn.MSELoss(reduction="mean")
gradient_descent_optim = {torch.optim.Adam: {"lr": 0.001, "betas": (0.99, 0.999), "eps": 1e-8}}
lbfgs_optim = {
    torch.optim.LBFGS: {
        "lr": 1,
        "max_iter": iter_stride,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-8,
        "tolerance_change": 1e-10,
    }
}
early_exit_rtol = 1e-11

# use graphics card if available
device = "cuda" if torch.cuda.is_available() else "cpu"


# Main script
# ===========

if __name__ == "__main__":

    if not tstamp_exists:
        # current timestamp
        tstmp = pd.Timestamp.now().isoformat().replace(":", "")
    outdir = fitting_polynomials_2d.build_output_folder_structure(tstmp, project_dir=PROJDIR)

    # Output dict - will be pickled
    output = {}

    # Prepare two independent variables on a grid
    x0, x1 = fitting_polynomials_2d.square_grid(
        min_val=x0x1_min, max_val=x0x1_max, nside=nx, endpoint=True, flatten_order="C")

    # Design matrices
    design_matrices = {
        _spec: fitting_polynomials_2d.chebyshev_design_matrix(x0, x1, degree=_deg)
        for _spec, _deg in zip(
            all_specifications, (fit_degree_lo, fit_degree_true, fit_degree_hi, fit_degree_vhi))
    }

    # Build the true / ideal 2D contour and plot
    if tstamp_exists:
        raise NotImplementedError()
    else:
        ctrue = np.random.randn(design_matrices["true"].shape[-1]) * coeff_signal_to_noise
        ztrue = (np.matmul(design_matrices["true"], ctrue)).reshape((nx, nx), order="C")

    fitting_polynomials_2d.plot_image(
        ztrue, "Ideal model", filename=os.path.join(outdir, "ideal_"+tstmp+".png"))
    output["ctrue"] = ctrue
    output["ztrue"] = ztrue

    # Add the random noise to generate the dataset and plot
    if tstamp_exists:
        raise NotImplementedError()
    else:
        zdata = ztrue + noise_sigma * np.random.randn(*ztrue.shape)
    fitting_polynomials_2d.plot_image(
        zdata, "Data", filename=os.path.join(outdir, "data_"+tstmp+".png"))
    output["zdata"] = zdata

    # Prepare torch tensor and run
    zflat = zdata.flatten(order="C")
    ztensor = torch.from_numpy(zflat).to(device).view(-1, 1)
    for _spec in run_specifications:

        _design_matrix = design_matrices[_spec]
        _lstsq_coeffs = np.linalg.lstsq(_design_matrix, zflat, rcond=None)[0].T
        _lstsq_prediction = _design_matrix.dot(_lstsq_coeffs).reshape((nx, nx), order="C")
        _lstsq_loss = ((zdata - _lstsq_prediction)**2).mean()
        output[f"lstsq_pred_{_spec}"] = _lstsq_prediction
        output[f"lstsq_loss_{_spec}"] = _lstsq_loss

        # Initialize pytorch elements
        _dataset = TensorDataset(torch.from_numpy(_design_matrix).to(device), ztensor)
        _loader = DataLoader(_dataset, batch_size=len(ztensor))
        _model = torch.nn.Linear(_design_matrix.shape[-1], 1, bias=False).double()
        _model.to(device)

        # Gradient descent optimization
        _optimizer = [
            _f(params=_model.parameters(), **_kw) for _f, _kw in gradient_descent_optim.items()][0]

        print(f"Running Gradient Descent with {gradient_descent_optim}")
        _model.train()
        _loss0 = 0.
        _losses = []
        for i in range(1 + n_epochs):
            for batch, (X, y) in enumerate(_loader):
                _optimizer.zero_grad()
                _pred = _model(X)
                _loss = loss_function(_pred, y)
                _losses.append(_loss.item())
                # Backpropagation
                _loss.backward()
                _optimizer.step()

            if (i == 0) or ((1 + i) % iter_stride == 0):  # report
                print(
                    f"GD/{tstmp}/{_spec}: epoch: {1 + i:d}/{n_epochs:d}, loss: {_loss:>7f}, "
                    f"dloss: {min(_loss - _loss0, 0):>7e}, ideal: {_lstsq_loss:>7f}"
                )
                _loss0 = _loss.item()
                if np.isclose(_loss0 / _lstsq_loss, 1., atol=0, rtol=early_exit_rtol):
                    break

        output[f"gd_pred_{_spec}"] = _pred.numpy(force=True).reshape((nx, nx), order="C")
        output[f"gd_losses_{_spec}"] = _losses

        # L-BFGS optimimization
        _optimizer = [_f(params=_model.parameters(), **_kw) for _f, _kw in lbfgs_optim.items()][0]

        print(f"Running L-BFGS with {lbfgs_optim}")
        _loss0 = 0.
        _losses = []
        for j in range(n_epochs // iter_stride):
            for batch, (X, y) in enumerate(_loader):
                def closure(_loss0=_loss0):
                    _optimizer.zero_grad()
                    _pred = _model(X)
                    _loss = loss_function(_pred, y)
                    _losses.append(_loss.item())
                    _loss.backward()
                    return _loss
                _optimizer.step(closure)

                # report
                _pred = _model(X)
                _loss = loss_function(_pred, y)
                print(
                    f"L-BFGS/{tstmp}/{_spec}: epoch: {(1 + j) * iter_stride:d}/{n_epochs:d}, "
                    f"loss: {_loss:>7f}, dloss: {min(_loss - _loss0, 0):>7e}, "
                    f"ideal: {_lstsq_loss:>7f}"
                )
                _loss0 = _loss.item()

            if np.isclose(_loss0 / _lstsq_loss, 1., atol=0, rtol=early_exit_rtol):
                break

        output[f"lbfgs_pred_{_spec}"] = _pred.numpy(force=True).reshape((nx, nx), order="C")
        output[f"lbfgs_losses_{_spec}"] = _losses

    # Calculate and plot residuals
    # rlo = zdata - pred_lo
    # plot_image(
    #     rlo,
    #     "Low degree polynomial residuals",
    #     filename=os.path.join(outdir, "lo_"+tstmp+".png"),
    #     clim=CLIM,
    # )
    # rtrue = zdata - pred_true
    # plot_image(
    #     rtrue,
    #     "Matching degree polynomial residuals",
    #     filename=os.path.join(outdir, "matching_"+tstmp+".png"),
    #     clim=CLIM,
    # )
    # rhi = zdata - pred_hi
    # plot_image(
    #     rhi,
    #     "High degree polynomial residuals",
    #     filename=os.path.join(outdir, "hi_"+tstmp+".png"),
    #     clim=CLIM,
    # )
    # rvhi = zdata - pred_vhi
    # plot_image(
    #     rvhi,
    #     "Very high degree polynomial residuals",
    #     filename=os.path.join(outdir, "vhi_"+tstmp+".png"),
    #     clim=CLIM,
    # )
    # output["rlo"] = rlo
    # output["rtrue"] = rtrue
    # output["rhi"] = rhi
    # output["rvhi"] = rvhi

    # Save output for further analysis
    outfile = os.path.join(outdir, "output_"+tstmp+".pickle")
    print("Saving to "+outfile)
    with open(outfile, "wb") as fout:
        pickle.dump(output, fout)
