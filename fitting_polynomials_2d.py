"""
fitting_polynomials_2d.py
=========================
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.linear_model
import sklearn.preprocessing


# Parameters
# ==========

# Side of unit square
nx = 10

# Sigma of iid pixel noise
noise_sigma = 1.

# True, low (insufficient) and high (overfitting) polynomial order to use when fitting
fit_degree_true = 8  # the real signal in the simulations will be a 2D polnoymial of this order
fit_degree_lo = 2
fit_degree_hi = 16

# Per coefficient "signal to noise" in random true pattern, i.e. ratio of standard deviation
# of true curve coefficient values to noise_sigma
coeff_signal_to_noise = 8.


# Functions
# =========

def polynomial_design_matrix(X, degree):
	"""Returns polynomial design matrix for input coordinates and polynomial degree.

	Args:
		X:
			array-like of shape (n_samples, n_dimensions), the data to input to
			the polynomial features
		degree: maximum polynomial degree of features in all dimensions
	"""
	poly_features = sklearn.preprocessing.PolynomialFeatures(degree=degree)
	return poly_features.fit_transform(X, y=None)


# Main script
# ===========

# Current timestamp, used in I/0
tstmp =  pd.Timestamp.now().isoformat().replace(":", "")
outdir = os.path.join(".", "plots", tstmp)
os.mkdir(outdir)

# Output dict - will be pickled
output = {}

# Define x, y grid coords for centroids of a unit square spanning square grid
# centred on the origin
xvals = np.linspace(-.5 + 1./(2. * nx), .5 - 1./(2. * nx), num=nx, endpoint=True)
Xxgrid, Xygrid = np.meshgrid(xvals, xvals)

# Use sklearn PolynomialFeatures to model simple polynomials in 2D
# x, y coordinates
X = np.stack([Xxgrid.flatten(order="C"), Xygrid.flatten(order="C")], axis=1)
# Design matrices for the true, too low and too high cases
features_true = polynomial_design_matrix(X=X, degree=fit_degree_true)
features_fit_lo = polynomial_design_matrix(X=X, degree=fit_degree_lo)
features_fit_hi = polynomial_design_matrix(X=X, degree=fit_degree_hi)

# Build the true 2D contour and plot
ctrue = np.random.randn(features_true.shape[-1]) * coeff_signal_to_noise
ztrue = (np.matmul(features_true, ctrue)).reshape((nx, nx), order="C")
output["ctrue"] = ctrue
output["ztrue"] = ztrue

plt.title("Ideal model curve")
plt.pcolor(ztrue, cmap="Greys"); plt.colorbar()
plt.savefig(os.path.join(outdir, "ideal_"+tstmp+".png"))
plt.show()

# Add the random noise to generate the dataset, and plot
zdata = ztrue + noise_sigma * np.random.randn(*ztrue.shape)
output["zdata"] = zdata

plt.title("Data")
plt.pcolor(zdata, cmap="Greys"); plt.colorbar()
plt.savefig(os.path.join(outdir, "data_"+tstmp+".png"))
plt.show()

# Perform too low order, true and too high order regressions
predictions = []
zflat = zdata.flatten(order="C")
for features in (features_fit_lo, features_true, features_fit_hi):

	regr = sklearn.linear_model.LinearRegression()
	regr.fit(features, zflat)
	predictions.append(regr.predict(features).reshape((nx, nx), order="C"))

pred_lo, pred_true, pred_hi = tuple(predictions)
output["pred_lo"] = pred_lo
output["pred_true"] = pred_true
output["pred_hi"] = pred_hi

# Plot residuals
plt.pcolor(zdata - pred_lo, cmap="Greys"); plt.colorbar(); plt.clim([-2.5, 2.5])
plt.title("Low order polynomial fit residuals")
plt.savefig(os.path.join(outdir, "lo_"+tstmp+".png"))
plt.show()
rlo = zdata - pred_lo
output["rlo"] = rlo

plt.pcolor(zdata - pred_true, cmap="Greys"); plt.colorbar(); plt.clim([-2.5, 2.5])
plt.title("Matching order polynomial fit residuals")
plt.savefig(os.path.join(outdir, "matching_"+tstmp+".png"))
plt.show()
rtrue = zdata - pred_true
output["rtrue"] = rtrue

plt.pcolor(zdata - pred_hi, cmap="Greys"); plt.colorbar(); plt.clim([-2.5, 2.5])
plt.title("High order polynomial fit residuals")
plt.savefig(os.path.join(outdir, "hi_"+tstmp+".png"))
plt.show()
rhi = zdata - pred_hi
output["rhi"] = rhi

outfile = os.path.join(outdir, "output_"+tstmp+".pickle")
print("Saving to "+outfile)
with open(outfile, "wb") as fout:
    pickle.dump(output, fout)
