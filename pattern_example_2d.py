"""
pattern_example_2d.py
=====================
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer


# Parameters
# ==========

# Side of unit square
nx = 32

# Sigma of iid pixel noise
noise_sigma = 1.

# True, low (insufficient) order and high (overfitting) order
fit_degree0 = 8
fit_degree_lo = 2
fit_degree_hi = 32

# Per coefficient "signal to noise" in random true pattern
coeff_signal_to_noise = 8.


# Main script
# ===========

tstmp =  pd.Timestamp.now().isoformat().replace(":", "")

# Define unit interval and x, y grid coords for centroids of a spanning square grid
xvals = np.linspace(-.5 + 1./(2. * nx), .5 - 1./(2. * nx), num=nx, endpoint=True)
Xxgrid, Xygrid = np.meshgrid(xvals, xvals)

# Define design matrix and transform to polynomial linear feature (?)
X = np.stack([Xxgrid.flatten(), Xygrid.flatten()], axis=1)
if True:
	poly_true = PolynomialFeatures(degree=fit_degree0)
	poly_fit_lo = PolynomialFeatures(degree=fit_degree_lo)
	poly_fit_hi = PolynomialFeatures(degree=fit_degree_hi)
	ftype = "poly"
else:
	poly_true = SplineTransformer(degree=3, n_knots=2)#, extrapolation="periodic")
	poly_fit_lo = SplineTransformer(degree=3, n_knots=fit_degree0)#, extrapolation="periodic")
	poly_fit_hi = SplineTransformer(degree=3, n_knots=fit_degree_hi)#, extrapolation="periodic")
	ftype = "b-spline"

features_true = poly_true.fit_transform(X)
features_fit_lo = poly_fit_lo.fit_transform(X)
features_fit_hi = poly_fit_hi.fit_transform(X)

# Build the true curve
ctrue = np.random.randn(features_true.shape[-1]) * coeff_signal_to_noise
ztrue = (np.matmul(features_true, ctrue)).reshape((nx, nx))
plt.title("Ideal model curve")
plt.pcolor(ztrue, cmap="Greys"); plt.colorbar()
plt.savefig(os.path.join(".", "plots", "ideal_"+tstmp+".png"))
plt.show()

# Add the random noise to generate the dataset
zdata = ztrue + noise_sigma * np.random.randn(*ztrue.shape)
plt.title("Data")
plt.pcolor(zdata, cmap="Greys"); plt.colorbar()
plt.savefig(os.path.join(".", "plots", "data_"+tstmp+".png"))
plt.show()

# Perform low order, true and high order regressions
regr_lo = linear_model.LinearRegression()
regr_lo.fit(features_fit_lo, zdata.flatten())
pred_lo = regr_lo.predict(features_fit_lo).reshape((nx, nx))

regr0 = linear_model.LinearRegression()
regr0.fit(features_true, zdata.flatten())
pred0 = regr0.predict(features_true).reshape((nx, nx))

regr_hi = linear_model.LinearRegression()
regr_hi.fit(features_fit_hi, zdata.flatten())
pred_hi = regr_hi.predict(features_fit_hi).reshape((nx, nx))

# Plot residuals
plt.pcolor(zdata - pred_lo, cmap="Greys"); plt.colorbar(); plt.clim([-2.5, 2.5])
plt.title("Low order polynomial fit residuals")
plt.savefig(os.path.join(".", "plots", "lo_"+tstmp+".png"))
plt.show()
rlo = zdata - pred_lo

plt.pcolor(zdata - pred0, cmap="Greys"); plt.colorbar(); plt.clim([-2.5, 2.5])
plt.title("Matching order polynomial fit residuals")
plt.savefig(os.path.join(".", "plots", "matching_"+tstmp+".png"))
plt.show()
r0 = zdata - pred0

plt.pcolor(zdata - pred_hi, cmap="Greys"); plt.colorbar(); plt.clim([-2.5, 2.5])
plt.title("High order polynomial fit residuals")
plt.savefig(os.path.join(".", "plots", "hi_"+tstmp+".png"))
plt.show()
rhi = zdata - pred_hi
