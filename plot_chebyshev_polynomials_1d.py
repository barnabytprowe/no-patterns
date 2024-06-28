"""
plot_chebyshev_polynomials_1d.py
================================

Creates a plot illustrating the first few Chebyshev polynomials of the first
kind for the paper "No patterns in regression residuals."
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev

from fitting_polynomials_fourier_1d import chebyshev_design_matrix


# x axis
NX = 2001
x = np.linspace(-1., 1., num=NX, endpoint=True)

# Number of degrees, legend labels
MAX_DEGREE = 7
DEGREE_LINESTYLE = {
    0: ":",
    1: "-",
    2: "--",
    3: "-.",
    4: (0, (1, 3)),
    5: (0, (6, 1)),
    6: (0, (1, 1, 14, 1)),
    7: (0, (6, 1, 1, 1, 1, 1)),
}
DEGREE_LABELS = {
    0: r"$T_0(x)$",
    1: r"$T_1(x)$",
    2: r"$T_2(x)$",
    3: r"$T_3(x)$",
    4: r"$T_4(x)$",
    5: r"$T_5(x)$",
    6: r"$T_6(x)$",
    7: r"$T_7(x)$",
}

# Output file & types
OUTFILE = os.path.join(".", "plots", "chebyshev_polynomials_1st_kind")
OUTFILE_EXTENSIONS = (".png", ".pdf")


# Make plot
cheby_polys = chebyshev_design_matrix(x, MAX_DEGREE)
fig = plt.figure(figsize=(10, 4))
for _degree in range(1 + MAX_DEGREE):

    plt.plot(
        x,
        cheby_polys[:, _degree],
        label=DEGREE_LABELS[_degree],
        ls=DEGREE_LINESTYLE[_degree],
        color="k",
        lw=1.5,
    )

plt.grid()
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$T_k(x)$", fontsize=12)
plt.title("Chebyshev polynomials of the first kind")
plt.tight_layout()

# Save
for _ext in OUTFILE_EXTENSIONS:

    _outfile = OUTFILE+_ext
    print(f"Saving to {_outfile}")
    plt.savefig(_outfile)

plt.show()
plt.close(fig)
