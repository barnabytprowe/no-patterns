import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def jacobian_dcdlambda(n):
    """Jacobian matrix of partial derivatives of the off-diagonal elements of
    an N x N real symmetric circulant correlation matrix with respect to its
    determining eigenvalues.

    If an N x N symmetric circulant correlation matrix R has off-diagonal
    elements c_1, c_2, ..., for example:

    R = (  1  c_1  c_2  c_3  ...  c_2  c_1)
        (c_1    1  c_1  c_2  ...  ...  c_2)
        (c_2  c_1    1  c_1  c_2  ...  ...)
        (c_3  c_2  c_1    1  c_1  c_2  ...)
        (...  ...  c_2  c_1    1  c_1  ...)
        (c_2  ...  ...  c_2  c_1  ...  ...)
        (c_1  c_2  ...  ...  ...  ...    1).

    The matrix is determined by the N//2 off-diagonal elements c_1, c_2, ...,
    c_(N//2), where N//2 denotes floored integer division of N by 2, so that,
    e.g., 4//2 = 2, 5//2 = 2, 6//2 = 3 etc.

    The matrix R can be equivalently defined in terms of a subset of its
    eigenvalues, defined as:

    lam_k = sum_{n=0}^{N-1} c_n exp(i * 2pi * nk / N),

    where c_0 = 1, always.  As R is real symmetric circulant, lam_k = lam_(N-k),
    and thus a subset of N//2 of the eigenvalues can suffice to determine R.
    """
    jmat = jacobian_dcdnu(n)
    if n % 2 == 0:
        jmat[:, :-1] *= 2. / n
        jmat[:, -1] /= n
    else:
        jmat *= 2. / n
    return jmat

def jacobian_dcdnu(n):
    nparams = n // 2
    _u = np.linspace(1, nparams, num=nparams, dtype=float)
    return np.cos(2. * np.pi * np.outer(_u, _u) / n)
