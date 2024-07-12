"""
plot_excess_residual_correlation_constraints.py
===============================================

Plotting constraints from positive semi-definiteness for the excess correlation
of residuals at different observations, described in the paper "No patterns in
regression residuals."
"""

import numpy as np
import matplotlib.pyplot as plt


# Functions

def xi_mid(rho_ggp, omega=1., psi=.8):
    return -omega * psi * rho_ggp / (1. - omega * psi)


def xi_half_range(rho_ggp, omega=1., psi=.8):
    return (
        2. * np.sqrt(psi * (1. - ((1. + omega)**2 * psi / 4.)) * (1. -  rho_ggp**2))
        / (1. - omega * psi)
    )

def xi_min(rho_ggp, omega=1., psi=.8):
    return xi_mid(rho_ggp, omega=omega, psi=psi) - xi_half_range(rho_ggp, omega=omega, psi=psi)


def xi_max(rho_ggp, omega=1., psi=.8):
    return xi_mid(rho_ggp, omega=omega, psi=psi) + xi_half_range(rho_ggp, omega=omega, psi=psi)


if __name__ == "__main__":

    x = np.linspace(-1, 1, 201)
    plt.plot(x, xi_mid(x, omega=1, psi=0.9), "k--")
    plt.plot(x, xi_min(x, omega=1, psi=0.9), "k", label=r"$\omega = 1, \psi = 0.9$")
    plt.plot(x, xi_max(x, omega=1, psi=0.9), "k")

    plt.plot(x, xi_mid(x, omega=1., psi=0.7), color="grey", ls="--")
    plt.plot(x, xi_max(x, omega=1., psi=0.7), color="grey", label=r"$\omega = 1, \psi = 0.7$")
    plt.plot(x, xi_min(x, omega=1., psi=0.7), color="grey")
    plt.axhline(-1, color="k", ls=":")
    plt.axhline(+1, color="k", ls=":")
    plt.grid()
    plt.legend(fontsize="large")
    plt.xlim(-1, 1)
    plt.ylim(-1.75, 1.75)
    plt.xlabel(r"$\rho_{gg'}$", fontsize="x-large")
    plt.ylabel(r"$\xi_{rr'}$", fontsize="x-large")
    plt.tight_layout()
    plt.show()
