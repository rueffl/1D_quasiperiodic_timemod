# Transmission properties of time-dependent one-dimensional metamaterials

We solve the wave equation with periodically time-modulated material parameters in a one-dimensional high-contrast resonator structure in the subwavelength regime exactly, for which we compute the subwavelength quasifrequencies numerically using Muller's method. We prove a formula in the form of an ODE using a capacitance matrix approximation. Comparison of the exact results with the approximations reveals that the method of capacitance matrix approximation is accurate and significantly more efficient. We prove various transmission properties in the aforementioned structure and illustrate them with numerical simulations. In particular, we investigate the effect of time-modulated material parameters on the formation of degenerate points, band gaps and k-gaps.

This repository provides the codes used in "Transmission properties of time-dependent one-dimensional metamaterials"

capacitance_approximation_plots.py: Produces the band functions using the capacitance matrix approximation.
muller2.py: Muller's method used to solve the problem exactly.
time_mod_kappa_rho.py: Solves the problem exactly for \rho and \kappa time-modulated.
1D_quasiperiodic.py: Solves the static time-dependent problem.
1D_qp_timemod__time_N_plot.py : Iterates through N and plots the run time as a function of N.
