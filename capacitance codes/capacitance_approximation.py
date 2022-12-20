import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from numpy import linalg as la
import utils

# Define the structure
li = [1, 1]
lij = [2, 1]
N = 2
delta = 0.001
vb = [1, 1]
# noinspection PyInterpreter
pwp = utils.PeriodicWaveProblem(N=N,
                                li=li,
                                lij=lij,
                                v=1,
                                vb=vb,
                                omega=None,
                                delta=delta)
L = np.sum(li) + np.sum(lij)
# Define the time modulation
epsr = 0
epsk = 0
Omega = 0.2
T = 2 * np.pi / Omega
phase_rho = np.array([0, np.pi / 2])
phase_kappa = np.array([0, np.pi / 2])


# Build the matrix M(t)
def rhot(t):
    return 1 / (1 + epsr * np.cos(Omega * t + phase_rho))


def sqrtkappa(t):
    return 1 / np.sqrt(1 + epsk * np.cos(Omega * t + phase_kappa))


def w3(t):
    return Omega ** 2 / 4 * (1 + ((epsk ** 2 - 1) / (1 + epsk * np.cos(Omega * t + phase_kappa)) ** 2))


def build_MM(C, t):
    Rho = np.diag(rhot(t))
    Rinv = np.diag(1 / rhot(t))
    K = np.diag(sqrtkappa(t))
    W3 = np.diag(w3(t))
    M = delta * vb[0] / li[0] * K @ Rho @ C @K @ Rinv + W3
    MM = np.block([[np.zeros((N, N), dtype=complex), np.identity(N, dtype=complex)], [-M, np.zeros((N, N), dtype=complex)]])
    return MM
# Compute the band functions
N_alpha = 100
alphas = np.linspace(-np.pi / L, np.pi / L, N_alpha)
resonant_frequencies = []

freq = np.zeros((2*N,N_alpha), dtype=complex)
for i in range(N_alpha):
    C = pwp.get_generalized_capacitance_matrix(alphas[i])
    I_N = np.identity(2 * N, dtype=complex)
    W = np.zeros((2*N,2*N), dtype=complex)
    def F(t,y):
        return build_MM(C, t)@y
    for l in range(2 * N):
        sol = solve_ivp(F, [0, T], I_N[:, l],t_eval=[T])
        y1 = sol.y
        W[:,l] = y1.T
    w,v = la.eig(W)
    w_real = np.sort(np.real(np.log(w)/1j/T))
    freq[:,i]=w_real.T
print(freq)
for k in range(2*N):
    plt.plot(alphas, np.real(freq[k, :]))
plt.show()
