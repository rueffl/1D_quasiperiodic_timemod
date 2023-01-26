#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 20 Dec 15∶18∶40 2022 

@authors: Jinghao Cao, Liora Rueff

The code approximates the subwavelength resonances through the capacitance matrix approximation.
It produces a plot of the subwavelength resonances as a function of \alpha and marks band gaps 
and k-gaps, if they exist.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from numpy import linalg as la
import utils

# Define the structure
li = [1, 1, 1]
lij = [1, 1, 2]
N = 3
delta = 0.0001
vb = [1, 1, 1]
v = 1
# noinspection PyInterpreter
pwp = utils.PeriodicWaveProblem(N=N,
                                li=li,
                                lij=lij,
                                v=v,
                                vb=vb,
                                omega=None,
                                delta=delta)
L = np.sum(li) + np.sum(lij)
# Define the time modulation
epsr = 0
epsk = 0
Omega = 0.03
T = 2 * np.pi / Omega
phase_rho = np.array([0, np.pi / 2, np.pi])
phase_kappa = np.array([0, np.pi / 2, np.pi])


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

freq_real = np.zeros((2*N,N_alpha))
freq_imag = np.zeros((2*N,N_alpha))
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
    # w_real = np.sort(np.real(np.log(w)/1j/T))
    # w_real_idx = np.argsort(np.real(np.log(w)/1j/T))
    # freq_real[:,i]=w_real.T
    w_imag = np.sort(np.imag(np.log(w)/1j/T))
    # w_imag = w_imag[w_real_idx]
    freq_imag[:,i]=w_imag.T
    w_reim = np.sort(np.log(w)/1j/T)
    freq_real[:,i] = np.real(w_reim).T
    # freq_imag[:,i] = np.imag(w_reim).T


fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
alphas_imag = alphas
# for i in range(28,33):
#     # freq_imag = np.delete(freq_imag,28,axis=1)
#     freq_real = np.delete(freq_real,28,axis=1)
#     alphas = np.delete(alphas,28)
#     # freq_imag = np.delete(freq_imag,-29,axis=1)
#     freq_real = np.delete(freq_real,-29,axis=1)
#     alphas = np.delete(alphas,-29)
#     N_alpha -= 2
for k in range(N,2*N):
    if k == N:
        ax.plot(alphas, freq_real[k, :], 'b-', linewidth=2, label='Re$(\\omega_{i}^{\\alpha})$')
        ax.plot(alphas_imag, freq_imag[k, :],'r.', linewidth=2, label='Im$(\\omega_{i}^{\\alpha})$')
        ax.plot(alphas_imag, freq_imag[k-N, :],'r.', linewidth=2)
    else:
        ax.plot(alphas, freq_real[k, :], 'b-', linewidth=2)
        ax.plot(alphas_imag, freq_imag[k, :],'r.', linewidth=2)
        ax.plot(alphas_imag, freq_imag[k-N, :],'r.', linewidth=2)
    if k != 2*N-1:
        if epsr != 0:
            if (k-N)%2 == 0:
                m_left = np.max(freq_real[k,0:int(N_alpha/2)])
                m_right= np.max(freq_real[k,int(N_alpha/2):N_alpha])
            else:
                m_left = np.min(freq_real[k,0:int(N_alpha/2)])
                m_right = np.min(freq_real[k,int(N_alpha/2):N_alpha])
            ax.plot(alphas[0:int(N_alpha/2)],np.ones((int(N_alpha/2)))*m_left,'g--')
            ax.plot(alphas[int(N_alpha/2):N_alpha],np.ones((int(N_alpha/2)))*m_right,'g--')
        if epsk != 0:
            test_left = np.abs(freq_real[k,0:int(N_alpha/2)]-freq_real[k+1,0:int(N_alpha/2)])
            test_right = np.abs(freq_real[k,int(N_alpha/2):N_alpha]-freq_real[k+1,int(N_alpha/2):N_alpha])
            min_left = 0
            min_right = 0
            max_left = 0
            max_right = 0
            for i in range(0,int(N_alpha/2)):
                if min_left == 0:
                    if test_left[i] < 10**(-5):
                        min_left += alphas[i]
                if min_right == 0:
                    if test_right[i] < 10**(-5):
                        min_right += alphas[int(N_alpha/2)+i]
            for i in range(int(N_alpha/2)-1,-1,-1):
                if max_left == 0:
                    if test_left[i] < 10**(-5):
                        max_left += alphas[i]
                if max_right == 0:
                    if test_right[i] < 10**(-5):
                        max_right += alphas[int(N_alpha/2)+i]
            if min_left != 0 and min_right != 0:
                ax.plot(np.ones((int(N_alpha/2)))*min_left,np.linspace(0,0.015,int(N_alpha/2)),'g--')
                ax.plot(np.ones((int(N_alpha/2)))*max_left,np.linspace(0,0.015,int(N_alpha/2)),'g--')
                ax.plot(np.ones((int(N_alpha/2)))*min_right,np.linspace(0,0.015,int(N_alpha/2)),'g--')
                ax.plot(np.ones((int(N_alpha/2)))*max_right,np.linspace(0,0.015,int(N_alpha/2)),'g--')

# ax.plot(np.ones((int(N_alpha/2)))*alphas[N_alpha-7],np.linspace(0,0.015,int(N_alpha/2)),'g--')   
ax.legend(fontsize=18,loc=4)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('$\\omega_i^{\\alpha}$', fontsize=18)


