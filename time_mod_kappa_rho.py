#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:48:45 2022

@author: Liora Rueff

Solves the 1D quasiperiodic time-modulated problem "exactly" using Muller's method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as lg
np.set_printoptions(precision=3)
mpl.rcParams['figure.dpi'] = 150
import muller2
from termcolor import colored
import sympy as sy
import scipy.integrate as integrate
import matplotlib.animation as animatio
from scipy.integrate import solve_ivp
import time

animate = False


class kappa_rho_tdep:
    def __init__(self, N, li=None, lij=None):
        """
        Parameters
        ----------
        N : INTEGER
            Number of resonators inside one cell.
        li : LIST, optional
            List of the lengths of each resonator (x_i^-,x_i^+). The default is None.
        lij : LIST, optional
            List of the lengths between the i-th and (i+1)-th resonaor. The default is None.

        Returns
        -------
        None
        
        """
        self. N = N

        if lij is None:
            # create a list of random scalars for \ell_{i(i+1)}
            self.lij = np.random.randint(1, 5, size=(N-1,))
        else:
            self.lij = np.asarray(lij)
        if li is None:
            # create a list of random scalars for \ell_{i}
            self.li = np.random.randint(1, 5, size=(N,))
        else:
            self.li = np.asarray(li)
        if N == 1:
            self.li = np.array([1])

        # Create a list of the lengths of resonators and the lengths between to adjacent resonators
        Ls = np.zeros((2*self.N))
        Ls[::2] = self.li
        Ls[1::2] = self.lij
        self.L = np.sum(self.li) + np.sum(self.lij)

        # Array of the coordinates x_i^- and x_i^+
        self.xipm = np.insert(np.cumsum(Ls), 0, 0)

        self.xipmCol = np.column_stack((self.xipm[0:-2:2], self.xipm[1::2]))
        self.xim = self.xipmCol[:, 0]  # Coordinates x_i^+
        self.xip = self.xipmCol[:, 1]  # Coordinates x_i^+

        self.r = min(np.pi/self.li)  # Forbidden frequency

    def setparams(self, v, vb, O, delta, alpha, omega=None):
        """
        Set the parameters of the problem.

        Parameters
        ----------
        v : FLOAT
            Wave speed in the background medium.
        vb : FLOAT
            Wave speed inside the resonators.
        omega : FLOAT, optional
            Frequency of the wave field. The default is None.
        O : FLOAT
            Frequency of the time-modulated \rho(t).
        delta : FLOAT
            Contrast parameter.
        alpha : FLOAT
            Quasi wave number.

        Returns
        -------
        None

        """
        n = 0
        self.alpha = alpha
        self.v = v  # Wavespeed in the background medium
        self.vb = vb  # Wavespeed in the resonators
        self.O = O  # Frequency of the time-modulated \rho(t)
        self.delta = delta  # Contrast parameter \rho_b/\rho
        if omega:
            self.omega = omega  # Frequency of the wave field
            self.k = (self.omega+n*self.O)/self.v # Wave number in the background medium
            self.kb_n = (self.omega+n*self.O)/self.vb # Wave number in the resonators 
            self.uin = lambda x: np.exp(1j*self.k*x-1j*self.k*self.L+1j*self.alpha*self.L)  # Incomming wave field
            self.duin = lambda x: 1j*self.k*np.exp(1j*self.k*x-1j*self.k*self.L+1j*self.alpha*self.L) # Derivative of incomming wave field

    def getPlottingPoints(self, sampling_points=100):
        """
        Returns a sampling of points both inside and outside the resonators.

        Parameters
        ----------
        sampling_points : FLOAT, optional
            Number of samples to be taken in the intervals. The default is 100.

        Returns
        -------
        pointsInt : ARRAY
            Sampling points inside the resonators.
        pointsExt : ARRAY 
            Sampling points outside the resonators.
            
        """
        kr = np.abs(self.k)
        if self.k == 0:
            kr = 0.1*self.r
        factor = 0.002 if self.omega < 0.005 else 0.01
        pointsExt = [np.linspace(-3*np.pi/factor, -10, 500)]
        pointsExt.append(np.linspace(-10, 0, 100))
        for i in range(self.N-1):
            pointsExt.append(np.linspace(self.xip[i], self.xim[i+1], 100))
        pointsExt.append(np.linspace(self.xip[-1], self.xip[-1]+10, 100))
        pointsExt.append(np.linspace(
            self.xip[-1]+10, self.xip[-1]+3*np.pi/factor, 500))

        pointsInt = []
        for (i, xm, xp) in zip(range(self.N), self.xipmCol[:, 0],
                               self.xipmCol[:, 1]):
            pointsInt.append(np.linspace(xm, xp, 30))
        return pointsInt, pointsExt

    def wf(self, f):
        """
        Defines the function w_f which solves the exterior problem (2.1).

        Parameters
        ----------
        f : LIST
            Boundary data of w_f at x_i^{\pm}

        Returns
        -------
        result : FUNCTION
            Gives the function w_f of x.
        ai : ARRAY
            Vector of coefficients a_i and b_i of u(x) in each interval (x_i^+,x_{i+1}^-).
            
        """
        assert len(
            f) == 2 * self.N, f"Need 2*N boundary condition, got {len(f)} instead of {2 * self.N}"
        if self.k == 0:
            def result(x):
                if x <= self.xim[0]:
                    return f[0]
                elif x >= self.xip[-1]:
                    return f[-1]
                for i in range(self.N - 1):
                    if x >= self.xip[i] and x <= self.xim[i+1]:
                        return f[2 * i] + (f[2 * i + 1] - f[2 * i]) / (self.xim[i + 1] - self.xip[i]) * (
                            x - self.xip[i])

            return [], result
        else:
            def exteriorblock(l, k, xp, xpm):
                return -1 / (2j * np.sin(k * l)) * np.asarray([[np.exp(-1j * k * xpm), -np.exp(-1j * k * xp)],
                                                               [-np.exp(1j * k * xpm), np.exp(1j * k * xp)]])

            TBlocks = lg.block_diag(*(exteriorblock(l, self.k, xp, xpm)
                                      for (l, xp, xpm) in zip(self.lij, self.xip[:-1], self.xim[1:])))
            # Compute the coefficients a_i and b_i as given in (2.3)
            ai = TBlocks.dot(f[1:-1])

            def result(x: np.ndarray):
                y = np.zeros(x.shape, dtype=complex)
                if x <= self.xim[0]:
                    return f[0]*np.exp(-1j*self.k*(x-self.xim[0]))
                if x >= self.xip[-1]:
                    return f[-1]*np.exp(1j*self.k*(x-self.xip[-1]))
                for i in range(self.N - 1):
                    mask = (x >= self.xip[i]) * (x <= self.xim[i + 1])
                    y[mask] = ai[2 * i] * np.exp(1j * self.k * x[mask]) \
                        + ai[2 * i + 1] * np.exp(-1j * self.k * x[mask])
                    return y

            return ai, result

    def plotExt(self, f, title):
        """
        Plots the solution u in the background medium.

        Parameters
        ----------
        f : LIST
            Boundary data on x_i^{\pm}.
        title : TYPE
            DESCRIPTION.

        Returns
        -------
        None

        """
        ai, wfres = self.wf(f)
        pointsInt, pointsExt = self.getPlottingPoints()
        wfs = [[wfres(p) for p in pts] for pts in pointsExt]

        plt.figure(figsize=(5, 2))
        for (pts, w) in zip(pointsExt, wfs):
            plt.plot(pts, np.real(w), color='C0')
        plt.title(title)
        plt.figure(figsize=(5, 2))
        for (pts, w) in zip(pointsExt[1:-1], wfs[1:-1]):
            plt.plot(pts, np.real(w), color='C0')
        plt.title(title)

    def get_DirichletNeumann_matrix(self, alpha, n) -> np.ndarray:
        """
        Returns the Quasiperiodic Dirichlet to Neumann map in matrix form as given in eq (2.5) of Florian's paper.

        Parameters
        ----------
        alpha : FLOAT
            Quasi wave number.
        n : INTEGER
            Specifies the Fourier mode.

        Returns
        -------
        T: ARRAY
            The 2Nx2N matrix defining the Dirichlet-to-Neumann map.
            
        """
        def Ak(l, k):
            return np.asarray([[-k * np.cos(k * l) / np.sin(k * l), k / np.sin(k * l)],
                               [k / np.sin(k * l), -k * np.cos(k * l) / np.sin(k * l)]])

        k = (self.omega+n*self.O)/self.vb
        T = lg.block_diag(-k[-1] * np.cos(k[-1] * self.lij[-1]) / np.sin(k[-1] * self.lij[-1]),
                          *[np.asarray(Ak(self.lij[i], k[i]))
                            for i in range(self.N-1)],
                          - k[-1] * np.cos(k[-1] * self.lij[-1]) / np.sin(k[-1] * self.lij[-1]))
        T = np.array(T, dtype=complex)
        T[-1, 0] = k[-1] * np.exp(1j * alpha * self.L) / \
            np.sin(k[-1] * self.lij[-1])
        T[0, -1] = k[-1] * np.exp(-1j * alpha * self.L) / \
            np.sin(k[-1] * self.lij[-1])
        return T

    def getMatcalA(self, rs, ks, alpha: float) -> np.ndarray:
        """
        Builds the matrix A(omega,delta) as defined by (3.3), which is used to determine the coefficients a_i, b_i of the solution.

        Parameters
        ----------
        rs : LIST
            The Fourier coefficients of 1/\rho(t).
        ks : LIST
            The Fourier coefficients of 1/\kappa(t).
        alpha : FLOAT
            Quasi wave number of the wave field.

        Returns
        -------
        A : ARRAY
            The 2(2M+1)Nx2(2M+1)N matrix as defined by (3.3).
            
        """
        M = int((len(rs[0,:])-1)/2)
        n = 0
        
        # print(f"omega={self.omega}")
        
        def gamma(i,n,m):
            if np.abs(self.omega+n*self.O) < 10**(-14):
                d = self.omega+0.00001+n*self.O
            else:
                d = self.omega+n*self.O
            gamma = np.sqrt((self.omega+(n-m)*self.O)/(d)*ks[i,M+m]/rs[i,M+m])
            return gamma 

        def G(n,m,rs):
            km = (self.omega+(n-m)*self.O)/self.vb
            G = 1j*lg.block_diag(*([[-km[i]*gamma(i,n,m)*rs[i,m+M]*np.exp(1j*gamma(i,n,m)*km[i]*self.xim[i]), km[i]*gamma(i,n,m)*rs[i,m+M]*np.exp(-1j*gamma(i,n,m)*km[i]*self.xim[i])],
                                    [km[i]*gamma(i,n,m)*rs[i,m+M]*np.exp(1j*gamma(i,n,m)*km[i]*self.xip[i]), -km[i]*gamma(i,n,m)*rs[i,m+M]*np.exp(-1j*gamma(i,n,m)*km[i]*self.xip[i])]]
                                   for i in range(self.N)))
            return G

        def V(n,m):
            km = (self.omega+(n-m)*self.O)/self.vb
            V = lg.block_diag(*([[np.exp(1j*gamma(i,n,m)*km[i]*self.xim[i]), np.exp(-1j*gamma(i,n,m)*km[i]*self.xim[i])],
                                 [np.exp(1j*gamma(i,n,m)*km[i]*self.xip[i]), np.exp(-1j*gamma(i,n,m)*km[i]*self.xip[i])]]
                                for i in range(self.N)))
            return V

        A = np.zeros((int(2*M+1)*2*self.N, int(2*M+1)*2*self.N), dtype=np.complex)
        # G_m1 = G(n,1,rs)
        # G_0 = G(n,0,rs)
        # G_p1 = G(n,-1,rs)
        # V_m1 = V(n,1)
        # V_0 = V(n,0)
        # V_p1 = V(n,-1)
        T_m1 = self.get_DirichletNeumann_matrix(alpha, -1)
        T_0 = self.get_DirichletNeumann_matrix(alpha, 0)
        T_p1 = self.get_DirichletNeumann_matrix(alpha, 1)
        
        G_nm1_m0 = G(-1,0,rs)
        G_nm1_mm1 = G(-1,-1,rs)
        G_n0_mp1 = G(0,1,rs)
        G_n0_m0 = G(0,0,rs)
        G_n0_mm1 = G(0,-1,rs)
        G_np1_mp1 = G(1,1,rs)
        G_np1_m0 = G(1,0,rs)
        
        V_nm1_m0 = V(-1,0)
        V_n0_m0 = V(0,0)
        V_np1_m0 = V(1,0)
        
        A[0*self.N:2*self.N, 0*self.N:2*self.N] = G_nm1_m0 - self.delta*T_m1.dot(V_nm1_m0)
        A[0*self.N:2*self.N, 2*self.N:4*self.N] = G_nm1_mm1
        A[2*self.N:4*self.N, 0*self.N:2*self.N] = G_n0_mp1
        A[2*self.N:4*self.N, 2*self.N:4*self.N] = G_n0_m0 - self.delta*T_0.dot(V_n0_m0)
        A[2*self.N:4*self.N, 4*self.N:6*self.N] = G_n0_mm1
        A[4*self.N:6*self.N, 2*self.N:4*self.N] = G_np1_mp1
        A[4*self.N:6*self.N, 4*self.N:6*self.N] = G_np1_m0 - self.delta*T_p1.dot(V_np1_m0)

        # A[0*self.N:2*self.N, 4*self.N:6*self.N] = G_m1 - self.delta*T_p1.dot(V_m1)
        # A[2*self.N:4*self.N, 2*self.N:4*self.N] = G_0 - self.delta*T_0.dot(V_0)
        # A[4*self.N:6*self.N, 0*self.N:2*self.N] = G_p1 - self.delta*T_p1.dot(V_p1)
        # A[0*self.N:2*self.N, 2*self.N:4*self.N] = G_0
        # A[2*self.N:4*self.N, 0*self.N:2*self.N] = G_p1
        # A[2*self.N:4*self.N, 4*self.N:6*self.N] = G_m1
        # A[4*self.N:6*self.N, 2*self.N:4*self.N] = G_0

        if (A[2*self.N:4*self.N, 0*self.N:2*self.N] == np.zeros((2*self.N, 2*self.N))).all():
            if (A[2*self.N:4*self.N, 4*self.N:6*self.N] == np.zeros((2*self.N, 2*self.N))).all():
                A = A[2*self.N:4*self.N, 2*self.N:4*self.N]

        return A

    def getu(self, rs, ks, alpha: float, sol=None):
        """
        Computes the solution to the exterior problem for a given quasi wave number.

        Parameters
        ----------
        rs : LIST
            The Fourier coefficients of 1/\rho(t).
        ks : LIST
            The Fourier coefficients of 1/\kappa(t).
        alpha : FLOAT
            Quasi wave number of the wave field.
        sol : ARRAY, optional
            2Nx1 array containing the coefficients a_i, b_i for all 1<=i<=N. The default is None.

        Returns
        -------
        utot: FUNCTION
            Total wave field solving the exterior problem.
            
        """
        M = int((len(rs)-1)/2)

        if sol is None:
            A = self.getMatcalA(rs, ks, alpha)
            # Solve the linear system (3.3) to get the coefficients a_i and b_i
            sol = np.linalg.solve(A, np.zeros(A.shape[0], dtype=np.complex))
            if sol.shape[0] != 2*self.N:
                sol = sol[2*self.N:4*self.N]

        def u(x):
            for (i, xim, xip) in zip(range(self.N), self.xipmCol[:, 0], self.xipmCol[:, 1]):
                if x >= xim and x <= xip:
                    # Construct the function u as given in Lemma 3.1
                    return sol[2*i]*np.exp(1j*self.kb_n[i]*x)+sol[2*i+1]*np.exp(-1j*self.kb_n[i]*x)

        f = [u(xi)-self.uin(xi)
             for xi in self.xipm[:-1]]  # Define boundary data
        # Gives the vector of coefficients a_i, b_i and the function w_f(x) defined by (2.2)
        ai, usext = self.wf(f)

        def utot(x):
            res = usext(x)
            if res is None:
                return u(x)
            else:
                # Total wave field solving the exterior problem
                return res+self.uin(x)

        self.sol = sol
        self.ai = ai
        self.f = f
        self.u = utot

        return utot

    def plot_u(self, alpha, animate=False, ylim=None, zoom=None):
        """
        Plots the solution u.

        Parameters
        ----------
        alpha : FLOAT
            Quasi wave number.
        animate : TYPE, optional
            DESCRIPTION. The default is False.
        ylim : TYPE, optional
            DESCRIPTION. The default is None.
        zoom : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        line : TYPE
            DESCRIPTION.

        """
        fig = plt.figure()
        pointsInt, pointsExt = self.getPlottingPoints()
        if zoom:
            pointsExt = pointsExt[1:-1]
        uint = [[self.u(p) for p in pts] for pts in pointsInt]
        uext = [[self.u(p) for p in pts] for pts in pointsExt]

        if not animate:
            phase = np.conj(uext[0][0])/np.abs(uext[0][0])
            for (pts, ui) in zip(pointsInt, uint):
                plt.plot(pts, np.real(np.asarray(ui)*phase), color='C1')

            for (pts, ue) in zip(pointsExt, uext):
                plt.plot(pts, np.real(np.asarray(ue)*phase), color='C0')
            # plt.xlim(-pointsExt[0][0],pointsExt[-1][-1])
            # if np.abs(np.real(self.omega))> 1e-8:
            #    plt.ylim(-1,1)
            if ylim:
                plt.ylim(-ylim, ylim)
            #plt.title(f'Solution at $\omega$={self.omega}')
        else:
            plt.ioff()
            plt.close('all')
            plt.title(f'Solution at $\omega$={self.omega}')
            line = ()

            dt = 2*np.pi/np.abs(self.omega)*0.01
            for (pts, ui) in zip(pointsInt, uint):
                line = line + (plt.plot(pts, np.real(ui), color='C1')[0],)
            plt.ylim(-3, 3)

            for (pts, ue) in zip(pointsExt, uext):
                line = line + (plt.plot(pts, np.real(ue), color='C0')[0],)

            def init():
                i = 0
                for (pts, ui) in zip(pointsInt, uint):
                    line[i].set_data([], [])
                    i += 1

                for (pts, ue) in zip(pointsExt, uext):
                    line[i].set_data([], [])
                    i += 1
                return line

            def animate(j):
                t = j * dt
                i = 0
                for (pts, ui) in zip(pointsInt, uint):
                    line[i].set_data(pts, np.real(
                        np.asarray(ui)*np.exp(-1j*self.omega*t)))
                    i += 1

                for (pts, ue) in zip(pointsExt, uext):
                    line[i].set_data(pts, np.real(
                        np.asarray(ue)*np.exp(-1j*self.omega*t)))
                    i += 1
                return line
            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=100, blit=True, interval=50, repeat=True)
            plt.show()
        plt.legend()
        plt.title(
            f'Solution of the quasiperiodic problem for $\\alpha=${alpha}')
        plt.xlabel('$x$')
        plt.ylabel('$u^{\alpha}(x)$')
        fig.show()

    def get_capacitance_matrix(self, alpha):
        """
        Computes the capacitance matrix.

        Parameters
        ----------
        alpha : FLOAT
            Quasi wave number.

        Returns
        -------
        C : ARRAY
            Capacitance matrix.

        """
        C = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                if i == j - 1:
                    C[i, j] += - 1 / self.lij[i]
                if i == j:
                    C[i, j] += (1 / self.lij[j - 1] + 1 / self.lij[j])
                if i == j + 1:
                    C[i, j] += - 1 / self.lij[j]
                if (j == 0) and (i == self.N - 1):
                    C[i, j] += - np.exp(1j * alpha * self.L) / self.lij[-1]
                if (i == 0) and (j == self.N - 1):
                    C[i, j] += - np.exp(-1j * alpha * self.L) / self.lij[-1]
        self.C = C
        return C


    def get_generalized_capacitance_matrix(self, alpha: float):
        """
        Computes the generalized capacitance matrix.

        Parameters
        ----------
        alpha : FLOAT
            Quasi wave number.

        Returns
        -------
        Cgen : ARRAY
            Generalized capacitance matrix.

        """
        V = np.diag(self.li)
        S = np.diag(self.vb)
        # TODO: Do we need to multiply by delta or not?
        Cgen = S ** 2 @ np.linalg.inv(V) @ self.get_capacitance_matrix(alpha=alpha)
        self.Cgen = Cgen
        return Cgen

    def resonantfrequencies(self, v, vb, delta, alpha):
        """
        Taken from the static case, might need modifications later!

        Parameters
        ----------
        v : SCALAR
            Wavespeed in the background medium.
        vb : SCALAR
            Wavespeed in the resonators.
        delta : SCALAR
            Contrast parameter \rho_b/\rho.
        alpha : INTEGER
            Quasi wave number.

        Returns
        -------
        freqs : ARRAY
            A list of all frequencies \omega_i, for i=1,...,N.
        vmodes : ARRAY
            A list of all modes.

        """
        # if self.N == 1:
        #     # Use the formula (3.28) for \omega_1(\delta)
        #     freqs = np.array(
        #         [-1j*vb[0]*np.log(1+2*vb[0]*delta/(v-vb[0]*delta))])
        # else:
        #     # Use the formula (3.23) for \omega_1(\delta)
        #     freqs = np.array([-1j*delta*2*vb[0]**2/(v*np.sum(self.li))])
        # # By rmk 3.3 the mode associated to the zero frequency is 1
        # vmodes = np.array([1]*self.N)
        # if self.N > 1:
        #     # d1 = np.concatenate(([1/self.lij[0]],
        #     #                      1/self.lij[:-1]+1/self.lij[1:],[1/self.lij[-1]]))
        #     # d2 = -1/self.lij[:]
        #     # C = np.diag(d1)+np.diag(d2,1)+np.diag(d2,-1) # Define matrix C as given by (1.13)
        #     C = self.get_capacitance_matrix(alpha)
        #     V = np.diag(self.li)  # Define matrix V as given by (1.14)
        #     Vhalf = np.sqrt(np.linalg.pinv(V))  # V^(-1/2)
        #     # Find the eigenvalues & -vectors of C
        #     lambdas, vmodes = np.linalg.eigh(Vhalf.dot(C).dot(Vhalf))
        #     vmodes = Vhalf.dot(vmodes)

        #     B = np.diag([1]+[0]*(self.N-2)+[1])
        #     aiBai = np.diag(vmodes[:, 1:].T.dot(B.dot(vmodes[:, 1:])))
        #     self.aiBai = aiBai
        #     self.lambdas = lambdas
        #     freqs = np.append(freqs, np.sqrt(delta)*vb[1:]*np.sqrt(lambdas[1:])
        #                       - 1j*delta*vb[1:]**2/(2*v)*aiBai)  # Compute the remaining N-1 frequencies as given by (3.24)
        # else:
        #     vmodes = vmodes[:, None]

        # return freqs
        
        # def rhot(t):
        #     return 1 / (1 + epsilon_rho * np.cos(self.O * t + phi_rho))


        # def sqrtkappa(t):
        #     return 1 / np.sqrt(1 + epsilon_kappa * np.cos(self.O * t + phi_kappa))


        # def w3(t):
        #     return self.O**2/4*(1+((epsilon_kappa**2-1)/(1+epsilon_kappa*np.cos(self.O*t+phi_kappa))**2))


        # def build_MM(C, t):
        #     Rho = np.diag(rhot(t))
        #     Rinv = np.diag(1 / rhot(t))
        #     K = np.diag(sqrtkappa(t))
        #     W3 = np.diag(w3(t))
        #     M = delta * vb[0] / self.li[0] * K @ Rho @ C @K @ Rinv + W3
        #     MM = np.block([[np.zeros((self.N, self.N), dtype=complex), np.identity(self.N, dtype=complex)], [-M, np.zeros((self.N, self.N), dtype=complex)]])
        #     return MM
        # # Compute the band functions

        # C = self.get_generalized_capacitance_matrix(alpha)
        # I_N = np.identity(2 * self.N, dtype=complex)
        # W = np.zeros((2*self.N,2*self.N), dtype=complex)
        # def F(t,y):
        #     return build_MM(C, t)@y
        # for l in range(2 * self.N):
        #     sol = solve_ivp(F, [0, T], I_N[:, l],t_eval=[T])
        #     y1 = sol.y
        #     W[:,l] = y1.T
        # w,v = np.linalg.eig(W)
        # # w_real = np.sort(np.real(np.log(w)/1j/T))
        # # freq = w_real.T
        # w_reim = np.sort((np.log(w)/1j/T))
        # freq = w_reim.T
        
        # return freq
    
        C_alpha = self.get_capacitance_matrix(alpha)
        V = np.diag(self.li)
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(V).dot(C_alpha))
        pos_freqs = np.sqrt(delta)*np.multiply(vb, np.sqrt(eigvals))
        neg_freqs = -np.sqrt(delta)*np.multiply(vb, np.sqrt(eigvals))
        freqs = np.append(pos_freqs,neg_freqs)
        return freqs
    
    def get_approximations(self, v, vb, delta, alpha):
        """
        Taken from the static case, might need modifications later!

        Parameters
        ----------
        v : SCALAR
            Wavespeed in the background medium.
        vb : SCALAR
            Wavespeed in the resonators.
        delta : SCALAR
            Contrast parameter \rho_b/\rho.
        alpha : INTEGER
            Quasi wave number.

        Returns
        -------
        freqs : ARRAY
            A list of all frequencies \omega_i, for i=1,...,N.
        vmodes : ARRAY
            A list of all modes.

        """
        def rhot(t):
            return 1 / (1 + epsilon_rho * np.cos(self.O * t + phi_rho))


        def sqrtkappa(t):
            return 1 / np.sqrt(1 + epsilon_kappa * np.cos(self.O * t + phi_kappa))


        def w3(t):
            return self.O**2/4*(1+((epsilon_kappa**2-1)/(1+epsilon_kappa*np.cos(self.O*t+phi_kappa))**2))


        def build_MM(C, t):
            Rho = np.diag(rhot(t))
            Rinv = np.diag(1 / rhot(t))
            K = np.diag(sqrtkappa(t))
            W3 = np.diag(w3(t))
            M = delta * vb[0] / self.li[0] * K @ Rho @ C @K @ Rinv + W3
            MM = np.block([[np.zeros((self.N, self.N), dtype=complex), np.identity(self.N, dtype=complex)], [-M, np.zeros((self.N, self.N), dtype=complex)]])
            return MM
        # Compute the band functions

        C = self.get_generalized_capacitance_matrix(alpha)
        I_N = np.identity(2 * self.N, dtype=complex)
        W = np.zeros((2*self.N,2*self.N), dtype=complex)
        def F(t,y):
            return build_MM(C, t)@y
        for l in range(2 * self.N):
            sol = solve_ivp(F, [0, T], I_N[:, l],t_eval=[T])
            y1 = sol.y
            W[:,l] = y1.T
        w,v = np.linalg.eig(W)
        # w_real = np.sort(np.real(np.log(w)/1j/T))
        # freq = w_real.T
        w_reim = np.sort((np.log(w)/1j/T))
        freq = w_reim.T
        
        return freq


def muller(N, alpha, delta, vb, v, O, rs, ks, li=None, lij=None):
    """
    Apply Muller's method to find \omega such that the smallest eigenvalue of A(\omega,\delta)
    is zero. These values of \omega are exactly the desired resonant frequencies.

    Parameters
    ----------
    N : INTEGER
        Number of resonators in one cell.
    alpha : FLOAT
        Quasi wave number.
    delta : FLOAT
        Contrast parameter between the resonators and the background.
    vb : FLOAT
        Wave speed inside the resonators.
    v : FLOAT
        Wave speed in the background medium.
    omega : FLOAT
        Frequency of the wave field.
    O : FLOAT
        Frequency of the time modulated \rho inside the resonators.
    rs : LIST
        1x(2*M+1) list with the Fourier coefficients of 1/\rho(t).
    ks : LIST
        1x(2*M+1) list with the Fourier coefficients of 1/\kappa(t).
    li : LIST, optional
        List of the lengths of each resonator (x_i^-,x_i^+). The default is None.
    lij : LIST, optional
        List of the lengths between the i-th and (i+1)-th resonaor. The default is None.

    Returns
    -------
    roots: ARRAY
        An array containing all roots found by Muller's method based on three initial values.
    run_time : ARRAY
        An array containing the time it takes for Muller's method to compute the result.
        
    """
    wavepb = kappa_rho_tdep(N, li, lij)
    wavepb.setparams(v, vb, O, delta, alpha, omega=None)

    def f(omega):
        """
        Computes the determinant of the matrix B, which represents the bilinear form, for given \omega.

        Parameters
        ----------
        omega : SCALAR
            Resonant frequency.

        Returns
        ------
        SCALAR
            Determinant of the matrix \mathcal{A} associated to the given \omega.
            
        """
        wavepb.setparams(v, vb, O, delta, alpha, omega)

        A_matrix = wavepb.getMatcalA(rs, ks, alpha)
        # Compute eigenvalues and right eigenvectors
        lamb, eigv = np.linalg.eig(A_matrix)
        order = np.argsort(np.abs(lamb))  # Indices of ordered eigenvalues
        wavepb.sol = eigv[:, order[0]]

        return np.linalg.det(A_matrix)
    

    freqs = wavepb.resonantfrequencies(v, vb, delta, alpha)
    roots = []
    run_time= []
    for i, omegai in enumerate(freqs):
        # Define an initial guess for Muller's method
        xk = [omegai*(1+0.01*np.exp(1j*k*2*np.pi/3)) for k in (0, 1, 2)]
        f(xk[0])
        # Root of f, i.e. \omega such that the smallest eigenvalue of A(\omega,\delta) is zero
        start_time = time.time()
        root = muller2.muller(xk, f)
        run_time.append(time.time()-start_time)
        print(f"Initial guess: {omegai}")
        print(f"Root found: {root}   Value: {np.abs(f(root))}")
        roots.append(root)
        if wavepb.sol.shape[0] == 2*wavepb.N:
            sol = wavepb.sol
        else:
            sol = wavepb.sol[2*wavepb.N:4*wavepb.N]
        # Computes the total wave field which solves the exterior problem
        wavepb.getu(rs, ks, alpha, sol)
        # Plots the total wave field u_i(x) corresponding to \omega_i
        wavepb.plot_u(alpha=alpha, animate=False)
        plt.xlabel('$x$')
        plt.ylabel(f'$u_{{i}}(x)$')
        # savefig(output+'/mode_'+str(i)+'_'+str(N)+'.pdf')
        wavepb.plot_u(alpha, zoom=True)  # Plot the total wave field near x=0
        plt.xlabel('$x$')
        plt.ylabel(f'$u_{{i}}(x)$')
        # savefig(output+'/mode_'+str(i)+'_'+str(N)+'_zoom.pdf')
        plt.title(f'Eigenmode $\omega_{i}=${root}')
        wavepb.setparams(v, vb, O, delta, alpha, np.real(root)*(1-delta**2))

        u = wavepb.getu(rs, ks, alpha)
        wavepb.plot_u(alpha, animate=animate, ylim=3)
        plt.xlabel('$\omega$')
        plt.ylabel('$u(x)$')
        # savefig(output+f'/u_omegar_{i}_{N}.pdf')
        wavepb.plot_u(alpha, animate=False, ylim=3, zoom=True)
        plt.xlabel('$\omega$')
        plt.ylabel('$u(x)$')
        # savefig(output+f'/u_omegar_{i}_{N}_zoom.pdf')
    print("\n")
    # Adding plot of the constant mode
    sol = np.array([1]*2*N)
    wavepb.omega = 0
    wavepb.k = 0
    wavepb.kb = np.array([0]*N)
    wavepb.getu(rs, ks, alpha)
    wavepb.plot_u(alpha)
    plt.xlabel('$x$')
    plt.ylabel(f'$u_{0}(x)$')
    # savefig(output+'/mode_'+str(N)+'_'+str(N)+'.pdf')
    wavepb.plot_u(alpha, zoom=True)
    plt.xlabel('$x$')
    plt.ylabel(f'$u_{0}(x)$')
    # savefig(output+'/mode_'+str(N)+'_'+str(N)+'_zoom.pdf')
    plt.title(f'Eigenmode $\omega_{{i}}=0$')
    for i, omegai in enumerate(freqs):
        re = format(np.real(omegai), '.5g')
        im = format(np.imag(omegai), '.5g')
        rec = format(np.real(roots[i]), '.5g')
        imc = format(np.imag(roots[i]), '.5g')
        plus = "+" if np.imag(omegai) > 0 else ""
        print(colored(
            f"Root {i} predicted - computed: ${re} {plus} {im}\\ii$ & ${rec} {plus} {imc}\\ii$", color="green"))
        print(colored(
            f"Root {i} predicted - computed: ${omegai}  & ${roots[i]}$", color="blue"))
    return np.sort(np.asarray(roots)), np.asarray(run_time)



N = 2
delta = 0.0001
vb = np.array([1, 1])
v = 1
O = 0.03
alpha = 0.02

epsilon_kappa = 0.00001
epsilon_rho = 0.00001
phi_rho = np.array([0, np.pi / 2])
phi_kappa = np.array([0, np.pi / 2])

rs = np.zeros((N,3),dtype=complex)
ks = np.zeros((N,3),dtype=complex)
rs[0,:] += epsilon_rho*np.exp(-1j*phi_rho[0]),1,epsilon_rho*np.exp(1j*phi_rho[0])
rs[1,:] += epsilon_rho*np.exp(-1j*phi_rho[1]),1,epsilon_rho*np.exp(1j*phi_rho[1])
# rs[2,:] += epsilon_rho*np.exp(-1j*phi_rho[2]),1,epsilon_rho*np.exp(1j*phi_rho[2])
# rs[3,:] += epsilon_rho*np.exp(-1j*phi_rho[3]),1,epsilon_rho*np.exp(1j*phi_rho[3])
# rs[4,:] += epsilon_rho*np.exp(-1j*phi_rho[4]),1,epsilon_rho*np.exp(1j*phi_rho[4])
ks[0,:] += epsilon_kappa*np.exp(-1j*phi_kappa[0]),1,epsilon_kappa*np.exp(1j*phi_kappa[0])
ks[1,:] += epsilon_kappa*np.exp(-1j*phi_kappa[1]),1,epsilon_kappa*np.exp(1j*phi_kappa[1])
# ks[2,:] += epsilon_kappa*np.exp(-1j*phi_kappa[2]),1,epsilon_kappa*np.exp(1j*phi_kappa[2])
# ks[3,:] += epsilon_kappa*np.exp(-1j*phi_kappa[3]),1,epsilon_kappa*np.exp(1j*phi_kappa[3])
# ks[4,:] += epsilon_kappa*np.exp(-1j*phi_kappa[4]),1,epsilon_kappa*np.exp(1j*phi_kappa[4])

li = [1, 1]
lij = [1,  1]
L = np.sum(li) + np.sum(lij)
T = 2*np.pi/O
# alpha = 0.02

assert len(rs) == len(ks), "The number of Fourier coefficients of 1/\rho(t) should be the same as the number of Fourier coefficients of 1/\kappa(t)"
assert len(li) == N, f"There are {N} resonators, thus, li must have {N} elements and not {len(li)} elements."
assert len(lij) == N, f"There are {N} resonators, thus, li must have {N} elements and not {len(lij)} elements."


# muller(N, alpha, delta, vb, v, O, rs, ks, li, lij)

sample_points = 100
alphas = np.linspace(-np.pi / L, np.pi / L, sample_points)
oms = np.zeros((sample_points,2*N), dtype=complex)
guess = np.zeros((sample_points,2*N), dtype=complex)
times = np.zeros((sample_points,2*N))
i = 0
for alpha in alphas:
    wavepb = kappa_rho_tdep(N, li, lij)
    wavepb.setparams(v, vb, O, delta, alpha, omega=None)
    roots, run_time = muller(N, alpha, delta, vb, v, O, rs, ks, li, lij)
    freqs = wavepb.get_approximations(v, vb, delta, alpha)
    oms[i,:] += roots
    guess[i,:] += freqs
    times[i,:] += run_time
    i += 1
plt.close('all')



## Compute relative error
# err = np.mean(np.linalg.norm(oms-guess)/np.linalg.norm(oms))
# err = np.max(np.sqrt((np.sum(np.real(oms-freqs))**2+np.sum(np.imag(oms-freqs))**2))/np.sqrt((np.sum(np.real(oms))**2+np.sum(np.imag(oms))**2)))
err = np.max(np.amax(np.sqrt(np.real(oms-guess)**2+np.imag(oms-guess)**2)/np.sqrt(np.real(oms)**2+np.imag(oms)**2), axis=1))

## Create Plot of error of real part
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
for i in range(0,2*N-1):
    ax.plot(alphas, np.abs(np.real(oms[:,i])-np.real(guess[:,i])), 'b-', linewidth=2)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('Error Real Part', fontsize=18)

## Create Plot of error of imaginary part
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
for i in range(0,2*N-1):
    ax.plot(alphas, np.abs(np.imag(oms[:,i])-np.imag(guess[:,i])), 'b-', linewidth=2)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('Error Imaginary Part', fontsize=18)

## Create Plot of relative error of imaginary part
# ax.set_ylabel('Error Imaginary Part', fontsize=18)
# fig, ax = plt.subplots(1, figsize=(10, 7))
# font = {'family' : 'normal',
#         'weight': 'normal',
#         'size'   : 14}
# plt.rc('font', **font)
# for i in range(0,2*N-1):
#     im_rel_error = np.divide((np.imag(guess[:,i])-np.imag(oms[:,i])),np.imag(oms[:,i]))
#     ax.plot(alphas,im_rel_error,'b-',linewidth=2)
# ax.set_xlabel('$\\alpha$', fontsize=18)
# ax.set_ylabel('Relative Error Imaginary Part', fontsize=18)

## Create plot of Re(\omega) calculated with Muller's method compared to those calculated by the capacitance matrix
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
ax.plot(alphas, np.real(oms[:,0]), 'b-', linewidth=2, label='Muller`s Method')
ax.plot(alphas, np.real(oms[:,1:2*N]), 'b-', linewidth=2)
ax.plot(alphas, np.real(guess[:,0]), 'r--', linewidth=2, label='Capacitance \nApproximation')
ax.plot(alphas, np.real(guess[:,1:2*N]), 'r--', linewidth=2)
ax.legend(fontsize=18,loc=1)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('Re$(\\omega_i^{\\alpha})$', fontsize=18)

## Create plot of Im(\omega) calculated with Muller's method compared to those calculated by the capacitance matrix
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
ax.plot(alphas, np.imag(oms[:,0]), 'b-', linewidth=2, label='Muller`s Method')
ax.plot(alphas, np.imag(oms[:,1:2*N]), 'b-', linewidth=2)
ax.plot(alphas, np.imag(guess[:,0]), 'r--', linewidth=2, label='Capacitance \nApproximation')
ax.plot(alphas, np.imag(guess[:,1:2*N]), 'r--', linewidth=2)
ax.legend(fontsize=18,loc=1)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('Im$(\\omega_i^{\\alpha})$', fontsize=18)

## Create plot of Re(\omega) and Im(\omega) calculated with Muller's method 
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
ax.plot(alphas, np.real(oms[:,0]), 'b-', linewidth=2, label='Re$(\\omega_i^{\\alpha})$')
ax.plot(alphas, np.real(oms[:,1:2*N]), 'b-', linewidth=2)
ax.plot(alphas, np.imag(oms[:,0]), 'r-', linewidth=2, label='Im$(\\omega_i^{\\alpha})$')
ax.plot(alphas, np.imag(oms[:,1:2*N]), 'r-', linewidth=2)
ax.legend(fontsize=18)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('Re$(\\omega_i^{\\alpha})$ resp. Im$(\\omega_i^{\\alpha})$', fontsize=18)

## Create plot of \omega calculated with Muller's method and the capacitance matrix
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 16}
plt.rc('font', **font)
ax.plot(alphas, np.real(oms[:,0]), '-', color='blue', linewidth=2, label='Muller`s Method \n(Real Part)')
ax.plot(alphas, np.real(oms[:,1:2*N]), '-', color='blue', linewidth=2)
ax.plot(alphas, np.real(guess[:,0]), '--', color='cyan', linewidth=2, label='Capacitance \nApproximation \n(Real Part)')
ax.plot(alphas, np.real(guess[:,1:2*N]), '--', color='cyan', linewidth=2)
ax.plot(alphas, np.imag(oms[:,0]), '-', color='red', linewidth=2, label='Muller`s Method \n(Imaginary Part)')
ax.plot(alphas, np.imag(oms[:,1:2*N]), '-', color='red', linewidth=2)
ax.plot(alphas, np.imag(guess[:,0]), '--', color='orange', linewidth=2, label='Capacitance \nApproximation \n(Imaginary Part)')
ax.plot(alphas, np.imag(guess[:,1:2*N]), '--', color='orange', linewidth=2)
ax.legend(fontsize=20,loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
ax.set_xlabel('$\\alpha$', fontsize=20)
ax.set_ylabel('$\\omega_i^{\\alpha}$', fontsize=20)

## Create plot of run time
fig, ax = plt.subplots(1, figsize=(10, 7))
font = {'family' : 'normal',
        'weight': 'normal',
        'size'   : 14}
plt.rc('font', **font)
ax.plot(alphas,times,'-', linewidth=2) 
ax.legend(fontsize=18)
ax.set_xlabel('$\\alpha$', fontsize=18)
ax.set_ylabel('Run Time [s]', fontsize=18)

    
    
    
    
    
    
    
