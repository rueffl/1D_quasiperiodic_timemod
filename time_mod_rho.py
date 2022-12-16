#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:28:53 2022

@author: Liora Rueff
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
import matplotlib.animation as animation

animate = False


class rho_tdep:
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
        None.
        """
        self. N = N
        
        if lij is None:
            self.lij = np.random.randint(1, 5, size=(N-1,)) # create a list of random scalars for \ell_{i(i+1)}
        else:
            self.lij = np.asarray(lij)
        if li is None:
            self.li = np.random.randint(1, 5, size=(N,)) # create a list of random scalars for \ell_{i}
        else:
            self.li = np.asarray(li)
        if N==1:
            self.li = np.array([1])
            
        Ls = np.zeros((N+N-1,)) # Create a list of the lengths of resonators and the lengths between to adjacent resonators
        Ls[::2] = self.li
        Ls[1::2] = self.lij
        self.L = np.sum(self.li) + np.sum(self.lij)
    
        self.xipm = np.insert(np.cumsum(Ls), 0, 0) # Array of the coordinates x_i^- and x_i^+
    
        self.xipmCol = np.column_stack((self.xipm[0::2], self.xipm[1::2]))
        self.xim = self.xipmCol[:, 0] # Coordinates x_i^+
        self.xip = self.xipmCol[:, 1] # Coordinates x_i^+
        
        self.long_lij = np.zeros(N)
        self.long_lij[0:N-1] = self.lij
        self.long_lij[-1] = self.L-self.xip[-1]+self.xim[0]
    
        self.r = min(np.pi/self.li) # Forbidden frequency
        
    
    def setparams(self, v, vb, omega, O, delta, alpha):
        """
        Set the parameters of the problem.

        Parameters
        ----------
        v : FLOAT
            Wave speed in the background medium.
        vb : FLOAT
            Wave speed inside the resonators.
        omega : FLOAT
            Frequency of the wave field.
        O : FLOAT
            Frequency of the time-modulated \rho(t).
        delta : FLOAT
            Contrast parameter.
        alpha : FLOAT
            Quasi wave number.

        Returns
        -------
        None.

        """
        n = 0
        self.alpha = alpha
        self.v = v # Wavespeed in the background medium
        self.vb = vb # Wavespeed in the resonators
        self.omega = omega # Frequency of the wave field
        self.O = O # Frequency of the time-modulated \rho(t)
        self.delta = delta # Contrast parameter \rho_b/\rho
        self.k = (self.omega+n*self.O)/self.v # Wave number in the background medium
        self.kb_n = (self.omega+n*self.O)/self.vb # Wave number in the resonators
        self.uin = lambda x: np.exp(1j*self.k*x) # Incomming wave field
        self.duin = lambda x: 1j*self.k*np.exp(1j*self.k*x) # Derivative of incomming wave field
    
    def getPlottingPoints(self, sampling_points=100):
        """
        Returns a sampling of points both inside and outside the resonators.
        
        Parameters
        ----------
        sampling_points : FLOAT, default=100
            Number of samples to be taken in the intervals.

        Returns
        -------
        pointsInt : ARRAY
            Sampling points inside the resonators.
        pointsExt : ARRAY 
            Sampling points outside the resonators.
        """
        kr = np.abs(self.k)
        if self.k==0:
            kr = 0.1*self.r
        factor = 0.002 if self.omega<0.005 else 0.01
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
        assert len(f) == 2 * self.N, f"Need 2*N boundary condition, got {len(f)} instead of {2 * self.N}"
        if self.k == 0:
            def result(x):
                if x<= self.xim[0]:
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
            ai = TBlocks.dot(f[1:-1]) # Compute the coefficients a_i and b_i as given in (2.3)

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
        None.

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
            return np.asarray([[-1*k* np.cos(k * l) / np.sin(k * l), k/ np.sin(k * l)],
                               [k / np.sin(k * l), -1*k* np.cos(k * l) / np.sin(k * l)]])
    
        k = (self.omega+n*self.O)/self.vb
        T = lg.block_diag(-k[-1]* np.cos(k[-1] * self.lij[-1]) / np.sin(k[-1]* self.lij[-1]),
                          *[np.asarray(Ak(self.lij[i], k[i])) for i in range(self.N-1)],
                          - k[-1] * np.cos(k[-1] * self.lij[-1]) / np.sin(k[-1] * self.lij[-1]))
        T = np.array(T, dtype=complex)
        T[-1, 0] = k[-1] * np.exp(1j * alpha * self.L) / np.sin(k[-1] * self.lij[-1])
        T[0, -1] = k[-1] * np.exp(-1j * alpha * self.L) / np.sin(k[-1] * self.lij[-1])
        return T
    
    def getMatcalA(self, As, alpha: float) -> np.ndarray:
        """
        Builds the matrix A(omega,delta) as defined by (3.3), which is used to determine the coefficients a_i, b_i of the solution.

        Parameters
        ----------
        As : LIST
            The Fourier coefficients of 1/\rho(t).
        alpha : FLOAT
            Quasi wave number of the wave field.

        Returns
        -------
        A : ARRAY
            The 2(2M+1)Nx2(2M+1)N matrix as defined by (3.3).
        """
        M = len(As)
        n = 0
        def G(m):
            km = (self.omega+m*self.O)/self.vb
            G = 1j*lg.block_diag(*([[-km[i]*np.exp(1j*km[i]*self.xim[i]), km[i]*np.exp(-1j*km[i]*self.xim[i])],
                                      [km[i]*np.exp(1j*km[i]*self.xip[i]), -km[i]*np.exp(-1j*km[i]*self.xip[i])]]
                                      for i in range(self.N)))
            return G
        def V(m):
            km = (self.omega+m*self.O)/self.vb
            V = lg.block_diag(*([[np.exp(1j*km[i]*self.xim[i]), np.exp(-1j*km[i]*self.xim[i])],
                                      [np.exp(1j*km[i]*self.xip[i]), np.exp(-1j*km[i]*self.xipm[i])]]
                                      for i in range(self.N)))
            return V
        
        A = np.zeros((M*2*self.N,M*2*self.N),dtype=np.complex)
        G_m1 = As[0]*G(n-1)
        G_0 = As[1]*G(n)
        G_p1 = As[2]*G(n+1)
        V_m1 = V(n-1)
        V_0 = V(n)
        V_p1 = V(n+1)
        T_m1 = self.get_DirichletNeumann_matrix(alpha, n-1)
        T_0 = self.get_DirichletNeumann_matrix(alpha, n)
        T_p1 = self.get_DirichletNeumann_matrix(alpha, n+1)
        
        A[0:2*self.N,0:2*self.N] = G_m1-self.delta*T_m1.dot(V_m1)
        A[2*self.N:4*self.N,2*self.N:4*self.N] = G_0-self.delta*T_0.dot(V_0)
        A[4*self.N:6*self.N,4*self.N:6*self.N] = G_p1-self.delta*T_p1.dot(V_p1)
        A[0:2*self.N,2*self.N:4*self.N] = G_0
        A[2*self.N:4*self.N,0:2*self.N] = G_m1
        A[2*self.N:4*self.N,4*self.N:6*self.N] = G_p1
        A[4*self.N:6*self.N,2*self.N:4*self.N] = G_0
        
        return A
        
    
    def getu(self, As, alpha: float, sol=None):
        """
        Computes the solution to the exterior problem for a given quasi wave number.
        
        Parameters
        ----------
        As : LIST
            The Fourier coefficients of 1/\rho(t).
        alpha : FLOAT
            Quasi wave number of the wave field.
        sol : ARRAY
            2Nx1 array containing the coefficients a_i, b_i for all 1<=i<=N.

        Returns
        -------
        utot: FUNCTION
            Total wave field solving the exterior problem.
        """
        M = int((len(As)-1)/2)
        
        if sol is None:
            A = self.getMatcalA(As, alpha)
            sol = np.linalg.solve(A, np.zeros(self.N * 2*(2*M+1),dtype=np.complex)) # Solve the linear system (3.3) to get the coefficients a_i and b_i
            sol = sol[2*self.N:4*self.N]

        def u(x):
            for (i, xim, xip) in zip(range(self.N), self.xipmCol[:, 0], self.xipmCol[:, 1]):
                if x >= xim and x <= xip:
                    return sol[2*i]*np.exp(1j*self.kb_n[i]*x)+sol[2*i+1]*np.exp(-1j*self.kb_n[i]*x) # Construct the function u as given in Lemma 3.1

        f = [u(xi)-self.uin(xi) for xi in self.xipm] # Define boundary data
        ai, usext = self.wf(f) # Gives the vector of coefficients a_i, b_i and the function w_f(x) defined by (2.2)

        def utot(x):
            res = usext(x)
            if res is None:
                return u(x)
            else:
                return res+self.uin(x) # Total wave field solving the exterior problem

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
            pointsExt=pointsExt[1:-1]
        uint = [[self.u(p) for p in pts] for pts in pointsInt]
        uext = [[self.u(p) for p in pts] for pts in pointsExt]

        if not animate:
            phase = np.conj(uext[0][0])/np.abs(uext[0][0])
            for (pts, ui) in zip(pointsInt, uint):
                plt.plot(pts, np.real(np.asarray(ui)*phase), color='C1')

            for (pts, ue) in zip(pointsExt, uext):
                plt.plot(pts, np.real(np.asarray(ue)*phase), color='C0')
            #plt.xlim(-pointsExt[0][0],pointsExt[-1][-1])
            #if np.abs(np.real(self.omega))> 1e-8:
            #    plt.ylim(-1,1)
            if ylim:
                plt.ylim(-ylim,ylim)
            #plt.title(f'Solution at $\omega$={self.omega}')
        else:
            plt.ioff()
            plt.close('all')
            plt.title(f'Solution at $\omega$={self.omega}')
            line = ()

            dt=2*np.pi/np.abs(self.omega)*0.01
            for (pts, ui) in zip(pointsInt, uint):
                line = line +(plt.plot(pts, np.real(ui), color='C1')[0],)
            plt.ylim(-3,3)

            for (pts, ue) in zip(pointsExt, uext):
                line = line+ (plt.plot(pts, np.real(ue), color='C0')[0],)

            def init():
                i=0
                for (pts, ui) in zip(pointsInt, uint):
                    line[i].set_data([],[])
                    i+=1

                for (pts, ue) in zip(pointsExt, uext):
                    line[i].set_data([],[])
                    i+=1
                return line

            def animate(j): 
                t = j * dt
                i=0
                for (pts, ui) in zip(pointsInt, uint):
                    line[i].set_data(pts, np.real(np.asarray(ui)*np.exp(-1j*self.omega*t)))
                    i+=1

                for (pts, ue) in zip(pointsExt, uext):
                    line[i].set_data(pts, np.real(np.asarray(ue)*np.exp(-1j*self.omega*t)))
                    i+=1
                return line
            ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, blit=True, interval=50, repeat=True)
            plt.show()
        plt.legend()
        plt.title(f'Solution of the quasiperiodic problem for $\\alpha=${alpha}')
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
        C = np.zeros((self.N,self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                if i == j-1:
                    C[i,j] += 1/self.long_lij[i]
                if i == j:
                    C[i,j] += -(1/self.long_lij[j-1] + 1/self.long_lij[j])
                if i == j+1:
                    C[i,j] += 1/self.long_lij[j]
                if (j == 0) and (i == self.N-1):
                    C[i,j] += np.exp(1j*alpha*self.L)/self.long_lij[-1]
                if (i == 0) and (j == self.N-1):
                    C[i,j] += np.exp(-1j*alpha*self.L)/self.long_lij[-1]
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
        Cgen = self.delta * S ** 2 @ np.linalg.inv(V) @ self.get_capacitance_matrix(alpha=alpha)
        self.Cgen = Cgen
        return Cgen
        
    def resonantfrequencies(self, v, vb, delta):
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

        Returns
        -------
        freqs : ARRAY
            A list of all frequencies \omega_i, for i=1,...,N.
        vmodes : ARRAY
            A list of all modes.

        """
        # C = self.get_capacitance_matrix(self.alpha)
        if self.N == 1:
            freqs = np.array([-1j*vb[0]*np.log(1+2*vb[0]*delta/(v-vb[0]*delta))]) # Use the formula (3.28) for \omega_1(\delta)
        else:
            freqs = np.array([-1j*delta*2*vb[0]**2/(v*np.sum(self.li))]) # Use the formula (3.23) for \omega_1(\delta)
        vmodes=np.array([1]*self.N) # By rmk 3.3 the mode associated to the zero frequency is 1
        if self.N>1:
            d1 = np.concatenate(([1/self.lij[0]],
                                 1/self.lij[:-1]+1/self.lij[1:],[1/self.lij[-1]]))
            d2 = -1/self.lij
            C = np.diag(d1)+np.diag(d2,1)+np.diag(d2,-1) # Define matrix C as given by (1.13)
            V = np.diag(self.li) # Define matrix V as given by (1.14)
            Vhalf=np.sqrt(np.linalg.pinv(V)) # V^(-1/2)
            lambdas, vmodes = np.linalg.eigh(Vhalf.dot(C).dot(Vhalf)) # Find the eigenvalues & -vectors of C
            vmodes=Vhalf.dot(vmodes)

            B=np.diag([1]+[0]*(self.N-2)+[1])
            aiBai = np.diag(vmodes[:,1:].T.dot(B.dot(vmodes[:,1:])))
            self.aiBai = aiBai
            self.lambdas= lambdas
            freqs = np.append(freqs,np.sqrt(delta)*vb[1:]*np.sqrt(lambdas[1:])
                              -1j*delta*vb[1:]**2/(2*v)*aiBai) # Compute the remaining N-1 frequencies as given by (3.24)
        else:
            vmodes=vmodes[:,None]

        return freqs, vmodes

    
def muller(N, alpha, delta, vb, v, omega, O, A, li=None, lij=None):
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
    A : LIST
        1x(2*M+1) list with the Fourier coefficients of 1/\rho(t).
    li : LIST, optional
        List of the lengths of each resonator (x_i^-,x_i^+). The default is None.
    lij : LIST, optional
        List of the lengths between the i-th and (i+1)-th resonaor. The default is None.
    
    Returns
    -------
    roots: ARRAY
        An array containing all roots found by Muller's method based on three initial values.
    """
    wavepb = rho_tdep(N, li, lij)
    
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
            Smallest eigenvalue associated to the given \omega.
        """
        wavepb.setparams(v, vb, omega, O, delta, alpha)
        
        A_matrix = wavepb.getMatcalA(A, alpha)
        lamb, eigv = np.linalg.eig(A_matrix) # Compute eigenvalues and right eigenvectors
        order = np.argsort(np.abs(lamb)) # Indices of ordered eigenvalues
        wavepb.sol = eigv[:,order[0]]
        
        return np.linalg.det(A_matrix)
        # return lamb[order[0]]
        

    freqs, vmodes = wavepb.resonantfrequencies(v, vb, delta)
    # freqs = np.linspace(0.1,1,num=1*wavepb.N)
    roots = []
    for i, omegai in enumerate(freqs):
        xk = [omegai*(1+0.01*np.exp(1j*k*2*np.pi/3)) for k in (0,1,2)] # Define an initial guess for Muller's method
        f(xk[0])
        root = muller2.muller(xk, f) # Root of f, i.e. \omega such that the smallest eigenvalue of A(\omega,\delta) is zero
        print(f"Initial guess: {omegai}")
        print(f"Root found: {root}   Value: {np.abs(f(root))}")
        roots.append(root)
        wavepb.getu(A, alpha, sol=wavepb.sol[2*wavepb.N:4*wavepb.N]) # Computes the total wave field which solves the exterior problem
        wavepb.plot_u(alpha=alpha, animate=False) # Plots the total wave field u_i(x) corresponding to \omega_i
        plt.xlabel('$x$')
        plt.ylabel(f'$u_{{i}}(x)$')
        #savefig(output+'/mode_'+str(i)+'_'+str(N)+'.pdf')
        wavepb.plot_u(alpha, zoom=True) # Plot the total wave field near x=0
        plt.xlabel('$x$')
        plt.ylabel(f'$u_{{i}}(x)$')
        #savefig(output+'/mode_'+str(i)+'_'+str(N)+'_zoom.pdf')
        plt.title(f'Eigenmode $\omega_{i}=${root}')
        wavepb.setparams(v, vb, np.real(root)*(1-delta**2), O, delta, alpha)
        
        u = wavepb.getu(A, alpha)
        wavepb.plot_u(alpha, animate=animate, ylim=3)
        plt.xlabel('$\omega$')
        plt.ylabel('$u(x)$')
        #savefig(output+f'/u_omegar_{i}_{N}.pdf')
        wavepb.plot_u(alpha, animate=False,ylim=3,zoom=True)
        plt.xlabel('$\omega$')
        plt.ylabel('$u(x)$')
        #savefig(output+f'/u_omegar_{i}_{N}_zoom.pdf')
    print("\n")
    #Adding plot of the constant mode
    sol = np.array([1]*2*N)
    wavepb.omega = 0
    wavepb.k=0
    wavepb.kb=0
    wavepb.getu(A, alpha)
    wavepb.plot_u(alpha)
    plt.xlabel('$x$')
    plt.ylabel(f'$u_{0}(x)$')
    #savefig(output+'/mode_'+str(N)+'_'+str(N)+'.pdf')
    wavepb.plot_u(alpha,zoom=True)
    plt.xlabel('$x$')
    plt.ylabel(f'$u_{0}(x)$')
    #savefig(output+'/mode_'+str(N)+'_'+str(N)+'_zoom.pdf')
    plt.title(f'Eigenmode $\omega_{{i}}=0$')
    for i, omegai in enumerate(freqs):
        re=format(np.real(omegai),'.5g')
        im=format(np.imag(omegai),'.5g')
        rec=format(np.real(roots[i]),'.5g')
        imc=format(np.imag(roots[i]),'.5g')
        plus = "+" if np.imag(omegai)>0  else ""
        print(colored(f"Root {i} predicted - computed: ${re} {plus} {im}\\ii$ & ${rec} {plus} {imc}\\ii$",color="green"))
        print(colored(f"Root {i} predicted - computed: ${omegai}  & ${roots[i]}$",color="blue"))
    return np.asarray(roots)


N = 3
delta = 0.1
vb = np.array([1,2,1])
v = 0.8 
omega = np.sqrt(delta)
M = 5
# O = 2*np.sqrt(delta) 
O = 0.2
# O = 1
epsilon = 0
phi = 0.1
A = [epsilon*np.exp(-1j*phi),1,epsilon*np.exp(1j*phi)]
# A = [1,1,1]
li=[1]*N
lij=[0.5]*(N-1)
L = np.sum(li) + np.sum(lij)
alpha = 0.001

assert len(li) == N, f"There are {N} resonators, thus, li must have {N} elements and not {len(li)} elements."
assert len(lij) == N-1, f"There are {N} resonators, thus, li must have {N-1} elements66 and not {len(lij)} elements."

muller(N, alpha, delta, vb, v, omega, O, A, li, lij)

# Plot \omega_1 as a function od \delta
sample_points = 50
alphas = np.linspace(-np.pi / L, np.pi / L, sample_points)
oms =np.zeros((sample_points,N), dtype=complex)
i = 0
for alpha in alphas:
    roots = muller(N, alpha, delta, vb, v, omega, O, A, li, lij)
    oms[i,:] += roots
    i += 1
plt.close('all')
fig, ax = plt.subplots(1, figsize=(10, 7))
ax.plot(alphas,np.real(oms),'b-',label='Re$(\\omega_i)$')
ax.plot(alphas,np.imag(oms),'r-',label='Im$(\\omega_i)$')
ax.legend()
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$\\omega_i^{\\alpha}$')
for i in range(N):
    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.plot(alphas,np.real(oms[:,i]),'b-',label='Re$(\\omega_i)$')
    ax.plot(alphas,np.imag(oms[:,i]),'r-',label='Im$(\\omega_i)$')
    ax.legend()
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\omega_i^{\\alpha}$')
