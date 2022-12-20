#  Copyright (c) 2022.
#  Copyright held by Silvio Barandun ETH Zurich
#  All rights reserved.
#  Please see the LICENSE file that should have been included as part of this package.


import numpy as np
from scipy import linalg as la
from scipy import sparse
import matplotlib.pyplot as plt
import warnings
import pdb
import matplotlib.cm as cm
from tqdm import tqdm
# from global_constants import constants


class PeriodicWaveProblem:
    """
    Class for 1D Periodic high contrast media
    """

    def __init__(self, N: int, li=None, lij=None, v=1, vb=None, omega=None, delta=1):
        """
        Creates a geometrical set up with physical parameters
        :param N: number of resonators
        :param li: list or None, if list: lengths of the resonators, if None a random list will be generated
        :param lij: list or None, if list: distances between the resonators, if None a random list will be generated
        :param v: float phase velocity outside of the resonators
        :param vb: array, phase velocity inside the resonators, one entry for each resonator
        :param omega: angular frequency
        :param delta: contrast parameter
        """
        self.N = N
        if type(li) is int:
            li = [li] * N
        elif li is None:
            self.li = np.random.randint(1, 5, size=(N,))
        self.li = np.array(li)

        if type(lij) is int:
            lij = [lij] * N
        elif lij is None:
            self.lij = self.lij = np.random.randint(1, 5, size=(N, 1))
        self.lij = np.asarray(lij)
        if not (N == self.li.shape[0]):
            self.N = self.li.shape[0]
            warnings.warn(f"\nNeed to have N=len(li), got {N} and {len(li)}. Using the latter", )

        self.L = np.sum(self.li) + np.sum(self.lij)
        # l is the array with the distances between the interesting points
        l = np.zeros(2 * self.N)
        l[::2] = self.li
        l[1::2] = self.lij

        self.xi = np.insert(np.cumsum(l), 0, 0)
        self.xiCol = np.column_stack((self.xi[0:-2:2], self.xi[1::2]))
        self.xim = self.xiCol[:, 0]
        self.xip = self.xiCol[:, 1]

        if type(vb) in [list, np.ndarray]:
            assert len(
                vb) == self.N, f"Need to have vb to be a scalar or list with len(vb)=self.N. Got list with len {len(vb)} instead of {self.N}"
        if type(vb) in [float, int, complex]:
            vb = [vb] * self.N
        vb = np.array(vb)
        self.v = v
        self.vb = vb
        self.omega = omega
        self.delta = delta
        self.k = self.omega / self.v if not (omega is None) else None
        self.kb = self.omega / self.vb if not (omega is None) else None

        self.resonant_frequencies = {}
        self.resonant_modes_approx = {}
        self.resonant_modes_approx_hermitian = {}

    def get_params(self):
        params = {
            'N': self.N,
            'li': self.li,
            'lij': self.lij,
            'v': self.v,
            'vb': self.vb,
            'omega': self.omega,
            'delta': self.delta
        }
        return params

    def get_DirichletNeumann_matrix(self, alpha) -> np.ndarray:
        """
        Returns the Quasiperiodic Dirichlet to Neumann map in matrix form as given in eq 2.5 of Florian's paper
        :param alpha: quasi periodicity
        :return: np.array
        """

        def Ak(l, k):
            return np.asarray([[-k * np.cos(k * l) / np.sin(k * l), k / np.sin(k * l)],
                               [k / np.sin(k * l), -k * np.cos(k * l) / np.sin(k * l)]])

        T = la.block_diag(- self.k * np.cos(self.k * self.lij[-1]) / np.sin(self.k * self.lij[-1]),
                          *[np.asarray(Ak(l, self.k)) for l in self.lij[:-1]],
                          - self.k * np.cos(self.k * self.lij[-1]) / np.sin(self.k * self.lij[-1]))
        T = np.array(T, dtype=complex)
        T[-1, 0] = self.k * np.exp(1j * alpha * self.L) / np.sin(self.k * self.lij[-1])
        T[0, -1] = self.k * np.exp(-1j * alpha * self.L) / np.sin(self.k * self.lij[-1])
        return T

    def getPlottingPoints(self, sampling_points=100):
        """
        Returns a sampling of points both inside and outside the resonators
        :param sampling_points: number of sample to take in intervals
        :return: np.array, np.array, points_inside, points outside
        """
        factor = 0.002 if self.omega < 0.005 else 0.01
        pointsExt = []  # [np.linspace(-3 * np.pi / factor, -10, 500)]
        pointsExt.append(np.linspace(-10, 0, 100))
        pointsExt = pointsExt + [np.linspace(self.xip[i], self.xim[i + 1], sampling_points) for i in range(self.N - 1)]
        pointsExt.append(np.linspace(self.xip[-1], self.xip[-1] + 10, 100))
        # pointsExt.append(np.linspace(self.xip[-1] + 10, self.xip[-1] + 3 * np.pi / factor, 500))
        pointsExt = np.concatenate(pointsExt)

        pointsInt = [np.linspace(self.xim[i], self.xip[i], sampling_points) for i in range(self.N)]
        pointsInt = np.concatenate(pointsInt)

        return pointsInt, pointsExt

    def wf(self, f):
        """
        Solution to external problem
        :param f: values of the boundary conditions
        :return: (ai, result) where ai are the coefficients of the solution and result is a function giving the solution
        in the fundamental domain
        """
        assert len(f) == 2 * self.N, f"Need 2*N boundary condition, got {len(f)} instead of {2 * self.N}"
        if self.k == 0:
            def result(x):
                # if x<= self.xim[0]:
                #     return f[0]
                # elif x >= self.xip[-1]:
                #     return f[-1]
                for i in range(self.N - 1):
                    if x >= self.xip[i] and x <= self.xim[i + 1]:
                        return f[2 * i] + (f[2 * i + 1] - f[2 * i]) / (self.xim[i + 1] - self.xip[i]) * (
                                x - self.xip[i])

            return [], result
        else:
            def exteriorblock(l, k, xp, xpm):
                return -1 / (2j * np.sin(k * l)) * np.asarray([[np.exp(-1j * k * xpm), -np.exp(-1j * k * xp)],
                                                               [-np.exp(1j * k * xpm), np.exp(1j * k * xp)]])

            TBlocks = la.block_diag(*(exteriorblock(l, self.k, xp, xpm)
                                      for (l, xp, xpm) in zip(self.lij, self.xip[:-1], self.xim[1:])))
            ai = TBlocks.dot(f[1:-1])

            def result(x: np.ndarray):
                y = np.zeros(x.shape, dtype=complex)
                for i in range(self.N - 1):
                    mask = (x >= self.xip[i]) * (x <= self.xim[i + 1])
                    y[mask] = ai[2 * i] * np.exp(1j * self.k * x[mask]) \
                              + ai[2 * i + 1] * np.exp(-1j * self.k * x[mask])
                return y

            return ai, result

    def plot_outer_problem(self, f: np.ndarray, re=True, im=False, show=True, sampling_points=100):
        """
        Plots the solution outside the resonators
        :param f: bounday conditions
        :param re: bool, whether to plot the real part of the solution
        :param im: bool, whether to plot the imaginary part of the solution
        :param show: bool, whether to show the plot
        :param sampling_points: int, number of sampling points
        :return: (fig, ax)
        """
        ai, result = self.wf(f=f)
        _, xs = self.getPlottingPoints(sampling_points=sampling_points)
        fig, ax = plt.subplots()
        if re:
            ax.plot(xs, np.real(result(xs)), '.', label='real')
        if im:
            ax.plot(xs, np.imag(result(xs)), '.', label='imaginary')
        ax.legend()
        ax.set_title('Solution outside of the resonators')
        ax.set_xlabel('x')
        ax.set_ylabel('w(x)')
        if show:
            fig.show()
        return (fig, ax)

    def getMatcalA(self, alpha: float) -> np.ndarray:
        """
        Returns A matrix from eq 3.2 of Florians paper to determine coefficents of solution
        :param alpha: quasi periodicity
        :return: np.ndarray
        """
        T = self.get_DirichletNeumann_matrix(alpha=alpha)
        left_blocks = [1j * self.kb[i] * np.array(
            [[-np.exp(1j * self.kb[i] * self.xim[i]), np.exp(-1j * self.kb[i] * self.xim[i])],
             [np.exp(1j * self.kb[i] * self.xip[i]), -np.exp(-1j * self.kb[i] * self.xip[i])]])
                       for i in range(self.N)]

        right_blocks = [np.array([[np.exp(1j * self.kb[i] * self.xim[i]), np.exp(-1j * self.kb[i] * self.xim[i])],
                                  [np.exp(1j * self.kb[i] * self.xip[i]), np.exp(-1j * self.kb[i] * self.xip[i])]])
                        for i in range(self.N)]

        A = la.block_diag(*left_blocks) - self.delta * T @ la.block_diag(*right_blocks)
        A = np.array(A)
        return A

    def getu(self, alpha: float):

        A = self.getMatcalA(alpha=alpha)
        assert abs(np.linalg.det(A)) > 1e-3, 'Non singular matrix, not a resonating frequence'
        coef_sol_interior = np.linalg.solve(A, np.zeros(self.N * 2))

        def u(x):
            y = np.zeros_like(x, dtype=complex)
            for i in range(self.N):
                mask = (x >= self.xim[i]) * (x <= self.xip[i])
                y[mask] = coef_sol_interior[2 * i] * np.exp(1j * self.kb[i] * x[mask]) + coef_sol_interior[
                    2 * i + 1] * np.exp(-1j * self.kb[i] * x[mask])
            return y

        f = u(self.xi[:-1])
        coef_sol_exterior, uext = self.wf(f)

        def utot(x):
            return u(x) + uext(x)

        self.coef_interior = coef_sol_interior
        self.coef_exterior = coef_sol_exterior
        self.f = f
        self.u = utot

        return utot

    def plot_u(self, alpha: float, re=True, im=False, int=True, out=True, sampling_points=500):
        u = self.getu(alpha=alpha)
        pointsInt, pointsExt = self.getPlottingPoints(sampling_points=sampling_points)
        fig, ax = plt.subplots()
        if re:
            if int:
                ax.plot(pointsInt, np.real(u(pointsInt)), '.', label='R(u_in)')
            if out:
                ax.plot(pointsExt, np.real(u(pointsExt)), '.', label='R(u_out)')
        if im:
            if int:
                ax.plot(pointsInt, np.imag(u(pointsInt)), '.', label='I(u_in)')
            if out:
                ax.plot(pointsExt, np.imag(u(pointsExt)), '.', label='I(u_out)')
        ax.legend()
        ax.set_title(f'Solution of the quasiperiodic problem for alpha={alpha}')
        ax.set_xlabel('x')
        ax.set_ylabel('u^alpha(x)')
        fig.show()

    def get_capacitance_matrix(self, alpha: float):
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
        V = np.diag(self.li)
        S = np.diag(self.vb)
        # TODO: Do we need to multiply by delta or not?
        Cgen = S ** 2 @ np.linalg.inv(V) @ self.get_capacitance_matrix(alpha=alpha)
        self.Cgen = Cgen
        return Cgen

    def get_resonant_frequencies(self, alpha: float, hermitian=False):
        if not hermitian:
            Cgen = self.get_generalized_capacitance_matrix(alpha=alpha)
        if hermitian:
            Cgen = self.get_generalized_capacitance_matrix(alpha=alpha).conj().T
        # TODO unclear if sort is right here
        if self.N == 2:
            npeigs, vec = np.linalg.eig(Cgen)
            vec = vec / vec[0, :]
            eigenvalues = npeigs
            # tr = np.trace(Cgen)
            # det = np.linalg.det(Cgen)
            #
            # val1 = (tr + np.sqrt(tr ** 2 - 4 * det)) / 2
            # val2 = (tr - np.sqrt(tr ** 2 - 4 * det)) / 2
            # val = np.array([val1, val2], dtype=complex)
            # if hermitian:
            #     eigenvalues = np.conj(val)
            # else:
            #     eigenvalues = val
            # if np.linalg.norm((Cgen @ vec[:, 0]) - eigenvalues[1] * vec[:, 0]) < 1e-7:
            #     print()
            #     old_vec0 = vec[:, 0].copy()
            #     vec[:, 0] = vec[:, 1]
            #     vec[:, 1] = old_vec0
            return eigenvalues, vec
        else:
            pass
            """
            val, vec = np.linalg.eig(Cgen)
            val = np.sort(val)[::-1]
            val = np.sqrt(val)
            mask = np.real(val) < 1e-7
            val[mask] = np.real(val[mask]) + 1j * abs(np.imag(val[mask]))
            return val, vec
            """

    def get_all_resonant_frequencies(self, sample_points=1000):
        alphas = np.linspace(-np.pi / self.L + 1 / sample_points,
                             np.pi / self.L - 1 / sample_points,
                             sample_points)
        for alpha in alphas:
            freq, mode = self.get_resonant_frequencies(alpha=alpha)

            # Same for the hermitian conjugate
            freqH, modeH = self.get_resonant_frequencies(alpha=alpha, hermitian=True)

            IP = np.conj(modeH).T @ mode

            # C = self.get_generalized_capacitance_matrix(alpha=alpha)

            # if np.linalg.norm(np.diag(IP)) < 1e-3:
            #     print(IP)
            for n in range(self.N):
                mode[:, n] = mode[:, n] / np.diag(IP)[n]
            inner_products = (np.conj(modeH).T @ mode) - np.eye(2, 2)
            assert np.linalg.norm(
                inner_products) < 1e-3, f"Left and right eigenvectors are not bi-orthogonal. IP are {inner_products}"
            self.resonant_frequencies[alpha] = freq
            self.resonant_modes_approx[alpha] = mode

            self.resonant_modes_approx_hermitian[alpha] = modeH
        return self.resonant_frequencies

    def plot_resonant_frequencies(self,
                                  type,
                                  re=True,
                                  im=False,
                                  edgemode=False,
                                  save=False):
        assert self.resonant_frequencies, "Resonant frequencies not yet computed. Please run get_all_resonant_frequencies()"
        alphas = np.array(list(self.resonant_frequencies.keys()))
        freq = np.stack(list(self.resonant_frequencies.values()))
        if type == 'sep':
            fig, axs = plt.subplots(self.N, figsize=(10, 7))
            for i in range(self.N):
                ax = axs[i]
                if re:
                    ax.plot(alphas, np.real(freq[:, i]), '.', label=f'w_{i}')
                if im:
                    ax.plot(alphas, np.imag(freq[:, i]), '.', label=f'w_{i}')
            fig.show()
        elif type == 'single':
            fig, ax = plt.subplots(figsize=(10, 7))

            for i in range(self.N):
                if re:
                    ax.plot(alphas, np.real(freq[:, i]), '.', label=f'Re($\\omega_{i}$)', color=cm.Set1.colors[i])
                if im:
                    ax.plot(alphas, np.imag(freq[:, i]), '.', label=f'Im($\\omega_{i}$)', color=cm.Set2.colors[i])
            if edgemode:
                if re:
                    ax.plot(alphas, np.real(np.ones_like(alphas) * self.compute_edge_mode_frequency()), '.',
                            label=f'Re(EdgeMode)', color=cm.Set1.colors[self.N])
                if im:
                    ax.plot(alphas, np.imag(np.ones_like(alphas) * self.compute_edge_mode_frequency()), '.',
                            label=f'Im(EdgeMode)', color=cm.Set2.colors[self.N])

            maxi = np.max(np.abs(np.imag(freq)))
            maxi = max(maxi, np.abs(np.imag(self.compute_edge_mode_frequency()))) * 1.1
            maxr = np.max(np.abs(np.real(freq)))
            maxr = max(maxr, np.abs(np.real(self.compute_edge_mode_frequency()))) * 1.1
            m = max(maxi, maxr)
            ax.set_ylim(-m, m)

            ax.legend(loc='lower left')
            ax.set_title('Resonant frequencies $\\omega^\\alpha$')
            ax.set_xlabel('Quasifrequencies', loc='right')
            ax.set_ylabel('$\\omega$', loc='top')
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            fig.show()
            if save:
                fig.savefig(
                    f"{constants.path_output_figures}ResFreq_alphdep_{'wEig' if edgemode else None}_l0={self.li[0]}_l01={self.lij[0]}_vb0={self.vb[0]}_vb1={self.vb[1]}.pdf",
                    bbox_inches='tight')
        elif type == 'complex':
            fig, ax = plt.subplots(figsize=(10, 7))
            for i in range(self.N):
                ax.plot(np.real(freq[:, i]), np.imag(freq[:, i]), '.', label=f'$\\omega_{i}$')
            if edgemode:
                ax.plot(np.real(self.compute_edge_mode_frequency()), np.imag(self.compute_edge_mode_frequency()), '*',
                        label=f'EdgeMode')
            ax.legend(loc='best')
            ax.set_title('Resonant frequencies $\\omega^\\alpha$')
            ax.set_xlabel('Re', loc='right')
            ax.set_ylabel('Im', loc='top')
            # Move left y-axis and bottim x-axis to centre, passing through (0,0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            fig.show()
            if save:
                fig.savefig(
                    f"{constants.path_output_figures}ResFreq_inC_{'wEig' if edgemode else None}_l0={self.li[0]}_l01={self.lij[0]}_vb0={self.vb[0]}_vb1={self.vb[1]}.pdf",
                    bbox_inches='tight')

    def compute_zak_phases(self):
        assert self.resonant_modes_approx, "Resonant modes not yet computed. Please run get_all_resonant_frequencies()"
        alphas = np.array(list(self.resonant_modes_approx.keys()))
        modes = np.stack(list(self.resonant_modes_approx.values()))
        dalpha = (alphas[1] - alphas[0])
        modesH = np.stack(list(self.resonant_modes_approx_hermitian.values()))

        dmodes = np.diff(modes, axis=0) / dalpha
        dmodesH = np.diff(modesH, axis=0) / dalpha
        # With diff we lose one of the entries, so we just drop it now
        modes = modes[1:]
        alphas = alphas[1:]
        modesH = modesH[1:]
        integrand = np.zeros((modes.shape[0], self.N), dtype=complex)
        integral = np.zeros(self.N, dtype=complex)


        for i in range(modes.shape[0]):
            integrand[i, :] = np.diag(modesH[i].conj().T @ dmodes[i]) + np.diag(modes[i].conj().T @ dmodesH[i])
        for n in range(self.N):
            integral[n] = 1j * np.trapz(integrand[:, n], dx=dalpha) / 2
        plt.plot(alphas, np.real(integrand[:, 0]), '.', label='re')
        plt.plot(alphas, np.imag(integrand[:, 0]), '.', label='imag')
        plt.legend()
        plt.show()
        return integral

    def is_PT(self):
        assert self.N == 2, f"Implemented only for N=2, got {self.N}"
        return np.equal(self.vb[0], np.conj(self.vb[1]))

    def is_broken_PT(self):
        v = self.vb[0] ** 2
        return np.abs(np.imag(v)) * np.sqrt(8) > np.abs(np.real(v))

    def compute_edge_mode_frequency(self):
        assert self.N == 2, "Works only for dimers"
        assert (self.li[0] == self.li[1]) and (self.lij[0] == self.lij[1]), "Works only for boring geometry"
        assert not (self.get_b() is None), "Unbroken PT symmetry, no edge mode"
        v1 = self.vb[0]
        v2 = self.vb[1]
        D = 9 * v1 ** 4 / v2 ** 4 - 14 * v1 ** 2 / v2 ** 2 + 9
        mu = self.delta * v1 ** 2 * self.lij[0] / (2 * (self.get_b() ** 2 * v2 ** 4 - v1 ** 4))
        mu = mu * (v1 ** 2 - v2 ** 2) * (3 * np.sqrt(2) / 2 + 5) * (np.sqrt(D) * v2 ** 2 + 3 * v1 ** 2 - 3 * v2 ** 2)
        return np.sqrt(mu / self.li[0])

    def get_lam(self):
        assert self.N == 2, "Implemented only for N=2"
        l12 = self.lij[0]
        l23 = self.lij[1]
        L = 2 + l12 + l23
        C11 = -(1 / l12 + 1 / l23)
        C12 = lambda alpha: 1 / l12 + np.exp(1j * alpha * L) / l23
        lam1 = C11 + C12(np.pi / L)
        lam2 = 2 * C11
        lam = (lam2 + lam1) / (lam2 - lam1)
        return lam

    def get_b(self):
        l = self.get_lam()
        v1 = self.vb[0]
        v2 = self.vb[1]
        bp = l * (1 - v1 ** 2 / v2 ** 2) + np.sqrt(l ** 2 * (1 - v1 ** 2 / v2 ** 2) ** 2 + 4 * v1 ** 2 / v2 ** 2)
        bp = bp / 2
        bm = l * (1 - v1 ** 2 / v2 ** 2) - np.sqrt(l ** 2 * (1 - v1 ** 2 / v2 ** 2) ** 2 + 4 * v1 ** 2 / v2 ** 2)
        bm = bm / 2
        if np.abs(bm) < 1:
            return bm
        elif np.abs(bp) < 1:
            return bp
        else:
            return None

    def dimers_material_defect_edgemode_matrix_has_constant_eigenvalue(self, tol=1e-3):
        assert self.N == 2, "Implemented only for dimers with material defect"
        np.random.seed(123)

        b = self.get_b()
        if b is None:
            return False, None
        v1 = self.vb[0]
        v2 = self.vb[1]

        A = np.array([[1, b],
                      [b, 1]])
        B = np.array([[1 / v2 ** 2, b / v1 ** 2],
                      [b / v1 ** 2, 1 / v2 ** 2]]) * 1 / self.delta
        alpha_sample = np.random.random(10) * 2 * np.pi / self.L - np.pi / self.L
        eigs = []
        for alpha in alpha_sample:
            C = self.get_capacitance_matrix(alpha)
            M = np.linalg.inv(B) @ C @ A
            eval, _ = np.linalg.eig(M)
            abseval = np.abs(eval)
            ind = np.argsort(abseval, axis=0)
            eval = eval[ind]
            eigs.append(eval)
        eigs = np.array(eigs)
        var = np.abs((np.max(eigs, axis=0) - np.min(eigs, axis=0)) / np.max(eigs, axis=0))
        # print(var)
        varb = var < tol
        if sum(varb) > 0:
            ind = np.argmax(varb)
            return True, eigs[:, ind]
        return False, None

    def get_finite_equivalent(self, n_repetition=50, omega=1, defect=None):
        assert defect in [None, 'parameters'], "Non supported defect type"
        params = self.get_params()
        params['N'] = params['N'] * n_repetition
        params['li'] = list(params['li']) * n_repetition
        params['lij'] = list(params['lij']) * n_repetition
        params['lij'] = params['lij'][:-1]
        if defect == 'parameters':
            assert n_repetition % 2 == 0, f"Need even n_repretition for defect 'paramters', got {n_repetition}"
            params['vb'] = list(np.flip(params['vb'])) * int(n_repetition / 2) + list(params['vb']) * int(
                n_repetition / 2)
        else:
            params['vb'] = list(params['vb']) * n_repetition
        params['omega'] = omega
        fwp = FiniteWaveProblem(**params,
                                uin=lambda x: np.zeros_like(x),
                                duin=lambda x: np.zeros_like(x))
        return fwp


class FiniteWaveProblem:
    """
    Class for 1D Finite high contrast media
    """

    def __init__(self, N: int, li=None, lij=None, v=1, vb=None, omega=None, delta=0.01, uin=None, duin=None):
        """
        Creates a geometrical set up
        :param N: number of resonators
        :param li: list or None, if list: lengths of the resonators, if None a random list will be generated
        :param lij: list or None, if list: distances between the resonators, if None a random list will be generated
        """
        self.N = N
        if lij is None:
            self.lij = self.lij = np.random.randint(1, 5, size=(N - 1, 1))
        else:
            self.lij = np.array(lij)
        if li is None:
            self.li = np.random.randint(1, 5, size=(N,))
        else:
            if not (N == len(li)):
                self.N = len(li)
                warnings.warn(f"\nNeed to have N=len(li), got {N} and {len(li)}. Using the latter", )
            self.li = np.asarray(li)

        self.L = np.sum(self.li) + np.sum(self.lij)
        # l is the array with the distances between the interesting points
        l = np.zeros(2 * self.N - 1)
        l[::2] = self.li
        l[1::2] = self.lij

        self.xi = np.insert(np.cumsum(l), 0, 0)
        self.xiCol = np.column_stack((self.xi[0::2], self.xi[1::2]))
        self.xim = self.xiCol[:, 0]
        self.xip = self.xiCol[:, 1]

        if type(vb) in [list, np.ndarray]:
            assert len(
                vb) == self.N, f"Need to have vb to be a scalar or list with len(vb)=self.N. Got list with len {len(vb)} instead of {self.N}"
        if type(vb) in [float, int, complex]:
            vb = [vb] * self.N
        assert not (type(omega) is None), "Need to specify a omega"
        vb = np.array(vb)
        self.v = v
        self.vb = vb
        self.omega = omega
        self.delta = delta
        self.k = self.omega / self.v if not (omega is None) else None
        self.kb = self.omega / self.vb if not (omega is None) else None
        self.uin = uin  # lambda x: np.exp(1j * self.k * x)
        self.duin = duin  # lambda x: 1j * self.k * np.exp(1j * self.k * x)

        self.resonant_frequencies = {}
        self.resonant_modes_approx = {}
        self.resonant_modes_approx_hermitian = {}

    def set_omega(self, omega):
        self.omega = omega
        self.k = self.omega / self.v if not (omega is None) else None
        self.kb = self.omega / self.vb if not (omega is None) else None

    def get_DirichletNeumann_matrix(self) -> np.ndarray:
        """
        Returns the Dirichlet to Neumann map in matrix form as given in eq 2.7 of Florian's paper
        :param alpha: quasi periodicity
        :return: np.array
        """

        def Ak(l, k):
            return np.asarray([[-k * np.cos(k * l) / np.sin(k * l), k / np.sin(k * l)],
                               [k / np.sin(k * l), -k * np.cos(k * l) / np.sin(k * l)]])

        T = la.block_diag(1j * self.k,
                          *[np.asarray(Ak(l, self.k)) for l in self.lij],
                          1j * self.k)
        T = np.array(T, dtype=complex)
        return T

    def getPlottingPoints(self, sampling_points=100, long_range=10):
        """
        Returns a sampling of points both inside and outside the resonators
        :param sampling_points: number of sample to take in intervals
        :return: np.array, np.array, points_inside, points outside
        """
        factor = 0.002 if self.omega < 0.005 else 0.01
        pointsExt = []  # [np.linspace(-3 * np.pi / factor, -10, 500)]
        pointsExt.append(np.linspace(-long_range, 0, sampling_points * int(long_range / self.li[0])))
        pointsExt = pointsExt + [np.linspace(self.xip[i], self.xim[i + 1], sampling_points) for i in range(self.N - 1)]
        pointsExt.append(
            np.linspace(self.xip[-1], self.xip[-1] + long_range, sampling_points * int(long_range / self.li[-1])))
        # pointsExt.append(np.linspace(self.xip[-1] + 10, self.xip[-1] + 3 * np.pi / factor, 500))
        pointsExt = np.concatenate(pointsExt)

        pointsInt = [np.linspace(self.xim[i], self.xip[i], sampling_points) for i in range(self.N)]
        pointsInt = np.concatenate(pointsInt)

        return pointsInt, pointsExt

    def wf(self, f):
        """
        Solution to external problem
        :param f: values of the boundary conditions
        :return: (ai, result) where ai are the coefficients of the solution and result is a function giving the solution
        in the fundamental domain
        """
        if f is None:
            f = (np.random.random(2 * self.N) - 1) * 2
        assert len(f) == 2 * self.N, f"Need 2*N boundary condition, got {len(f)} instead of {2 * self.N}"
        if self.k == 0:
            def result(x):
                if x < self.xim[0]:
                    return f[0]  # * np.exp(-1j*self.k*(x-self.xim[0]))
                elif x > self.xip[-1]:
                    return f[-1]  # * np.exp(1j*self.k*(x-self.xip[-1]))
                for i in range(self.N - 1):
                    if x > self.xip[i] and x < self.xim[i + 1]:
                        return f[2 * i] + (f[2 * i + 1] - f[2 * i]) / (self.xim[i + 1] - self.xip[i]) * (
                                x - self.xip[i])
                return np.nan

            return [], result
        else:
            def exteriorblock(l, k, xp, xpm):
                return -1 / (2j * np.sin(k * l)) * np.asarray([[np.exp(-1j * k * xpm), -np.exp(-1j * k * xp)],
                                                               [-np.exp(1j * k * xpm), np.exp(1j * k * xp)]])

            TBlocks = la.block_diag(*(exteriorblock(l, self.k, xp, xpm)
                                      for (l, xp, xpm) in zip(self.lij, self.xip[:-1], self.xim[1:])))
            ai = TBlocks.dot(f[1:-1])

            def result(x: np.ndarray):
                y = np.zeros(x.shape, dtype=complex)
                mask = (x < self.xim[0])
                y[mask] = f[0] * np.exp(-1j * self.k * (x[mask] - self.xim[0]))
                mask = (x > self.xip[-1])
                y[mask] = f[-1] * np.exp(1j * self.k * (x[mask] - self.xip[-1]))
                for i in range(self.N - 1):
                    mask = (x > self.xip[i]) * (x < self.xim[i + 1])
                    y[mask] = ai[2 * i] * np.exp(1j * self.k * x[mask]) \
                              + ai[2 * i + 1] * np.exp(-1j * self.k * x[mask])
                return y

            return ai, result

    def plot_outer_problem(self, f: np.ndarray, re=True, im=False, show=True, sampling_points=100):
        """
        Plots the solution outside the resonators
        :param f: bounday conditions
        :param re: bool, whether to plot the real part of the solution
        :param im: bool, whether to plot the imaginary part of the solution
        :param show: bool, whether to show the plot
        :param sampling_points: int, number of sampling points
        :return: (fig, ax)
        """
        ai, result = self.wf(f=f)
        _, xs = self.getPlottingPoints(sampling_points=sampling_points)
        fig, ax = plt.subplots()
        if re:
            ax.plot(xs, np.real(result(xs)), '.', label='real')
        if im:
            ax.plot(xs, np.imag(result(xs)), '.', label='imaginary')
        ax.legend()
        ax.set_title('Solution outside of the resonators')
        ax.set_xlabel('x')
        ax.set_ylabel('w(x)')
        if show:
            fig.show()
        return (fig, ax)

    def getMatcalA(self) -> np.ndarray:
        """
        Returns A matrix from eq 3.2 of Florians paper to determine coefficents of solution
        :param alpha: quasi periodicity
        :return: np.ndarray
        """
        T = self.get_DirichletNeumann_matrix()
        left_blocks = [1j * self.kb[i] * np.array(
            [[-np.exp(1j * self.kb[i] * self.xim[i]), np.exp(-1j * self.kb[i] * self.xim[i])],
             [np.exp(1j * self.kb[i] * self.xip[i]), -np.exp(-1j * self.kb[i] * self.xip[i])]])
                       for i in range(self.N)]

        right_blocks = [np.array([[np.exp(1j * self.kb[i] * self.xim[i]), np.exp(-1j * self.kb[i] * self.xim[i])],
                                  [np.exp(1j * self.kb[i] * self.xip[i]), np.exp(-1j * self.kb[i] * self.xip[i])]])
                        for i in range(self.N)]

        A = la.block_diag(*left_blocks) - self.delta * T @ la.block_diag(*right_blocks)
        A = np.array(A)

        uinxi = self.uin(self.xi)
        duinp = self.duin(self.xip)
        duinm = self.duin(self.xim)

        Tuinxi = T @ uinxi
        Tuinp = Tuinxi[1::2]
        Tuinm = Tuinxi[::2]

        RHS = np.zeros((2 * self.N, 1), dtype=complex)
        for i in range(self.N):
            RHS[2 * i] = -self.delta * Tuinm[i] - self.delta * duinm[i]
            RHS[2 * i + 1] = -self.delta * Tuinp[i] - self.delta * duinp[i]
        return A, RHS

    def uj(self, x, j):
        y = np.zeros_like(x, dtype=complex)
        mask = (x >= self.xim[j]) * (x <= self.xip[j])
        y[mask] = 1 / self.li[j] + self.omega / (self.vb[j] ** 2 * self.li[j] ** 2)
        return y

    def getu(self):
        A, RHS = self.getMatcalA()
        tol = 1e-7
        resonant = abs(np.linalg.det(A)) < tol
        # resonant = False
        # if resonant:
        #     print(f"This is a resonant frequency:  log10(abs(det(A))={np.log10(np.abs(np.linalg.det(A)))}")
        #     ker = la.null_space(A)
        #     print(A.shape)
        #     print(ker.shape)
        #     #exit()
        #     freqs, vmodes = self.getResonatFrequencies()
        #     arg_min_omega = np.argmin(np.abs(freqs - self.omega))
        #     vmode = vmodes[arg_min_omega]
        #
        #     def u(x):
        #         y = np.zeros_like(x, dtype=complex)
        #         for i in range(self.N):
        #             mask = (x >= self.xim[i]) * (x <= self.xip[i])
        #             y[mask] = self.uj(x[mask], j=i) * vmode[i]
        #         return y
        # else:
        if resonant:
            print(f"This is a resonant frequency:  log10(abs(det(A))={np.log10(np.abs(np.linalg.det(A)))}")
            ker = la.null_space(A, rcond=tol)
            print(ker.shape)
            coef_sol_interior = ker[:, 0]
        else:
            print(f"This is NOT a resonant frequency:  log10(abs(det(a))={np.log10(np.abs(np.linalg.det(A)))}")
            coef_sol_interior = np.linalg.solve(A, RHS)

        def u(x):
            y = np.zeros_like(x, dtype=complex)
            for i in range(self.N):
                mask = (x >= self.xim[i]) * (x <= self.xip[i])
                y[mask] = coef_sol_interior[2 * i] * np.exp(1j * self.kb[i] * x[mask]) + coef_sol_interior[
                    2 * i + 1] * np.exp(-1j * self.kb[i] * x[mask])
            return y

        f = u(self.xi)
        coef_sol_exterior, uext = self.wf(f)

        def utot(x):
            return u(x) + uext(x)

        self.coef_interior = coef_sol_interior if not (resonant) else None
        self.coef_exterior = coef_sol_exterior
        self.f = f
        self.u = utot

        return utot

    def plot_u(self, re=True, im=False, int=True, out=True, sampling_points=500, long_range=10):
        u = self.getu()
        pointsInt, pointsExt = self.getPlottingPoints(sampling_points=sampling_points, long_range=long_range)
        fig, ax = plt.subplots()
        size = 1
        if re:
            if int:
                ax.scatter(pointsInt, np.real(u(pointsInt)), s=size, label='Re(u_in)')
            if out:
                ax.scatter(pointsExt, np.real(u(pointsExt)), s=size, label='Re(u_out)')
        if im:
            if int:
                ax.scatter(pointsInt, np.imag(u(pointsInt)), s=size, label='Im(u_in)')
            if out:
                ax.scatter(pointsExt, np.imag(u(pointsExt)), s=size, label='Im(u_out)')
        ax.legend()
        ax.set_title(f'Solution of the finite problem')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(x)$')
        ax.set_yscale('symlog')
        fig.show()

    def getApproxCapacitanceMatrix(self):
        d1 = np.concatenate(([1 / self.lij[0]],
                             1 / self.lij[:-1] + 1 / self.lij[1:], [1 / self.lij[-1]]))
        d2 = -1 / self.lij
        C = np.diag(d1) + np.diag(d2, 1) + np.diag(d2, -1)
        return C

    def getResonatFrequencies(self):
        assert np.max(np.abs(
            self.vb - np.ones_like(self.vb) * self.vb[0])) < 1e-3, "Works only for constant vb across the resonators"
        vb = self.vb[0]
        freqs = np.array([0, -2j * self.delta * vb ** 2 / (self.v * sum(self.li))])
        vmodes = np.ones(self.N)
        vmodes = np.append(vmodes, np.ones(self.N) / np.sqrt(sum(self.li)))
        if self.N > 1:
            C = self.getApproxCapacitanceMatrix()
            V = np.diag(self.li)
            Vhalf = np.sqrt(np.linalg.pinv(V))
            lambdas, vmodes = np.linalg.eigh(Vhalf.dot(C).dot(Vhalf))
            vmodes = Vhalf.dot(vmodes)
            B = np.diag([1] + [0] * (self.N - 2) + [1])
            aiBai = np.diag(vmodes[:, 1:].T.dot(B.dot(vmodes[:, 1:])))
            freqs = np.append(freqs, vb * np.sqrt(self.delta) * np.sqrt(lambdas[1:])
                              - 1j * self.delta * vb ** 2 / (2 * self.v) * aiBai)
            freqs = np.append(freqs, - vb * np.sqrt(self.delta) * np.sqrt(lambdas[1:])
                              - 1j * self.delta * vb ** 2 / (2 * self.v) * aiBai)
        else:
            vmodes = vmodes[:, None]

        return freqs, vmodes


def get_good_material_paramters_edgemode_PT():
    tol = 1e-5
    xx = np.linspace(-10, 10, 1000)
    v1s = []
    for x in tqdm(xx):
        for y in xx:
            v0 = x + 1j * y
            v1 = v0 ** 2
            v2 = (x - 1j * y - 5j) ** 2

            D = np.sqrt(9 * v1 ** 2 - 14 * v1 * v2 + 9 * v2 ** 2)
            betas = [-1, 1]
            sigmas = [-1, 1]
            for beta in betas:
                for sigma in sigmas:
                    condition = 3 * (v1 - v2) + beta * D + sigma * np.sqrt(2) * np.sqrt(
                        9 * (v1 ** 2 + v2 ** 2) + 3 * D * (v1 - v2) - 16 * v1 * v2)
                    if np.abs(condition) < tol:
                        v1s.append(v1)


if __name__ == '__main__':
    pass
