import math
import numpy as np
from sympy import Poly, exp
from sympy.abc import sigma, xi
from sympy.matrices import zeros
from inv_algorithms import adaptiveWheeler
# from scipy.optimize import ridder
from scipy.special import beta
from scipy.optimize import brentq, brent
from scipy.special import roots_jacobi, roots_genlaguerre


class EQMOM(object):

    def __init__(self, nMoments, nGaussQuadPoints, kernel):

        if nMoments < 3:
            print("Error:\nEQMOM requires at least 3 moments")
            exit()
        if nMoments % 2 == 0:
            print("Warning:\nlast moment is ignored")
            self.nMoments = nMoments - 1
            self.nNodes = int(nMoments / 2 - 1)
        else:
            self.nMoments = nMoments
            self.nNodes = int((nMoments - 1) / 2)

        if isinstance(nGaussQuadPoints, list) and \
            len(nGaussQuadPoints) == self.nNodes:
            self.nGaussQuadPoints = nGaussQuadPoints
        else:
            print("Error:\nNumber of Gaussian quadrature points should be",
                "specified as a list with {} elements".format(self.nNodes))
            exit()

        self.kernel = kernel
        self.A, self.A_inv = transform_matrix(
            self.nMoments, kernel=self.kernel)
        if self.kernel == 'beta':
            self.lowerRange = 0.0
            self.upperRange = 1.0001
        elif self.kernel == 'gamma':
            self.lowerRange = 0.0
            self.upperRange = np.inf

    def moment_inversion(self, moments):

        self.moments = moments

        if self.nMoments > 3:
            kernel_param_max = min(
                moments[2]/moments[1] - moments[1]/moments[0],
                moments[3]/moments[2] - moments[2]/moments[1])
        else:
            kernel_param_max = moments[2]/moments[1] - moments[1]/moments[0]

        try:
            kernel_param = brentq(self.func, 0, 100*kernel_param_max, maxiter=50)

        except:
            print("Warning: root finding failed!")
            kernel_param = brent(
                self.func_abs, brack=(0, 100*kernel_param_max), maxiter=50)

        A_inv = np.array(
            self.A_inv.subs(sigma, kernel_param)).astype(np.float64)

        degenerated_moments = np.dot(A_inv[:-1, :-1], self.moments[:-1])

        weights, abscissas, rNNodes = adaptiveWheeler(
            degenerated_moments, self.nNodes, self.upperRange)

        return weights, abscissas, kernel_param

    def func(self, kernel_param):

        A_inv = np.array(
            self.A_inv.subs(sigma, kernel_param)).astype(np.float64)

        degenerated_moments = np.dot(A_inv[:-1, :-1], self.moments[:-1])

        weights, abscissas, rNNodes = adaptiveWheeler(
            degenerated_moments, self.nNodes, self.upperRange)

        order_2N = 2*rNNodes
        degenerated_moment_2N = 0
        for i in range(rNNodes):
            degenerated_moment_2N += weights[i] * abscissas[i]**order_2N

        moment_2N = float(np.dot(
            self.A[order_2N, :order_2N + 1].subs(sigma, kernel_param),
            np.append(degenerated_moments[:order_2N],
                      degenerated_moment_2N))[0])

        return self.moments[order_2N] - moment_2N

    def func_abs(self, kernel_param):
        return abs(self.func(kernel_param))

    def evaluate_quadrature(self, moments, points):

        weights, abscissas, kernel_param = self.moment_inversion(moments)

        q_n = evaluate_quadrature(
            weights, abscissas, kernel_param, 'beta', points)

        return q_n

    def point_representation(self, moments, x_min, x_max):

        weights, abscissas, kernel_param = self.moment_inversion(moments)

        weights_pr = list()
        abscissas_pr = list()
        for i, (weight, abscissa) in enumerate(zip(weights, abscissas)):
            if weight > 0:
                if self.kernel == 'beta':
                    lambda_coeff = abscissa / kernel_param
                    mu_coeff = (1 - abscissa) / kernel_param

                    G_J_x, G_J_w = roots_jacobi(
                        self.nGaussQuadPoints[i], mu_coeff - 1, lambda_coeff - 1)

                    G_J_x = (G_J_x + 1) / 2

                    G_J_w *= (
                        0.5**(lambda_coeff + mu_coeff - 1)
                        ) / beta(lambda_coeff, mu_coeff)

                    weights_pr.append(weight * G_J_w)
                    abscissas_pr.append(G_J_x)

                elif self.kernel == 'gamma':
                    lambda_coeff = abscissa / kernel_param

                    G_L_x, G_L_w = roots_genlaguerre(
                        self.nGaussQuadPoints[i], lambda_coeff - 1)

                    G_L_x *= kernel_param
                    G_L_w /= math.gamma(lambda_coeff)

                    weights_pr.append(weight * G_L_w)
                    abscissas_pr.append(G_L_x)

        return weights_pr, abscissas_pr


def transform_matrix(order, kernel):

    if kernel not in ['gamma', 'beta']:
        print("Error:\nonly gamma and beta kernels are available")
        exit()

    A = zeros(order)

    A[0, 0] = 1

    for k in range(1, order):
        G_k = 1
        for i in range(k):
            G_k *= 1 + i*sigma
        G_k = Poly(G_k, sigma)

        for i in range(k):
            if kernel=='gamma':
                A[k, k - i] = G_k.coeff_monomial(sigma**i) * sigma**i
            else:
                A[k, k - i] = G_k.coeff_monomial(sigma**i) * sigma**i / G_k

    A_inv = A.inv(method='LU')

    return A, A_inv

def evaluate_quadrature(weights, abscissas, kernel_param, kernel, points):

    q = quadrature(weights, abscissas, kernel_param, kernel)

    q_n = list()
    for x in points:
        q_n.append(q.subs(xi, x))

    return q_n

def quadrature(weights, abscissas, kernel_param, kernel):

    q = 0
    for weight, abscissa in zip(weights, abscissas):
        if weight > 0:
            q += weight * kernel_func(abscissa, kernel_param, kernel)

    return q

def kernel_func(xi_ave, kernel_param, kernel):

    if kernel == 'gamma':
        lambda_coeff = xi_ave / kernel_param

        return xi**(lambda_coeff - 1) * exp(-xi / kernel_param) / (
            kernel_param**lambda_coeff * math.gamma(lambda_coeff))

    elif kernel == 'beta':
        lambda_coeff = xi_ave / kernel_param
        mu_coeff = (1 - xi_ave) / kernel_param

        return xi**(lambda_coeff - 1) * (1 - xi)**(mu_coeff - 1) / \
            beta(lambda_coeff, mu_coeff)

    else:
        print("Error:\nonly gamma and beta kernels are available")
        exit()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.integrate import quad

    sigma_g = 2.5
    D_g = 1e-5

    f_ = lambda D: math.exp(-math.log(D / D_g)**2 / 2 / math.log(sigma_g)**2) \
        / D / math.log(sigma_g) / math.sqrt(2*math.pi)

    meanSize_0 = D_g  # average size

    minSize_0 = 5e-7
    maxSize_0 = 1e-4

    normalization_factor = quad(f_, minSize_0, maxSize_0)[0]

    f = lambda x: f_(x) / normalization_factor

    points = np.linspace(minSize_0, maxSize_0, 1000)

    nMoments = 11

    moments = list()
    for i in range(nMoments):
        g = lambda x: x**i * f(x)
        moments.append(quad(g, minSize_0, maxSize_0)[0])

    diffMaxMin_0 = maxSize_0 - minSize_0

    f_scaled = lambda x: diffMaxMin_0 * f(diffMaxMin_0*x + minSize_0)

    moments_scaled = list()
    for i in range(nMoments):
        g = lambda x: x**i * f_scaled(x)
        moments_scaled.append(quad(g, 0, 1)[0])
    print(moments_scaled)

    nGaussQuadPoints = [nMoments + 1]*int((nMoments - 1)/2)

    pbm_solver = EQMOM(nMoments, nGaussQuadPoints, 'beta')

    weights, abscissas, kernel_param = pbm_solver.moment_inversion(
        np.array(moments_scaled))

    print(weights, abscissas, kernel_param)

    points = np.linspace(0, 1, 1000)
    q_n = evaluate_quadrature(weights, abscissas, kernel_param, 'beta', points)

    plt.plot(
        (points*diffMaxMin_0 + minSize_0)/D_g,
        [q_n_i*D_g/diffMaxMin_0 for q_n_i in q_n], ':')

    points = D_g * np.linspace(0.001, 5, 1000)
    f_n = list()
    for x in points:
        f_n.append(f(x)*D_g)
    plt.plot(points/D_g, f_n)

    plt.xlim([0, 5])
    plt.show()
