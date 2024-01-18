import math
import numpy as np
import scipy.linalg as linalg
from exceptions import RealizabilityErr


def adaptiveWheeler(m, n, upperRange, time=None):

    a, b, zeta = zeta_chebyshev(m)

    # Reduce number of nodes (n) if the moments are unrealizable
    _, rN = nodeReduction(zeta, n, time)

    if rN == 1:
        w = np.zeros(n)
        x = np.zeros(n)

        w[0] = m[0]
        x[0] = m[1] / m[0]

        for i in range(1, n):
            w[i] = 0.0
            x[i] = 1e-20  # This size is not important since the weight is zero

        return w, x, rN

    for k in range(rN, 1, -1):

        # Reduced "a" and "b" arrays (in case of unrealizable moment set)
        aR = np.zeros(k)
        bR = np.zeros(k-1)

        for i in range(k-1):
            aR[i] = a[i]
            bR[i] = -1.0 * (b[i]**0.5)

        aR[k-1] = a[k-1]

        D, V = linalg.eigh_tridiagonal(aR, bR)

        w = np.zeros(n)
        x = np.zeros(n)

        nodes_in_range = True
        for i in range(0, k):
            w[i] = (V[0, i]**2)*m[0]
            if w[i] < 0.0:
                raise RealizabilityErr(m)
            elif w[i] > 1e-15:
                x[i] = D[i]
                if x[i] < 0.0:
                    raise RealizabilityErr(m, x[i], negNode=True)
                if x[i] > upperRange:
                    nodes_in_range = False
            else:
                x[i] = 1e-20

        if nodes_in_range:
            for i in range(k, n):
                w[i] = 0.0
                x[i] = 1e-20  # This size is not important since the weight is zero

            return w, x, k

    w = np.zeros(n)
    x = np.zeros(n)

    w[0] = m[0]
    x[0] = m[1] / m[0]

    for i in range(1, n):
        w[i] = 0.0
        x[i] = 1e-20  # This size is not important since the weight is zero

    return w, x, 1


def wheeler(m, n):

    sigma = np.zeros([2*n + 1, 2*n + 1])

    for i in range(1, 2*n + 1):
        sigma[1, i] = m[i-1]

    a = np.zeros(n)
    b = np.zeros(n)

    a[0] = m[1]/m[0]

    b[0] = 0

    for i in range(2, n+1):
        for j in range(i, 2*n - i + 2):
            sigma[i, j] = sigma[i - 1, j + 1] - a[i - 2]*sigma[i - 1, j] \
                - b[i - 2]*sigma[i - 2, j]

        a[i - 1] = sigma[i, i + 1]/sigma[i, i] \
            - sigma[i - 1, i]/sigma[i - 1, i - 1]
        b[i - 1] = sigma[i, i]/sigma[i - 1, i - 1]

    negSqrt_b = np.zeros(n-1)

    for i in range(0, n-1):
        b_i = b[i + 1]

        if b_i < 0:
            raise RealizabilityErr(m)

        negSqrt_b[i] = -math.sqrt(b_i)

    D, V = linalg.eigh_tridiagonal(a, negSqrt_b)

    w = np.zeros(n)
    x = np.zeros(n)

    for i in range(0, n):
        w[i] = (V[0, i]**2)*m[0]
        if w[i] < 0:
            raise RealizabilityErr(m)

        x[i] = D[i]
        if x[i] < 0:
            raise RealizabilityErr(m, x[i], negNode=True)

    return w, x


def PD(m, n):

    fIndex = 2*n + 1

    P = np.zeros(int(fIndex*(fIndex + 1)/2))

    P[0] = 1.0

    for i in range(fIndex - 1):
        P[i + fIndex] = ((-1.0)**i) * m[i]

    zeta = np.zeros(fIndex - 1)

    for j in range(2, fIndex):
        fb = int(j*(2*fIndex + 1 - j) / 2)
        fb_1 = fb - fIndex + j - 1
        fb_2 = fb_1 - fIndex + j - 2

        for i in range(fIndex - j):
            P[i + fb] = P[fb_1] * P[i + 1 + fb_2] - P[fb_2] * P[i + 1 + fb_1]

            product = P[fb_1] * P[fb_2]

            if (product > 0):
                zeta[j - 1] = P[fb] / product

    a = np.zeros(n)
    b = np.zeros(n-1)

    for i in range(n - 1):
        a[i] = zeta[2*i + 1] + zeta[2*i]
        b[i] = -1.0 * np.sqrt(zeta[2*i + 2] * zeta[2*i + 1])

    a[n-1] = zeta[fIndex - 2] + zeta[fIndex - 3]


    D, V = linalg.eigh_tridiagonal(a, b)

    w = np.zeros(n)
    x = np.zeros(n)

    for i in range(0, n):
        w[i] = (V[0, i]**2)*m[0]
        if w[i] < 0:
            raise RealizabilityErr(m)

        x[i] = D[i]
        if x[i] < 0:
            raise RealizabilityErr(m, x[i], negNode=True)

    return w, x


def zeta_chebyshev(m):

    nMoms = len(m)

    n = int(nMoms / 2)

    a = np.zeros(n)
    a[0] = m[1]/m[0]

    b = np.zeros(n)

    zeta = np.zeros(nMoms)
    zeta[1] = a[0]

    sigma = np.zeros((nMoms, nMoms))

    sigma[0, :] = m

    for l in range(1, nMoms - 1):
        sigma[1, l] = sigma[0, l+1] - a[0]*sigma[0, l]

    for k in range(1, n):
        b[k] = sigma[k, k] / sigma[k-1, k-1]
        a[k] = sigma[k, k+1] / sigma[k, k] - sigma[k-1, k] / sigma[k-1, k-1]

        zeta[2*k] = b[k] / zeta[2*k - 1]
        zeta[2*k + 1] = a[k] - zeta[2*k]

        for l in range(k+1, nMoms - k - 1):
            sigma[k+1, l] = sigma[k, l+1] - a[k]*sigma[k, l] - b[k]*sigma[k-1, l]

    if (nMoms % 2) > 0:
        b = np.append(b, sigma[n, n] / sigma[n-1, n-1])
        zeta[-1] = b[n] / zeta[2*n - 1]

    # print(a, b, zeta)

    return a, b[1:], zeta[1:]


def interior_moment_space_stieltjes(m):

    _, _, zeta = zeta_chebyshev(m)

    N_mN = 0
    for z in zeta:
        if z > 1e-12:
            N_mN += 1
        else:
            break

    if N_mN < len(m) - 1:
        return False, N_mN
    else:
        return True, N_mN


def interior_moment_space_hausdorff(m):

    _, _, zeta = zeta_chebyshev(m)

    canonical_moments = np.zeros(zeta.size)

    canonical_moments[0] = m[1] / m[0]

    for k in range (1, canonical_moments.size):
        canonical_moments[k] = zeta[k] / (1 - canonical_moments[k-1])

    N_mN = 0
    for p in canonical_moments:
        if p >= 1.0 or p < 1e-12:
            break
        else:
            N_mN += 1

    if N_mN < len(m) - 1:
        return False, N_mN
    else:
        return True, N_mN


def nodeReduction(zeta, n, time):

    isRealizable = True

    rN = 1

    if (zeta[0] <= 0.0):
        # print("Warning:\nNumber of nodes is reduced to " +
        #       "{0} at time {1}\n".format(rN, time))
        return False, rN

    for i in range(2, 2*n-1, 2):
        if (zeta[i] <= 0 or zeta[i-1] <= 0):
            isRealizable = False
            # print("Warning:\nNumber of nodes is reduced to " +
            #       "{0} at time {1}\n".format(rN, time))
            break
        rN += 1

    return isRealizable, rN


if __name__ == "__main__":

    nNodes = 3

    x = [0.2, 0.7, 0.99]
    w = [1.5, 2.5, 1.1]

    nMoms = 7

    moments = np.zeros(nMoms)
    for k in range(nMoms):
        moment = 0
        for j in range(len(x)):
            if k > 0:
                moment += w[j]*(x[j]**k)
            else:
                moment += w[j]
        moments[k] = moment

    # moments[nMoms - 1] *= 1.0001

    upperRange = 10.0001

    weights, abscissas, rNNodes = adaptiveWheeler(moments, nNodes, upperRange)

    print(weights, abscissas, rNNodes)

    print(interior_moment_space_stieltjes(moments))

    print(interior_moment_space_hausdorff(moments))
