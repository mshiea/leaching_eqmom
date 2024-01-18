import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad, simpson
from eqmom import EQMOM, evaluate_quadrature

# uncomment to enable widgets when running in interactive window
# %matplotlib widget

plt.rc('text', usetex=True)
plt.rc('font', size=12)
plt.rc('axes', linewidth=1.0, labelpad=5.0)
plt.rc('lines', linewidth=2.0)
plt.rc('xtick.major', size=5.5, width=1.25)
plt.rc('ytick.major', size=5.5, width=1.25)


def main(argv, C_A_0, nMoments, moments_0, m0_scale, tau, minSize_0, maxSize_0,
         meanSize_0, nGaussQuadPoints, kernel, params):

    b = params["b"]

    rho_B_s = params["rho_B_s"] / C_A_0

    dl_dt = lambda x, X, c, t: \
        params["dl_dt"](x*meanSize_0, X, c*C_A_0, t) * tau / meanSize_0

    print("Time scale (hr):", tau)

    G = lambda xd, X, xmin, xmax, c, t: params["G"](
        dl_dt, xd, X, xmin, minSize_0, xmax, maxSize_0, xmax - xmin, c,
        t) # arguments except X, minSize_0 & maxSize_0 are dimensionless

    # instantiate an object of the eqmom class
    pbm_solver = EQMOM(nMoments, nGaussQuadPoints, kernel)

    weights_pr, abscissas_pr = pbm_solver.point_representation(
        moments_0, -1, 1)

    abscissas_pr = np.concatenate(abscissas_pr)
    weights_pr = np.concatenate(weights_pr)

    abscissas_0 = abscissas_pr*(maxSize_0 - minSize_0) + minSize_0

    n_abscissas_pr = sum(nGaussQuadPoints)

    y0 = abscissas_pr.tolist() + [
        C_A_0/C_A_0, minSize_0/meanSize_0, maxSize_0/meanSize_0]

    t_end = params["t_eval"][-1]
    t_eval = params["t_eval"]

    # solve the dimensionless equation defined in the functio "derivative"
    sol = solve_ivp(
        derivative, [0, t_end], y0, t_eval=t_eval, events=[event_zero_conc],
        args=(abscissas_0, n_abscissas_pr, weights_pr, m0_scale, G, dl_dt,
              meanSize_0, minSize_0, maxSize_0, pbm_solver, rho_B_s, b),
        **params["ode_solver_options"])

    print("Final concentration:", sol.y[n_abscissas_pr, -1])

    print("Final bounds:", sol.y[n_abscissas_pr + 1, -1],
          sol.y[n_abscissas_pr + 2, -1])

    abscissas_pr_final = sol.y[:n_abscissas_pr, -1]

    moments = np.zeros(nMoments)
    for k in range(nMoments):
        moment = 0
        for weight, abscissa in zip(weights_pr, abscissas_pr_final):
            moment += weight * (abscissa ** k)
        moments[k] = moment
    print("Final scaled moments:", moments)

    weights, abscissas, kernel_param = pbm_solver.moment_inversion(moments)
    print("Final time quadrature:", weights, abscissas, kernel_param)

    q_n = list()
    m_n = list()
    points = list()
    for i in range(sol.y.shape[1]):

        t = sol.t[i]

        abscissas_pr = sol.y[:n_abscissas_pr, i]

        moments = np.zeros(nMoments)
        for k in range(nMoments):
            moment = 0
            for weight, abscissa in zip(weights_pr, abscissas_pr):
                moment += weight * (abscissa ** k)
            moments[k] = moment

        weights, abscissas, kernel_param = pbm_solver.moment_inversion(moments)

        minSize_t = sol.y[n_abscissas_pr + 1, i] * meanSize_0
        maxSize_t = sol.y[n_abscissas_pr + 2, i] * meanSize_0

        diffMaxMin_t = maxSize_t - minSize_t

        points_t = np.linspace(
            max(minSize_t, 0)+abs(minSize_t*.0001), maxSize_t*0.9999, 1000)

        p_eval_t = (points_t - minSize_t) / diffMaxMin_t

        q_n_scaled_t = evaluate_quadrature(
            weights, abscissas, kernel_param, kernel, p_eval_t)

        q_n_t = [
            float(element*m0_scale/diffMaxMin_t) for element in q_n_scaled_t]

        points.append(points_t)
        q_n.append(q_n_t)

        x_min = max(0, -minSize_t/diffMaxMin_t)

        m_t = np.zeros(nMoments)
        for k in range(nMoments):
            m_k_t = 0
            for x, w in zip(abscissas_pr, weights_pr):
                if x_min <= x:
                    x_orig = x * diffMaxMin_t + minSize_t
                    m_k_t += w * (x_orig**k)
            m_t[k] = m_k_t*m0_scale

        m_n.append(m_t)

    C_A = sol.y[n_abscissas_pr, :] * C_A_0
    minSize = sol.y[n_abscissas_pr + 1] * meanSize_0
    maxSize = sol.y[n_abscissas_pr + 2] * meanSize_0

    return q_n, C_A, minSize, maxSize, points, sol.t*tau, m_n


def derivative(
        t, y, abscissas_0, n_abscissas, weights, m0_scale, G, dl_dt,
        meanSize_0, minSize_0, maxSize_0, pbm_solver, rho_B_s, b):
    print("time", t)

    dy = np.zeros(n_abscissas + 3)

    # Variables are dimensionless
    abscissas = y[:n_abscissas]
    C_A = y[n_abscissas]
    minSize = y[n_abscissas + 1]
    maxSize = y[n_abscissas + 2]

    for i, (x, X) in enumerate(zip(abscissas, abscissas_0)):
        dy[i] = G(x, X, minSize, maxSize, C_A, t)

    diffMaxMin = maxSize - minSize

    x_min = max(0, -minSize/diffMaxMin)

    dm3_dt = 0.0
    for x, X, w in zip(abscissas, abscissas_0, weights):
        if x_min <= x:
            x_orig = x * diffMaxMin + minSize
            dm3_dt += w * (x_orig**2) * dl_dt(x_orig, X, C_A, t)
    dm3_dt *= 3 * m0_scale * (meanSize_0**3)

    dy[n_abscissas] = math.pi * rho_B_s * dm3_dt / 6 / b

    dy[n_abscissas + 1] = dl_dt(minSize, minSize_0, C_A, t)
    dy[n_abscissas + 2] = dl_dt(maxSize, maxSize_0, C_A, t)

    return dy

# Stop the simulation if the concentration decreases below a threshold value
def event_zero_conc(
        t, y, abscissas_0, n_abscissas, weights, m0_scale, G, dl_dt, meanSize,
        minSize_0, maxSize_0, pbm_solver, rho_B_s, b):

    C_A = y[n_abscissas]

    return C_A - 1e-6

event_zero_conc.terminal = True


if __name__ == "__main__":

    params = {
        "b": 1,
        "k_s": 4e-5,  # first-order rate constant for surface reaction
        "D_e": 8e-9,  # diffusion coefficient in the product layer
        "rho_B_s": 10,  # concentration in the solid
        "t_eval": [0, 0.4, 1, 2],
        "num_t_eval": 10
    }

    # Shrinking-core model in the surface-reaction controlling regime
    params["dl_dt"] = lambda x, X, c, t: \
        -2 * params["b"] * c * params["k_s"] / params["rho_B_s"]

    # rate for the scaled distribution
    params["G"] = lambda func, xd, X, xmin, Xmin, xmax, Xmax, deltaX, c, t: (
        func(xd*(deltaX) + xmin, X, c, t) +
        (xd - 1) * func(xmin, Xmin, c, t) -
        xd * func(xmax, Xmax, c, t)) / deltaX

    params["ode_solver_options"] = {
        'method': 'LSODA',  # LSODA, BDF
        'first_step': 0.001,
        'rtol': 1e-6,
        'atol': 1e-8,
        'max_step': 0.1
    }

    C_A_0 = 5  # concentration of the leaching agent

    # parameters of the initial distribution
    sigma_g = 2.5
    D_g = 1e-5

    meanSize_0 = D_g  # average size

    # time-scale
    tau = meanSize_0 * params["rho_B_s"] * (
        1/params["k_s"]) / \
        2 / params["b"] / C_A_0

    minSize_0 = 4.5e-7
    maxSize_0 = 3e-5

    diffMaxMin_0 = maxSize_0 - minSize_0

    # initial number of particles per unit volume
    m0_0 = 4e10

    f_ = lambda D: (diffMaxMin_0 / (D - minSize_0) / (maxSize_0 - D)) \
        * math.exp(-math.log((D - minSize_0) * diffMaxMin_0 / (maxSize_0 - D) / D_g)**2
        / 2 / math.log(sigma_g)**2) / math.log(sigma_g) / math.sqrt(2*math.pi)

    normalization_factor = quad(f_, minSize_0, maxSize_0)[0]

    f = lambda x: f_(x) / normalization_factor

    ndf_l = lambda x: m0_0 * f(x)

    size_axis_scale = 1e6

    points = np.linspace(minSize_0*1.00001, maxSize_0*0.99999, 1000)

    ndf_l_n = list()
    for x in points:
        ndf_l_n.append(ndf_l(x))

    nMoments = 7

    moments = list()
    for i in range(nMoments):
        g = lambda x: x**i * ndf_l(x)
        moments.append(quad(g, minSize_0, maxSize_0)[0])

    f_scaled = lambda x: diffMaxMin_0 * f(diffMaxMin_0*x + minSize_0)

    moments_scaled = list()
    for i in range(nMoments):
        g = lambda x: x**i * f_scaled(x)
        moments_scaled.append(quad(g, 0, 1)[0])

    nGaussQuadPoints = [10] * 3

    q_n, C_A, min_size, max_size, points, times, m_n = main(
        sys.argv[1:], C_A_0, nMoments, moments_scaled, m0_0, tau, minSize_0,
        maxSize_0, meanSize_0, nGaussQuadPoints, 'beta', params)

    times /= tau

    C_B = params["b"] * (C_A_0 - C_A)

    q_n_final = q_n[-1]
    points_final = points[-1]

    m3_final = simpson(q_n_final * points_final**3, points_final, even='first')

    dissolved_B = math.pi * params["rho_B_s"] * (moments[3] - m3_final) / 6
    consumed_A = C_A_0 - C_A[-1]

    mass_balance_error = abs(dissolved_B - params["b"]*consumed_A)/dissolved_B
    print("Mass balance error: {:.2f}".format(mass_balance_error * 100), "%")

    points = [points_t / D_g for points_t in points]

    q_n_scaled = []
    for q_n_i in q_n:
        q_n_scaled.append([element * D_g / m0_0 for element in q_n_i])

    fig, ax = plt.subplots(1, 1, figsize=[4, 3])
    fig.subplots_adjust(
        top=0.98, bottom=0.16, left=0.14, right=0.96)

    colors = ['C0', 'C3', 'C1', 'C2']
    styles = ['solid', 'dashed', 'dashdot', 'dotted']
    styles = ['solid', (0, (3, 2)), (0, (4, 2, 1.1, 2)), (0, (1, 1))]
    labels = ['$t^*=0$', r'$t^*=0.4$', r'$t^*=1.0$', r'$t^*=2.0$']

    for i, _ in enumerate(q_n_scaled):
        ax.plot(points[i], q_n_scaled[i], linestyle=styles[i], color=colors[i],
                label=' ', zorder=2)

    plt.gca().set_prop_cycle(None)

    A = 1 / math.log(sigma_g) / math.sqrt(2*math.pi)
    B = 1 / 2 / math.log(sigma_g)**2

    markers = ['o', '^', 's', 'd']
    marker_sizes = [5, 5, 4, 5]
    jumps = [15, 15, 24, 60]
    m_a = list()
    for theta, points_i, marker, marker_size, jump, color, label in zip(
        params["t_eval"], points, markers, marker_sizes, jumps, colors, labels):
        phi = lambda D: A * (diffMaxMin_0 / (D + theta - minSize_0 / D_g) / \
            (maxSize_0 / D_g - D - theta) / D_g) * math.exp(-B * math.log(
                (D + theta - minSize_0 / D_g) * diffMaxMin_0 /
                (maxSize_0 / D_g - D - theta) / D_g)**2)
        ax.plot(points_i[::jump], list(map(phi, points_i[::jump])),
                linestyle='None', marker=marker, markersize=marker_size,
                markerfacecolor='None', markeredgewidth=0.5, color=color,
                label=label, zorder=1)

        m_t = np.zeros(nMoments)
        for k in range(nMoments):
            phiByD_to_k = lambda D: phi(D) * (D**k)

            G = -2*params["b"]*C_A_0*params["k_s"] / params["rho_B_s"]
            minSize_t = max((G*theta*tau + minSize_0)/D_g, 0)
            maxSize_t = (G*theta*tau + maxSize_0)/D_g

            m_t[k] = quad(phiByD_to_k, minSize_t, maxSize_t)[0] * m0_0*(D_g**k)

        m_a.append(m_t)

    ax.set_xlabel("$l^*$")
    ax.set_ylabel("$\phi^*$", rotation=0, labelpad=10)
    ax.set_xlim(left=0, right=3)
    ax.set_ylim(bottom=0, top=1)

    ax.legend(
        loc='upper left', bbox_to_anchor=(0.4, 0.98), ncol=2, markerfirst=True,
        frameon=False, alignment='left', columnspacing=1.5, handlelength=3.1,
        handletextpad=1.8, fontsize='x-small', markerscale=0.75,
        title='$\mathrm{EQMOM} \quad \, \mathrm{Analytical}$',
        title_fontsize='x-small')

    fig.savefig('surface-reaction.jpg', dpi=300)

    X_B = [1 - m_t[3] / moments[3] for m_t in m_n]

    f_v_ = lambda D: (diffMaxMin_0 / (D - minSize_0) / (maxSize_0 - D)) \
        * math.exp(-math.log((D - minSize_0) * diffMaxMin_0 / (maxSize_0 - D) / D_g)**2
        / 2 / math.log(sigma_g)**2) / math.log(sigma_g) / math.sqrt(2*math.pi) * (D**3)

    normalization_factor_v = quad(f_v_, minSize_0, maxSize_0)[0]

    f_v = lambda x: f_v_(x) / normalization_factor_v

    fraction_unreacted_D = lambda D, t: \
        (1 + params["dl_dt"](D, D, C_A_0, t) * t / D)**3

    X_B_ref = list()
    for theta in np.linspace(params["t_eval"][0], params["t_eval"][-1]):

        D_t = -params["dl_dt"](0, 0, C_A_0, theta*tau) * theta*tau

        fraction_unreacted = lambda D: fraction_unreacted_D(D, theta*tau) \
            * f_v(D)

        D_min = max(D_t, minSize_0)

        X_B_ref.append(1 - quad(fraction_unreacted, D_min, maxSize_0)[0])

    fig, ax = plt.subplots(1, 1, figsize=[4, 3])
    fig.subplots_adjust(
        top=0.98, bottom=0.16, left=0.14, right=0.96)

    ax.plot(
        params["t_eval"], X_B, 'o', markersize=6, markeredgewidth=1.25,
        markeredgecolor='black', label='$\mathrm{EQMOM}$', color='C3', zorder=2)
    ax.plot(np.linspace(params["t_eval"][0], params["t_eval"][-1]), X_B_ref,
            label='$\mathrm{Reference \; solution}$',
            linestyle='--', color='dimgray', zorder=0)

    ax.set_xlabel("$t^*$")
    ax.set_ylabel("$X_\mathrm{B}$", rotation=0, labelpad=10)
    ax.set_xlim(left=0, right=params["t_eval"][-1])
    ax.set_ylim(bottom=0, top=1)

    ax.legend(
        loc='upper left', bbox_to_anchor=(0.25, 0.35), ncol=1, markerfirst=False,
        frameon=False, alignment='left', columnspacing=1.5, handlelength=3.0,
        handletextpad=1.5, fontsize='small', markerscale=1)

    fig.savefig('surface-reaction-conversion.jpg', dpi=300)

    error = []
    for k in range(nMoments):
        error_k = []
        for i, theta in enumerate(params["t_eval"]):
            error_k.append(abs(m_a[i][k] - m_n[i][k]) / m_a[i][k])
        error.append(error_k)

    fig, ax = plt.subplots(1, 1, figsize=[4, 3])
    fig.subplots_adjust(
        top=0.97, bottom=0.16, left=0.14, right=0.97)

    colors = ['C0', 'C3', 'C1', 'C2', 'C4', 'C6', 'C9']
    styles = ['solid', (0, (3, 2)), (0, (4, 2, 1.1, 2)), (0, (1, 1)),
              'solid', (0, (3, 2)), (0, (1, 1))]
    markers = ['o', '^', 'x', '+', 's', 'v', '*']
    markersize = [4, 4, 5, 7, 4, 4, 5]

    for k in range(min(nMoments, len(colors))):
        ax.plot(params["t_eval"], error[k], linestyle=styles[k], marker=markers[k],
                clip_on=False, color=colors[k], label=str(k), zorder=100)

    ax.set_xlabel("$t^*$")
    ax.set_ylabel("$\mathrm{Relative \; error \; of \; moments}$", rotation=90)
    ax.set_xlim(left=0, right=params["t_eval"][-1])
    # ax.set_ylim(bottom=1e-12, top=1)
    ax.set_ylim(bottom=0, top=1)
    # ax.set_yscale('log')

    ax.legend(
        loc='upper left', bbox_to_anchor=(0.05, 0.95), ncol=1, markerfirst=False,
        frameon=False, alignment='left', columnspacing=1.5, handlelength=3.1,
        handletextpad=1.8, fontsize='x-small', markerscale=0.75,
        title='$k$', title_fontsize='x-small')

    fig.savefig('surface-reaction-moment-error.jpg', dpi=300)

    plt.show()
