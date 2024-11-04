"""
Code developped by Paul Malisani
IFP Energies nouvelles
Applied mathematics department
paul.malisani@ifpen.fr


Mathematical details on the methods can be found in

Interior Point Methods in Optimal Control Problems of Affine Systems: Convergence Results and Solving Algorithms
SIAM Journal on Control and Optimization, 61(6), 2023
https://doi.org/10.1137/23M1561233

and

Interior Point methods in Optimal Control, in review,
http://dx.doi.org/10.13140/RG.2.2.20384.76808

Please cite these papers if you are using these methods.
"""
from Zermelo.ZermeloPrimalDualCasadi import ZermeloPrimalDualOCP
from bvpdae_solver import BVPDAE, BVPSol
import numpy as np
from timeit import default_timer as dtime
from utils_functions import plot_sequence
import matplotlib.pyplot as plt
import pickle


def main_primal_dual():
    cumul_time = 0.
    eps = .1
    alpha = .5
    tol = 1e-7
    ocp = ZermeloPrimalDualOCP()
    ocp.set_eps(eps)
    time, xp, z = ocp.initialize()
    initial_solution = dict(time=time, xp=xp, z=z)
    bvp_sol = BVPSol(time=time, xp=xp, z=z)
    options = dict(control_odes_error=True, display=1, res_tol=1e-3)
    bvp_solver = BVPDAE(**options)
    convergence = False
    iter = 0
    times, x0s, x1s, p0s, p1s, u0s, u1s = [list() for _ in range(7)]
    while not convergence:
        print("eps = ", eps)
        print("alpha = ", alpha)
        ts0 = dtime()
        bvp_sol, infos = bvp_solver.solve(bvp_sol, ocp)
        ts1 = dtime()
        cumul_time += ts1 - ts0
        if not infos.success:
            raise Exception("Primal-dual Zermelo problem failed before convergence")
        time, xp, z = bvp_sol.time, bvp_sol.xp, bvp_sol.z
        convergence = eps <= tol
        eps *= alpha
        ocp.set_eps(eps)
        times.append(xp[2, 0] * time)
        x0s.append(xp[0, :])
        x1s.append(xp[1, :])
        p0s.append(xp[3, :])
        p1s.append(xp[4, :])
        u0s.append(z[0])
        u1s.append(z[1])
        iter += 1
        print(" ")
    optimal_solution = dict(time=time, xp=xp, z=z, eps=eps / alpha, alpha=alpha, exec_time=cumul_time, iter=iter)
    dict_save = dict(initial_solution=initial_solution, optimal_solution=optimal_solution)
    with open("Zermelo/results_zermelo_primal_dual.pickle", "wb") as fh:
        pickle.dump(dict_save, fh)
    plot_sequence(times, x0s, "$x_0(t)$",
                  "Sequence of optimal penalized $\\bar{x}_0$")
    plot_sequence(times, x1s, "$x_1(t)$",
                  "Sequence of optimal penalized $\\bar{x}_1$")
    plot_sequence(times, p0s, "$p_0(t)$",
                  "Sequence of optimal penalized $\\bar{p}_0$")
    plot_sequence(times, p1s, "$x_1(t)$",
                  "Sequence of optimal penalized $\\bar{p}_1$")
    plot_sequence(times, u0s, "$u_0(t)$",
                  "Sequence of optimal penalized $\\bar{u}_0$")
    plot_sequence(times, u1s, "$u_1(t)$",
                  "Sequence of optimal penalized $\\bar{u}_1$")

    t = np.linspace(0., 2. * np.pi, 1000)
    x1t = ocp.r * ocp.a1 * np.cos(t) + ocp.xfinal[0] / 2.
    x2t = ocp.r * ocp.a2 * np.sin(t) + ocp.xfinal[1] / 2.5
    plt.figure()
    plt.plot(xp[0], xp[1])
    plt.plot(x1t, x2t)

    plt.figure()
    plt.plot(time, ocp.state_constraint(xp[0], xp[1]))

    print("Time final = ", xp[2, 0])
    print("Execution time = ", cumul_time)
    print("Number of iterations = ", len(times))
    print("Number of final time steps = ", len(time))
    plt.show()


if __name__ == "__main__":
    main_primal_dual()