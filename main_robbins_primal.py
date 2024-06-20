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
from Robbins.RobbinsPrimalOCP import RobbinsPrimal, log_pen
import numpy as np
from bvpdae_solver import BVPDAE, BVPSol
from timeit import default_timer as dtime
from utils_functions import plot_sequence
import matplotlib.pyplot as plt
import pickle


def main_primal():
    cumul_time = 0.
    eps = .1
    alpha = .5
    ocp = RobbinsPrimal()
    ocp.set_eps(eps)
    time = np.linspace(0., 6., 61)
    xp = np.zeros((6, len(time)))
    xp[0] = 1.
    z = np.ones((1, len(time))) * .5
    initial_solution = dict(time=time, xp=xp, z=z)
    options = dict(control_odes_error=True, display=1)
    bvp_solver = BVPDAE(**options)
    convergence = False
    iter = 0
    times, x1s, x2s, x3s, us, p1s, p2s, p3s, mus, nups, nums = [list() for _ in range(11)]
    bvp_sol = BVPSol(time=time, xp=xp, z=z)
    while not convergence:
        print("eps = ", eps)
        ts0 = dtime()
        bvp_sol, infos = bvp_solver.solve(bvp_sol, ocp)
        ts1 = dtime()
        cumul_time += ts1 - ts0
        if not infos.success:
            raise Exception("Primal Robbins problem failed before convergence")
        time, xp, z = bvp_sol.time, bvp_sol.xp, bvp_sol.z
        if np.mod(iter, 1) == 0:
            times.append(time)
            x1s.append(xp[0])
            x2s.append(xp[1])
            x3s.append(xp[2])
            us.append(z[0])
            p1s.append(xp[3])
            p2s.append(xp[4])
            p3s.append(xp[5])
            mus.append(ocp.eps * log_pen(-xp[0], d=1))
            nups.append(ocp.eps * log_pen(z[0] - ocp.up, d=1))
            nums.append(ocp.eps * log_pen(ocp.um - z[0], d=1))
        convergence = eps < 1e-7
        eps *= alpha
        ocp.set_eps(eps)
        iter += 1
        print(" ")
    print("Execution time = ", cumul_time)
    optimal_solution = dict(time=time, xp=xp, z=z, eps=eps / alpha, alpha=alpha, exec_time=cumul_time, iter=iter)
    dict_save = dict(initial_solution=initial_solution, optimal_solution=optimal_solution)
    with open("Robbins/results_robbins_primal.pickle", "wb") as fh:
        pickle.dump(dict_save, fh)
    nplots = 7
    plot_sequence(times, p3s, "$p_3(t)$", "Sequence of optimal penalized adjoint state $\\bar{p}_{3,\epsilon}$", nplots=nplots)
    plot_sequence(times, p2s, "$p_2(t)$", "Sequence of optimal penalized adjoint state $\\bar{p}_{2, \epsilon}$", nplots=nplots)
    plot_sequence(times, p1s, "$p_1(t)$", "Sequence of optimal penalized adjoint state $\\bar{p}_{1, \epsilon}$", nplots=nplots)
    plot_sequence(times, us, "$u(t)$", "Sequence of optimal penalized control $\\bar{u}_{\epsilon}$", nplots=nplots)
    plot_sequence(times, x3s, "$x_3(t)$", "Sequence of optimal penalized state $\\bar{x}_{3, \epsilon}$", nplots=nplots)
    plot_sequence(times, x2s, "$x_2(t)$", "Sequence of optimal penalized state $\\bar{x}_{2, \epsilon}$", nplots=nplots)
    plot_sequence(times, x1s, "$x_1(t)$", "Sequence of optimal penalized state $\\bar{x}_{1, \epsilon}$", nplots=nplots)
    plot_sequence(times, mus, "$\mu(t)$", "Sequence of optimal penalized state-constraint multiplier $\\bar{\lambda}_\epsilon^g$", nplots=nplots)
    plot_sequence(times, nups, "$\eta_1(t)$", "Sequence of optimal penalized first mixed-constraint multiplier $\\bar{\lambda}_{1, \epsilon}^c$", nplots=nplots)
    plot_sequence(times, nums, "$\eta_2(t)$", "Sequence of optimal penalized second mixed-constraint multiplier $\\bar{\lambda}_{2, \epsilon}^c$", nplots=nplots)
    plt.show()


if __name__ == "__main__":
    main_primal()