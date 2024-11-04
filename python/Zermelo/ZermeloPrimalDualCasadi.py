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
import numpy as np
import casadi as cs


def FB(mult, const, eps):
   return mult - const - cs.sqrt(mult ** 2 + const ** 2 + 2. * eps)


class ZermeloPrimalDualOCP:

    def __init__(self,):
        # parameters
        self.a = 0.
        self.b = 1.
        self.a1 = 2.
        self.a2 = .1
        self.r = 2.

        # initial conditions
        self.xfinal = np.array([20., 1.])
        self.x0 = np.zeros((2,))
        self.phi0 = 0.
        self.theta0 = 0.
        self.gamma0 = -1.
        self.psi0 = 90.

        # constraints
        self.u1max = 1.
        self.u1min = 0.
        self.u0max = np.pi * 2.
        self.u0min = 0.
        self.eps = 1.

    def set_eps(self, eps):
        self.eps = eps

    def initialize(self):
        n = 201
        time = np.linspace(0., 1., n)
        xp = np.zeros((6, len(time)))
        xp[0] = np.linspace(self.x0[0], self.xfinal[0], n)
        xp[1] = np.linspace(self.x0[1], self.xfinal[1], n)
        xp[2] = np.full_like(time, 20.)
        xp[-1] = 1.
        z = np.zeros((7, len(time)))
        z[0] = np.pi / 2.
        z[1] = .5
        return time, xp, z

    def h(self, x1):
        return 3. + .2 * x1 * (1. - x1)

    def partial_x1_h(self, x1):
        return .2 - .4 * x1

    def state_constraint(self, x0, x1):
        return (- (x0 - self.xfinal[0] / 2.) ** 2 / self.a1 ** 2
                - (x1 - self.xfinal[1] / 2.5) ** 2 / self.a2 ** 2 + self.r ** 2)

    def partial_x0_state_constraint(self, x0, x1):
        return - 2. * (x0 - self.xfinal[0] / 2.) / self.a1 ** 2

    def partial_x1_state_constraint(self, x0, x1):
        return - 2. * (x1 - self.xfinal[1] / 2.5) / self.a2 ** 2

    def ode(self, time, xp, z):
        x0, x1, x2, p0, p1, p2 = [xp[i] for i in range(6)]
        u0, u1, lg, lu0p, lu0m, lu1p, lu1m = [z[i] for i in range(7)]
        dxpdt0 = x2 * (u1 * cs.cos(u0) + self.h(x1))
        dxpdt1 = x2 * u1 * cs.sin(u0)
        dxpdt2 = 0.
        dxpdt3 = - lg * self.partial_x0_state_constraint(x0, x1)
        dxpdt4 = - p0 * x2 * self.partial_x1_h(x1) - lg * self.partial_x1_state_constraint(x0, x1)
        dxpdt5 = - p0 * (u1 * cs.cos(u0) + self.h(x1)) - p1 * u1 * cs.sin(u0)
        return cs.vertcat(dxpdt0, dxpdt1, dxpdt2, dxpdt3, dxpdt4, dxpdt5)

    def algeq(self, time, xp, z):
        x0, x1, x2, p0, p1, p2 = [xp[i] for i in range(6)]
        u0, u1, lg, lu0p, lu0m, lu1p, lu1m = [z[i] for i in range(7)]

        dhdu0 = - p0 * x2 * u1 * cs.sin(u0) + p1 * x2 * u1 * cs.cos(u0) + lu0p - lu0m
        dhdu1 = p0 * x2 * cs.cos(u0) + p1 * x2 * cs.sin(u0) + lu1p - lu1m
        dhdu2 = FB(lg, self.state_constraint(x0, x1), self.eps)
        dhdu3 = FB(lu0p, u0 - self.u0max, self.eps)
        dhdu4 = FB(lu0m, self.u0min - u0, self.eps)
        dhdu5 = FB(lu1p, u1 - self.u1max, self.eps)
        dhdu6 = FB(lu1m, self.u1min - u1, self.eps)
        return cs.vertcat(dhdu0, dhdu1, dhdu2, dhdu3, dhdu4, dhdu5, dhdu6)

    def twobc(self, xp0, xpT, z0, zT):
        return cs.vertcat(xp0[:2] - self.x0, xpT[:2] - self.xfinal, xp0[-1], xpT[-1] - 1.)