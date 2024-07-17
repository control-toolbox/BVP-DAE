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
from utils_functions import FB


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
        n = 501
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


    def h(self, x1, d=0):
        if d == 0:
            return 3. + .2 * x1 * (1. - x1)
        if d == 1:
            return .2 - .4 * x1
        if d == 2:
            return np.full_like(x1, - 4.)
        return np.zeros_like(x1)

    def state_constraint(self, x0, x1, dx0=0, dx1=0):
        if dx0 == 0 and dx1 == 0:
            return - (x0 - self.xfinal[0] / 2.) ** 2 / self.a1 ** 2 - (
                        x1 - self.xfinal[1] / 2.5) ** 2 / self.a2 ** 2 + self.r ** 2
        if dx0 == 1 and dx1 == 0:
            return - 2. * (x0 - self.xfinal[0] / 2.) / self.a1 ** 2
        if dx0 == 0 and dx1 == 1:
            return - 2. * (x1 - self.xfinal[1] / 2.5) / self.a2 ** 2
        if dx0 == 2 and dx1 == 0:
            return - 2. / self.a1 ** 2
        if dx0 == 0 and dx1 == 2:
            return - 2. / self.a2 ** 2
        return np.zeros_like(x1)

    def ode(self, time, xp, z):
        x0, x1, x2, p0, p1, p2 = xp
        u0, u1, lg, lu0p, lu0m, lu1p, lu1m = z
        dxpdt = np.zeros_like(xp)
        dxpdt[0] = x2 * (u1 * np.cos(u0) + self.h(x1))
        dxpdt[1] = x2 * u1 * np.sin(u0)
        dxpdt[2] = 0.
        dxpdt[3] = - lg * self.state_constraint(x0, x1, dx0=1)
        dxpdt[4] = - p0 * x2 * self.h(x1, d=1) - lg * self.state_constraint(x0, x1, dx1=1)
        dxpdt[5] = - p0 * (u1 * np.cos(u0) + self.h(x1)) - p1 * u1 * np.sin(u0)
        return dxpdt

    def odejac(self, time, xp, z):
        x0, x1, x2, p0, p1, p2 = xp
        u0, u1, lg, lu0p, lu0m, lu1p, lu1m = z
        jacx = np.zeros((xp.shape[0], xp.shape[0], len(time)))
        jacx[0, 1] = x2 * self.h(x1, d=1)
        jacx[0, 2] = u1 * np.cos(u0) + self.h(x1)
        jacx[1, 2] = u1 * np.sin(u0)
        jacx[3, 0] = - lg * self.state_constraint(x0, x1, dx0=2)
        jacx[3, 1] = - lg * self.state_constraint(x0, x1, dx0=1, dx1=1)
        jacx[4, 0] = - lg * self.state_constraint(x0, x1, dx0=1, dx1=1)
        jacx[4, 1] = - p0 * x2 * self.h(x1, d=2) - lg * self.state_constraint(x0, x1, dx1=2)
        jacx[4, 2] = - p0 * self.h(x1, d=1)
        jacx[4, 3] = - x2 * self.h(x1, d=1)
        jacx[5, 1] = - p0 * self.h(x1, d=1)
        jacx[5, 3] = - (u1 * np.cos(u0) + self.h(x1))
        jacx[5, 4] = - u1 * np.sin(u0)

        jacz = np.zeros((xp.shape[0], z.shape[0], len(time)))
        jacz[0, 0] = - x2 * u1 * np.sin(u0)
        jacz[0, 1] = x2 * np.cos(u0)
        jacz[1, 0] = x2 * u1 * np.cos(u0)
        jacz[1, 1] = x2 * np.sin(u0)
        jacz[3, 2] = - self.state_constraint(x0, x1, dx0=1)
        jacz[4, 2] = - self.state_constraint(x0, x1, dx1=1)
        jacz[5, 0] = p0 * u1 * np.sin(u0) - p1 * u1 * np.cos(u0)
        jacz[5, 1] = - p0 * np.cos(u0) - p1 * np.sin(u0)
        return jacx, jacz

    def algeq(self, time, xp, z):
        x0, x1, x2, p0, p1, p2 = xp
        u0, u1, lg, lu0p, lu0m, lu1p, lu1m = z

        dhdu = np.zeros_like(z)
        dhdu[0] = - p0 * x2 * u1 * np.sin(u0) + p1 * x2 * u1 * np.cos(u0) + lu0p - lu0m
        dhdu[1] = p0 * x2 * np.cos(u0) + p1 * x2 * np.sin(u0) + lu1p - lu1m
        dhdu[2] = FB(lg, self.state_constraint(x0, x1), self.eps)
        dhdu[3] = FB(lu0p, u0 - self.u0max, self.eps)
        dhdu[4] = FB(lu0m, self.u0min - u0, self.eps)
        dhdu[5] = FB(lu1p, u1 - self.u1max, self.eps)
        dhdu[6] = FB(lu1m, self.u1min - u1, self.eps)
        return dhdu

    def algjac(self, time, xp, z):
        x0, x1, x2, p0, p1, p2 = xp
        u0, u1, lg, lu0p, lu0m, lu1p, lu1m = z

        gx = np.zeros((z.shape[0], xp.shape[0], len(time)))
        gx[0, 2] = - p0 * u1 * np.sin(u0) + p1 * u1 * np.cos(u0)
        gx[0, 3] = - x2 * u1 * np.sin(u0)
        gx[0, 4] = x2 * u1 * np.cos(u0)
        gx[1, 2] = p0 * np.cos(u0) + p1 * np.sin(u0)
        gx[1, 3] = x2 * np.cos(u0)
        gx[1, 4] = x2 * np.sin(u0)
        gx[2, 0] = FB(lg, self.state_constraint(x0, x1), self.eps, dy=1) * self.state_constraint(x0, x1, dx0=1)
        gx[2, 1] = FB(lg, self.state_constraint(x0, x1), self.eps, dy=1) * self.state_constraint(x0, x1, dx1=1)

        gz = np.zeros((z.shape[0], z.shape[0], len(time)))
        gz[0, 0] = - p0 * x2 * u1 * np.cos(u0) - p1 * x2 * u1 * np.sin(u0)
        gz[0, 1] = - p0 * x2 * np.sin(u0) + p1 * x2 * np.cos(u0)
        gz[0, 3] = 1.
        gz[0, 4] = - 1.
        gz[1, 0] = - p0 * x2 * np.sin(u0) + p1 * x2 * np.cos(u0)
        gz[1, 5] = 1.
        gz[1, 6] = - 1.
        gz[2, 2] = FB(lg, self.state_constraint(x0, x1), self.eps, dx=1)
        gz[3, 0] = FB(lu0p, u0 - self.u0max, self.eps, dy=1)
        gz[3, 3] = FB(lu0p, u0 - self.u0max, self.eps, dx=1)
        gz[4, 0] = - FB(lu0m, self.u0min - u0, self.eps, dy=1)
        gz[4, 4] = FB(lu0m, self.u0min - u0, self.eps, dx=1)
        gz[5, 1] = FB(lu1p, u1 - self.u1max, self.eps, dy=1)
        gz[5, 5] = FB(lu1p, u1 - self.u1max, self.eps, dx=1)
        gz[6, 1] = - FB(lu1m, self.u1min - u1, self.eps, dy=1)
        gz[6, 6] = FB(lu1m, self.u1min - u1, self.eps, dx=1)
        return gx, gz

    def twobc(self, xp0, xpT, z0, zT):
        bc = np.zeros_like(xp0)
        bc[:2] = xp0[:2] - self.x0
        bc[2:4] = xpT[:2] - self.xfinal
        bc[4] = xp0[-1]
        bc[5] = xpT[-1] - 1.
        return bc

    def bcjac(self, xp0, xpT, z0, zT):
        gx0 = np.zeros((xp0.size, xp0.size))
        gx0[:2, :2] = np.eye(2)
        gx0[4, -1] = 1.

        gxT = np.zeros((xpT.size, xpT.size))
        gxT[2:4, :2] = np.eye(2)
        gxT[5, -1] = 1.

        gz0 = np.zeros((xp0.size, z0.size))
        gzT = gz0.copy()

        return gx0, gxT, gz0, gzT