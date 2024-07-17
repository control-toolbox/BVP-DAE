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
from utils_functions import log_pen


class VanDerPolPrimalOCP:

    def __init__(self):
        self.x0 = np.ones((2,))
        self.r = .2
        self.umax = 1.
        self.umin = - 1.
        self.eps = 1.

    def set_eps(self, x):
        self.eps = x

    def state_const(self, x1, d=0):
        if d == 0:
            return -.4 - x1
        if d == 1:
            return - np.ones_like(x1)
        return np.zeros_like(x1)

    def initialize(self):
        time = np.linspace(0., 4., 41)
        xp = np.zeros((4, len(time)))
        xp[0] = self.x0[0]
        xp[1] = self.x0[1]
        z = np.zeros((1, len(time)))
        return time, xp, z

    def ode(self, time, xp, z):
        x0, x1, p0, p1 = xp
        u = z[0]
        lg = self.eps * log_pen(self.state_const(x1), d=1)
        dxpdt = np.zeros_like(xp)
        dxpdt[0] = x1
        dxpdt[1] = - x0 + x1 * (1. - x0 ** 2) + u
        dxpdt[2] = - 2. * x0 + p1 + 2. * p1 * x1 * x0
        dxpdt[3] = - 2. * x1 - p0 + p1 * (x0 ** 2 - 1.) - lg * self.state_const(x1, d=1)
        return dxpdt

    def odejac(self, time, xp, z):
        x0, x1, p0, p1 = xp
        gx = self.state_const(x1)
        lg = self.eps * log_pen(gx, d=1)
        dlg = self.eps * log_pen(gx, d=2)
        dgx_dx1 = self.state_const(x1, d=1)
        d2gx_gx1 = self.state_const(x1, d=2)
        fx = np.zeros((xp.shape[0], xp.shape[0], len(time)))
        fx[0, 1] = 1.
        fx[1, 0] = - 1. - 2. * x1 * x0
        fx[1, 1] = 1. - x0 ** 2
        fx[2, 0] = - 2. + 2. * p1 * x1
        fx[2, 1] = 2. * p1 * x0
        fx[2, 3] = 1. + 2. * x1 * x0
        fx[3, 0] = 2. * p1 * x0
        fx[3, 1] = - 2. - (dlg * dgx_dx1 ** 2 + lg * d2gx_gx1)
        fx[3, 2] = - 1.
        fx[3, 3] = x0 ** 2 - 1.

        fz = np.zeros((xp.shape[0], z.shape[0], len(time)))
        fz[1, 0] = 1.

        return fx, fz

    def algeq(self, time, xp, z):
        x0, x1, p0, p1 = xp
        u = z[0]
        lup = self.eps * log_pen(u - self.umax, d=1)
        lum = self.eps * log_pen(self.umin - u, d=1)
        alg = np.zeros_like(z)
        alg[0] = 2. * u + p1 + lup - lum
        return alg

    def algjac(self, time, xp, z):
        u = z[0]
        dlup = self.eps * log_pen(u - self.umax, d=2)
        dlum = - self.eps * log_pen(self.umin - u, d=2)

        gx = np.zeros((z.shape[0], xp.shape[0], len(time)))
        gx[0, 3] = 1.
        gz = np.zeros((z.shape[0], z.shape[0], len(time)))
        gz[0, 0] = 2. + dlup - dlum

        return gx, gz

    def twobc(self, xp0, xpT, z0, zT):
        bc = np.zeros_like(xp0)
        bc[:2] = xp0[:2] - self.x0
        bc[2] = xpT[0] ** 2 + xpT[1] ** 2 - self.r ** 2
        bc[3] = xpT[2] * xpT[1] - xpT[3] * xpT[0]
        return bc

    def bcjac(self, xp0, xpT, z0, zT):
        bcxp0, bcxpT, bcz0, bczT = (np.zeros((xp0.size, xp0.size)), np.zeros((xp0.size, xp0.size)),
                                    np.zeros((xp0.size, z0.size)), np.zeros((xp0.size, zT.size)))

        bcxp0[:2, :2] = np.eye(2)
        bcxpT[2, 0] = 2. * xpT[0]
        bcxpT[2, 1] = 2. * xpT[1]
        bcxpT[3, 0] = - xpT[3]
        bcxpT[3, 1] = xpT[2]
        bcxpT[3, 2] = xpT[1]
        bcxpT[3, 3] = - xpT[0]
        return bcxp0, bcxpT, bcz0, bczT