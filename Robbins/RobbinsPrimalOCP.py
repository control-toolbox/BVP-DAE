"""
Code developped by Paul Malisani
IFP Energies nouvelles
Applied mathematics department
paul.malisani@ifpen.fr


Mathematical details on the methods can be found in

Interior Point Methods in Optimal Control Problems of Affine Systems: Convergence Results and Solving Algorithms
SIAM Journal on Control and Optimization, 61(6), 2023
https://doi.org/10.1137/23M1561233

Please cite this paper if you are using these methods.
"""
import numpy as np
from utils_functions import log_pen


class RobbinsPrimal:
    def __init__(self):
        self.eps = 1.
        self.up = 1.
        self.um = -1.

    def set_eps(self, eps):
        self.eps = eps

    def ode(self, time, xp, z):
        n = xp.shape[0] // 2
        x, p = xp[:n], xp[n:]
        u = z[0, :]
        dxpdt = np.zeros_like(xp)
        dxpdt[0] = x[1]
        dxpdt[1] = x[2]
        dxpdt[2] = u
        dxpdt[3] = - 1. + self.eps * log_pen(-x[0], d=1)
        dxpdt[4] = - p[0]
        dxpdt[5] = - p[1]
        return dxpdt

    def odejac(self, time, xp, z):
        n = xp.shape[0] // 2
        m = z.shape[0]
        x, p = xp[:n], xp[n:]
        u = z[0, :]

        fx = np.zeros((2 * n, 2 * n, time.size))
        fx[0, 1] = 1.
        fx[1, 2] = 1.
        fx[3, 0] = - self.eps * log_pen(- x[0], d=2)
        fx[4, 3] = -1.
        fx[5, 4] = -1.

        fz = np.zeros((2 * n, m, time.size))
        fz[2, 0] = 1.
        return fx, fz

    def algeq(self, time, xp, z):
        n = xp.shape[0] // 2
        m = z.shape[0]
        x, p = xp[:n], xp[n:]
        u = z[0, :]
        g = np.zeros_like(z)
        g[0] = p[2] + self.eps * (log_pen(u - self.up, d=1) - log_pen(self.um - u, d=1))
        return g

    def algjac(self, time, xp, z):
        n = xp.shape[0] // 2
        m = z.shape[0]
        x, p = xp[:n], xp[n:]
        u = z[0, :]

        gx = np.zeros((m, 2 * n, time.size))
        gx[0, -1] = 1.

        gz = np.zeros((m, m, time.size))
        gz[0, 0] = self.eps * (log_pen(u - self.up, d=2) + log_pen(self.um - u, d=2))
        return gx, gz

    def twobc(self, x0, xT, z0, zT):
        return np.array([x0[0] -1., x0[1], x0[2], xT[3], xT[4], xT[5]])

    def bcjac(self, x0, xT, z0, zT):
        bcx0, bcxt = np.zeros((x0.size, x0.size)), np.zeros((x0.size, x0.size))
        bcz0, bczt = np.zeros((x0.size, z0.size)), np.zeros((x0.size, z0.size))
        bcx0[0, 0] = 1.
        bcx0[1, 1] = 1.
        bcx0[2, 2] = 1.
        bcxt[3, 3] = 1.
        bcxt[4, 4] = 1.
        bcxt[5, 5] = 1.
        return bcx0, bcxt, bcz0, bczt