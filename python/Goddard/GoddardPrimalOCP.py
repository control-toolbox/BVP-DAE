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


class GoddardPrimal:
    def __init__(self):
        self.beta = 500.
        self.c = .5
        self.big_const = 620.
        self.g = 9.81
        self.up = 3.5
        self.um = 0.
        self.eps = 1.
        self.qmax = 10.
        self.tau = 1.

    def set_eps(self, eps):
        self.eps = eps

    def initialize(self):
        time = np.linspace(0., 1., 101)
        xp = np.zeros((8, len(time)))
        xp[0] = 1.2
        xp[1] = .05
        xp[2] = 1.
        xp[3] = .3
        xp[5] = 1.
        z = np.ones((1, len(time)))
        return time, xp, z

    def drag(self, x1, x2, dx1=0, dx2=0):
        if dx1 == 0 and dx2 == 0:
            return 310. * x2 ** 2 * np.exp(self.beta * (1. - x1))
        if dx1 == 1 and dx2 == 0:
            return 310. * x2 ** 2 * (- self.beta * np.exp(self.beta * (1 - x1)))
        if dx1 == 0 and dx2 == 1:
            return 620. * x2 * np.exp(self.beta * (1 - x1))
        if dx1 == 1 and dx2 == 1:
            return 620. * x2 * (- self.beta * np.exp(self.beta * (1 - x1)))
        if dx1 == 2 and dx2 == 0:
            return 310. * x2 ** 2 * self.beta ** 2 * np.exp(self.beta * (1 - x1))
        if dx1 == 0 and dx2 == 2:
            return 620. * np.exp(self.beta * (1 - x1))
        raise Exception("cumulative derivative order must <= 2")

    def dyn_press(self, x1, x2, dx1=0, dx2=0):
        return 20. * self.drag(x1, x2, dx1=dx1, dx2=dx2)

    def state_constraint(self, x1, x2, dx1=0, dx2=0):
        if dx1 == 0 and dx2 == 0:
            return self.dyn_press(x1, x2, dx1, dx2) - self.qmax
        else:
            return self.dyn_press(x1, x2, dx1, dx2)

    def ode(self, time, xp, z):
        x = xp[:4]
        p = xp[4:]
        u = z[0]
        fx = np.zeros(xp.shape)
        g = self.state_constraint(x[0], x[1])
        dgx1 = self.state_constraint(x[0], x[1], dx1=1)
        dgx2 = self.state_constraint(x[0], x[1], dx2=1)

        dlog = log_pen(g, d=1)
        fx[0] = x[1] * x[3]
        fx[1] = x[3] * ((u - self.drag(x[0], x[1])) / x[2] - 1. / x[0] ** 2)
        fx[2] = - x[3] * u / self.c
        fx[4] = - 2. * p[1] * x[3] / x[0] ** 3 + p[1] * x[3] / x[2] * self.drag(x[0], x[1], dx1=1, dx2=0) \
                - self.eps * dlog * dgx1
        fx[5] = x[3] - p[0] * x[3] + p[1] * x[3] / x[2] * self.drag(x[0], x[1], dx1=0, dx2=1) - self.eps * dlog * dgx2
        fx[6] = p[1] * x[3] * (u - self.drag(x[0], x[1])) / x[2] ** 2
        fx[7] = x[1] - p[0] * x[1] - p[1] * ((u - self.drag(x[0], x[1])) / x[2] - 1. / x[0] ** 2) + p[2] * u / self.c
        return fx

    def odejac(self, time, xp, z):
        x = xp[:4]
        p = xp[4:]
        u = z[0]
        drag = self.drag(x[0], x[1])
        drag_x1 = self.drag(x[0], x[1], dx1=1)
        drag_x2 = self.drag(x[0], x[1], dx2=1)
        drag_x1x1 = self.drag(x[0], x[1], dx1=2)
        drag_x1x2 = self.drag(x[0], x[1], dx1=1, dx2=1)
        drag_x2x2 = self.drag(x[0], x[1], dx2=2)

        g = self.state_constraint(x[0], x[1])
        dgx1 = self.state_constraint(x[0], x[1], dx1=1)
        dgx2 = self.state_constraint(x[0], x[1], dx2=1)
        d2gx1x1 = self.state_constraint(x[0], x[1], dx1=2)
        d2gx2x2 = self.state_constraint(x[0], x[1], dx2=2)
        d2gx1x2 = self.state_constraint(x[0], x[1], dx1=1, dx2=1)

        dlog = log_pen(g, d=1)
        d2log = log_pen(g, d=2)

        jx = np.zeros((xp.shape[0], xp.shape[0], xp.shape[1]))
        jz = np.zeros((xp.shape[0], z.shape[0], xp.shape[1]))

        jx[0, 1] = x[3]
        jx[0, 3] = x[1]

        jx[1, 0] = 2. * x[3] / x[0] ** 3 - x[3] / x[2] * drag_x1
        jx[1, 1] = - drag_x2 * x[3] / x[2]
        jx[1, 2] = - x[3] / x[2] ** 2 * (u - drag)
        jx[1, 3] = (u - drag) / x[2] - 1. / x[0] ** 2

        jx[2, 3] = - u / self.c

        jx[4, 0] = 6. * p[1] * x[3] / x[0] ** 4 + p[1] * x[3] / x[2] * drag_x1x1 - self.eps * (d2log * dgx1 ** 2
                                                                                               + dlog * d2gx1x1)
        jx[4, 1] = p[1] * x[3] / x[2] * drag_x1x2 - self.eps * (d2log * dgx1 * dgx2 + dlog * d2gx1x2)
        jx[4, 2] = - p[1] * x[3] * drag_x1 / x[2] ** 2
        jx[4, 3] = - 2. * p[1] / x[0] ** 3 + p[1] * drag_x1 / x[2]
        jx[4, 5] = - 2 * x[3] / x[0] ** 3 + x[3] / x[2] * drag_x1

        jx[5, 0] = p[1] * x[3] / x[2] * drag_x1x2 - self.eps * (d2log * dgx1 * dgx2 + dlog * d2gx1x2)
        jx[5, 1] = p[1] * x[3] / x[2] * drag_x2x2 - self.eps * (d2log * dgx2 ** 2 + dlog * d2gx2x2)
        jx[5, 2] = - p[1] * x[3] / x[2] ** 2 * drag_x2
        jx[5, 3] = 1. - p[0] + p[1] / x[2] * drag_x2
        jx[5, 4] = - x[3]
        jx[5, 5] = x[3] / x[2] * drag_x2

        jx[6, 0] = - p[1] * x[3] / x[2] ** 2 * drag_x1
        jx[6, 1] = - p[1] * x[3] / x[2] ** 2 * drag_x2
        jx[6, 2] = - 2. * p[1] * x[3] / x[2] ** 3 * (u - drag)
        jx[6, 3] = p[1] / x[2] ** 2 * (u - drag)
        jx[6, 5] = x[3] / x[2] ** 2 * (u - drag)

        jx[7, 0] = - p[1] * (- drag_x1 / x[2] + 2. / x[0] ** 3)
        jx[7, 1] = 1. - p[0] + p[1] / x[2] * drag_x2
        jx[7, 2] = - p[1] * drag / x[2] ** 2
        jx[7, 4] = - x[1]
        jx[7, 5] = - ((u - drag) / x[2] - 1. / x[0] ** 2)
        jx[7, 6] = u / self.c

        # algebraic
        jz[1, 0] = x[3] / x[2]
        jz[2, 0] = - x[3] / self.c
        jz[6, 0] = p[1] * x[3] / x[2] ** 2
        jz[7, 0] = (- p[1] / x[2] + p[2] / self.c)
        return jx, jz

    def algeq(self, time, xp, z):
        x = xp[:4]
        p = xp[4:]
        u = z[0]
        g = np.zeros_like(z)
        g[0] = x[3] * (p[1] / x[2] - p[2] / self.c)+ self.eps * (log_pen(u - self.up, d=1) - log_pen(self.um - u, d=1))
        return g

    def algjac(self, time, xp, z):
        x = xp[:4]
        p = xp[4:]
        u = z[0]
        gx = np.zeros((z.shape[0], xp.shape[0], xp.shape[1]))
        gx[0, 2] = - x[3] * p[1] / x[2] ** 2
        gx[0, 3] = (p[1] / x[2] - p[2] / self.c)
        gx[0, 5] = x[3] / x[2]
        gx[0, 6] = - x[3] / self.c

        gz = np.zeros((z.shape[0], z.shape[0], z.shape[1]))
        gz[0, 0] = self.eps * (log_pen(u - self.up, d=2) + log_pen(self.um - u, d=2))
        return gx, gz

    def twobc(self, xp0, xpT, z0, zT):
        x0 = xp0[:4]
        p0 = xp0[4:]
        xT = xpT[:4]
        pT = xpT[4:]
        g = np.array([x0[0] - 1., x0[1], x0[2] - 1., p0[3], xT[2] - .6, pT[0], pT[1], pT[3]])
        return g

    def bcjac(self, xp0, xpT, z0, zT):

        gx0 = np.zeros((xp0.size, xp0.size))
        gx0[0, 0] = 1.
        gx0[1, 1] = 1.
        gx0[2, 2] = 1.
        gx0[3, 7] = 1.

        gxT = np.zeros((xp0.size, xp0.size))
        gxT[4, 2] = 1.
        gxT[5, 4] = 1.
        gxT[6, 5] = 1.
        gxT[7, 7] = 1.

        gz0 = np.zeros((xp0.size, z0.size))
        gzT = gz0.copy()

        return gx0, gxT, gz0, gzT
