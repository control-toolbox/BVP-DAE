import numpy as np
import casadi as cs


def FB(mult, const, eps):
   return mult - const - cs.sqrt(mult ** 2 + const ** 2 + 2. * eps)


class GoddardPrimalDual:
    def __init__(self):
        self.beta = 500.
        self.c = .5
        self.g = 9.81
        self.up = 3.5
        self.um = 0.
        self.eps = .1
        self.tau = 1.
        self.qmax = 10.

    def set_eps(self, eps):
        self.eps = eps

    def initialize(self):
        time = np.linspace(0., 1., 101)
        xp = np.zeros((8, len(time)))
        xp[0, :] = 1.2
        xp[1, :] = .05
        xp[2, :] = 1.
        xp[3, :] = .3
        xp[5, :] = 1.
        z = np.ones((4, len(time)))
        z[0, :] = 1.
        return time, xp, z

    def drag(self, x1, x2):
        return 310. * x2 ** 2 * cs.exp(self.beta * (1. - x1))

    def drag_dx1(self, x1, x2):
        return 310. * x2 ** 2 * (- self.beta * cs.exp(self.beta * (1 - x1)))

    def drag_dx2(self, x1, x2):
        return 620. * x2 * cs.exp(self.beta * (1 - x1))

    def state_constraint(self, x1, x2):
        return 20. * self.drag(x1, x2) - self.qmax

    def state_constraint_dx1(self, x1, x2):
        return 20. * self.drag_dx1(x1, x2)

    def state_constraint_dx2(self, x1, x2):
        return 20. * self.drag_dx2(x1, x2)

    def dyn_press(self, x1, x2):
        return 20. * self.drag(x1, x2)

    def ode(self, time, xp, z):
        x = xp[:4]
        p = xp[4:]
        u = z[0]
        mu = z[1]
        fx0 = x[1] * x[3]
        fx1 = x[3] * ((u - self.drag(x[0], x[1])) / x[2] - 1. / x[0] ** 2)
        fx2 = - x[3] * u / self.c
        fx3 = 0.
        fx4 = - 2. * p[1] * x[3] / x[0] ** 3 + p[1] * x[3] / x[2] * self.drag_dx1(x[0], x[1]) \
                   - mu * self.state_constraint_dx1(x[0], x[1])
        fx5 = x[3] - p[0] * x[3] + p[1] * x[3] / x[2] * self.drag_dx2(x[0], x[1]) \
                   - mu * self.state_constraint_dx2(x[0], x[1])
        fx6 = p[1] * x[3] * (u - self.drag(x[0], x[1])) / x[2] ** 2
        fx7 = x[1] - p[0] * x[1] - p[1] * ((u - self.drag(x[0], x[1])) / x[2] - 1. / x[0] ** 2) + p[2] * u / self.c
        return cs.vertcat(fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7)

    def algeq(self, time, xp, z):
        x = xp[:4]
        p = xp[4:]
        u = z[0]
        mu, nup, num = z[1], z[2], z[3]
        g0 = x[3] * (p[1] / x[2] - p[2] / self.c) + nup - num
        g1 = FB(mu, self.state_constraint(x[0], x[1]), self.eps)
        g2 = FB(nup, u - self.up, self.eps)
        g3 = FB(num, - u, self.eps)
        return cs.vertcat(g0, g1, g2, g3)

    def twobc(self, xp0, xpT, z0, zT):
        x0 = xp0[:4]
        p0 = xp0[4:]
        xT = xpT[:4]
        pT = xpT[4:]
        g = cs.vertcat(x0[0] - 1., x0[1], x0[2] - 1., p0[3], xT[2] - .6, pT[0], pT[1], pT[3])
        return g

    def u(self, z, d=0):
        if d == 0:
            return self.up * z
        return self.up * np.ones_like(z)