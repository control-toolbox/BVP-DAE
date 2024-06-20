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

import matplotlib.pyplot as plt
import numpy as np


def repmat(a, rep_dim):
    """
    This function allows to replicated a 2D-matrix A along first, second and optionally third dimension.
    :param a: Matrix to be replicated
    :param rep_dim: tuple of integer (d0, d1, [d2]) giving the number of times matrix a is replicated along each dimension
    :return: numpy array of replicated a matrix
    """
    if len(rep_dim) < 2:
        raise Exception("Repmat needs at least 2 dimensions")
    if len(rep_dim) == 2:
        return np.tile(a, rep_dim)
    if len(rep_dim) == 3:
        d0, d1, d2 = rep_dim
        ad0, ad1 = a.shape
        return np.reshape(np.tile(np.tile(a, (d0, d1)), (1, d2)), (ad0*d0, ad1*d1, d2), order="F")


def matmul3d(a, b):
    """
    3D multiplication of matrices a, b
    """
    if len(a.shape) == 2 and len(b.shape) == 3:
        return np.einsum('ij,jlk->ilk', a, b)
    elif len(a.shape) == 3 and len(b.shape) == 3:
        return np.einsum('ijk,jlk->ilk', a, b)
    elif len(a.shape) == 3 and len(b.shape) == 2:
        return np.einsum('ijk,jl->ilk', a, b)
    else:
        raise Exception("not a 3D matrix product")


def FB(x, y, eps, dx=0, dy=0):
    if dx == 0 and dy == 0:
        return x - y - np.sqrt(x ** 2 + y ** 2 + 2. * eps)
    if dx == 1 and dy == 0:
        return 1. - x / np.sqrt(x ** 2 + y ** 2 + 2. * eps)
    if dx == 0 and dy == 1:
        return -1. - y / np.sqrt(x ** 2 + y ** 2 + 2. * eps)
    raise Exception("Only first order derivatives are defined for Fisher-Burmeister complementarity function")


def log_pen(x, d=0):
    if d == 0:
        return - np.log(-x)
    if d == 1:
        return - 1. / x
    if d == 2:
        return 1. / x ** 2
    raise Exception("logarithmic penalty is only defined for d in {0, 1, 2}")


def plot_sequence(times, seq, name, title, nplots=1):
    nsteps = len(seq) // nplots
    plt.figure()
    for i, y in enumerate(seq):
        if i == len(seq) - 1:
            color = "black"
            width = 3
            style = "solid"
            plt.plot(times[i], y, color=color, linestyle=style, linewidth=width, marker="+", label="iter = " + str(i))
        else:
            width = 2
            style = "dashed"
            plt.plot(times[i], y, linestyle=style, linewidth=width, label="iter = " + str(i))
    plt.suptitle(title, fontsize=24)
    plt.xlabel('time', fontsize=20)
    plt.ylabel(name, fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)


def plot_figure(time, y, name, title):
    plt.figure()
    color = "black"
    width = 3
    style = "solid"
    plt.plot(time, y, color=color, linestyle=style, linewidth=width)
    plt.suptitle(title, fontsize=24)
    plt.xlabel('time', fontsize=20)
    plt.ylabel(name, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)