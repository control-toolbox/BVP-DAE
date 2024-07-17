import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
from scipy.interpolate import interp1d
import sys


class NLSEInfos:
    def __init__(self, success, iter, ode_residual, ode_tol, ae_residual, ae_tol, bc_residual, bc_tol):
        self.success = success
        self.iter = iter
        self.ode_residual = ode_residual
        self.ode_tol = ode_tol
        self.ae_residual = ae_residual
        self.ae_tol = ae_tol
        self.bc_residual = bc_residual
        self.bc_tol = bc_tol


def res_colocation(time, xp, z, zmid, ocp):
    h = np.diff(time)  # vector containing the length of every time step of the time-grid

    tmid = (time[:-1] + time[1:]) / 2.  # Time collocation point
    rhs = ocp.ode(time, xp, z)  # Evaluation of the diffential equations
    xmid = (xp[:, :-1] + xp[:, 1:]) / 2. - (rhs[:, 1:] - rhs[:, :-1]) * h / 8.  # State colocation point
    rhsmid = ocp.ode(tmid, xmid, zmid)  # ODEs at colocation point

    alg = ocp.algeq(time, xp, z)  # AEs at time t
    algmid = ocp.algeq(tmid, xmid, zmid)

    # Computation of the boundary condition
    bound_const = ocp.twobc(xp[:, 0], xp[:, -1], z[:, 0], z[:, -1])
    odes = xp[:, 1:] - xp[:, :-1] - (rhs[:, 1:] + 4. * rhsmid + rhs[:, :-1]) * h / 6.
    # Concatenation

    res = np.concatenate((bound_const, np.vstack((odes, alg[:, :-1], algmid)).ravel(order="F"), alg[:, -1]))
    return res, rhsmid


def solve_newton(time, xp, z, zmid, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                     res_tol=1e-3, max_iter=100, display=0, linear_solver=0, atol=1e-9,
                     coeff_damping=2., max_probes=8):
        success = False
        iter = 0
        alpha = -1.
        h = np.diff(time)
        tol_odes = 2/3 * h * 5e-2 * res_tol
        while not success and iter < max_iter:
            res, rhsmid = res_colocation(time, xp, z, zmid, ocp)
            jac = jac_res_colocation(time, xp, z, zmid, ocp, Inn, rowis, colis, shape_jac)
            if display == 2:
                print('        * Newton Iteration # ' + str(iter))
                print('        * Initial error = ' + str(np.max(np.abs(res))))
            if linear_solver == 0:
                direction = splu(jac).solve(res)
            else:
                direction = spsolve(jac, res, use_umfpack=True)
            xp_old, z_old, zmid_old, alpha_old = xp, z, zmid, alpha
            xp, z, zmid, cost, alpha, rhsmid, res, armflag = armijo(time, xp, z, zmid, ocp, direction, res, rhsmid,
                                                                    coeff_damping, max_probes)
            max_res = np.max(np.abs(res))
            tol = (tol_odes * (1. + np.abs(rhsmid))).ravel(order="F")
            if (np.all(np.abs(res[res_odeis]) <= tol)
                    and np.all(np.abs(res[res_algis]) <= atol)
                    and np.all(np.abs(res[:xp.shape[0]]) <= atol)):
                success = True
                if display == 2:
                    print('           Success damped Newton step = ' + str(max_res) + ' alpha = ' + str(
                        alpha) + '; iter = ' + str(iter))

            elif display == 2:
                print(
                    '                  Newton step = ' + str(max_res) + ' alpha = ' + str(alpha) +
                    '; iter = ' + str(iter))

                sys.stdout.flush()
            if np.any(np.isnan(res)) or np.any(np.isinf(res)):
                nlse_infos = NLSEInfos(success, iter, None, tol, None, atol, None, atol)
                return xp, z, zmid, rhsmid, nlse_infos
            if (np.linalg.norm(xp - xp_old) == 0. and np.linalg.norm(z - z_old) == 0.
                    and np.linalg.norm(zmid - zmid_old) == 0. and alpha == alpha_old):
                break
            iter += 1
        ode_res = res[res_odeis].reshape((xp.shape[0], len(time) - 1), order="F")
        bc_res = res[:xp.shape[0]]
        nlse_infos = NLSEInfos(success, iter, ode_res, tol, res_algis, atol, bc_res, atol)
        return xp, z, zmid, rhsmid, nlse_infos


def armijo(time, xp0, z0, zmid0, OCP, direction, res0, rhsmid0, coeff_damping, max_probes):
    iarm = 0
    sigma1 = 1. / coeff_damping
    alpha = 1e-4
    armflag = True
    lbd, lbdm, lbdc = 1., 1., 1.
    dxp, dz, dzmid = get_solution_from_X(direction, xp0.shape[0], z0.shape[0])
    xpt = xp0 - lbd * dxp
    zt = z0 - lbd * dz
    zmidt = zmid0 - lbd * dzmid
    rest, rhsmidt = res_colocation(time, xpt, zt, zmidt, OCP)
    nft, nf0 = np.linalg.norm(rest), np.linalg.norm(res0)
    ff0, ffc = nf0 * nf0, nft * nft
    ffm = ffc
    best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res = xp0, z0, zmid0, ff0, lbd, rhsmid0, res0
    if ffc < best_cost:
        best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res = xpt, zt, zmidt, ffc, lbd, rhsmidt, rest
    while nft >= (1. - alpha * lbd) * nf0:
        if iarm == 0 or np.isinf(ffm) or np.isinf(ffc) or np.isnan(ffm) or np.isnan(ffc):
            lbd = sigma1 * lbd
        else:
            lbd = parab3p(lbdc, lbdm, ff0, ffc, ffm)
        xpt = xp0 - lbd * dxp
        zt = z0 - lbd * dz
        zmidt = zmid0 - lbd * dzmid
        lbdm = lbdc
        lbdc = lbd
        rest, rhsmidt = res_colocation(time, xpt, zt, zmidt, OCP)
        nft = np.linalg.norm(rest)
        ffm = ffc
        ffc = nft * nft
        if ffc < best_cost:
            best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res = xpt, zt, zmidt, ffc, lbd, rhsmidt, rest
        iarm += 1
        if iarm > max_probes:
            armflag = False
            return best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res, armflag
    return xpt, zt, zmidt, .5 * ffc, lbd, rhsmidt, rest, armflag


def parab3p(lbdc, lbdm, ff0, ffc, ffm):
    sigma0, sigma1 = .1, .5
    c2 = lbdm * (ffc - ff0) - lbdc * (ffm - ff0)
    if c2 >= 0.:
        return sigma1 * lbdc
    c1 = lbdc * lbdc * (ffm - ff0) - lbdm * lbdm * (ffc - ff0)
    lbdp = -c1 * .5 / c2
    if lbdp < sigma0 * lbdc:
        lbdp = sigma0 * lbdc
    if lbdp > sigma1 * lbdc:
        lbdp = sigma1 * lbdc
    return lbdp


def _get_X_from_solution(xp, z, zmid):
    return np.concatenate((np.concatenate([xp[:, :-1], z[:, :-1], zmid], axis=0).ravel(order='F'),
                           xp[:, -1], z[:, -1]))


def get_solution_from_X(x, ne, na):
    nt = (x.size - ne - na) // (ne + 2 * na)
    xzzmid = x[:nt * (ne + 2 * na)].reshape((ne + 2 * na, nt), order="F")
    xp = np.zeros((ne, nt + 1))
    xp[:, :-1] = xzzmid[:ne, :]
    xp[:, -1] = x[nt*(ne + 2 * na):nt*(ne + 2 * na) + ne]
    z = np.zeros((na, nt+1))
    z[:, :-1] = xzzmid[ne:ne+na, :]
    z[:, -1] = x[nt * (ne + 2 * na)+ne:]
    zmid = xzzmid[ne+na:, :]
    return xp, z, zmid


def jac_res_colocation(time, xp, z, zmid,ocp, Inn, rowis, colis, shape_jac):
    values = jac_res_values(time, xp, z, zmid, ocp, Inn)
    non_zeros_indices = np.nonzero(values)
    jac = sparse.csc_matrix((values[non_zeros_indices], (rowis[non_zeros_indices], colis[non_zeros_indices])), shape_jac)
    return jac


def jac_res_values(time, xp, z, zmid, ocp, Inn):
    N = len(time) - 1  # number of time step - 1
    h = np.diff(time)  # vector of length of every time step
    ne, na = xp.shape[0], z.shape[0]
    tmid = time[:-1] + h / 2.  # vector containing the midpoints of the time grid
    h3d = h.reshape((1, 1, N))
    h3d6 = h3d / 6.
    h3d8 = h3d / 8.
    rhs = ocp.ode(time, xp, z)  # ODEs at time t
    xmid = (xp[:, :-1] + xp[:, 1:]) / 2. - h / 8. * (rhs[:, 1:] - rhs[:, :-1])

    # Calling AEs jacobian
    gx, gz = ocp.algjac(time, xp, z)  # evaluate the jacobian of the algebraic equations
    gxmid, gzmid = ocp.algjac(tmid, xmid, zmid)
    # Calling ODEs jacobian
    fx, fz = ocp.odejac(time, xp, z)  # evaluate the jacobian of the ODEs
    fxmid, fzmid = ocp.odejac(tmid, xmid, zmid)  # evaluate the ODEs jacobian at midpoints

    dxmid_dxk = Inn / 2. + h3d8 * fx[:, :, :-1]
    dxmid_dzk = h3d8 * fz[:, :, :-1]
    dxmid_dxkp1 = Inn / 2. - h3d8 * fx[:, :, 1:]
    dxmid_dzkp1 = - h3d8 * fz[:, :, 1:]

    block_ode_alg_algmid = np.zeros((ne + 2 * na, 2 * ne + 3 * na, N))
    # derivative rhs wrt xk
    block_ode_alg_algmid[:ne, :ne, :] = - Inn - h3d6 * (4. * matmul3d(fxmid, dxmid_dxk) + fx[:, :, :-1])
    # derivative rhs wrt zk
    block_ode_alg_algmid[:ne, ne:ne+na, :] = - h3d6 * (4. * matmul3d(fxmid, dxmid_dzk) + fz[:, :, :-1])
    # derivative rhs wrt zmid
    block_ode_alg_algmid[:ne, ne+na:ne + 2*na, :] = - h3d6 * 4. * fzmid
    # derivative rhs wrt xk+1
    block_ode_alg_algmid[:ne, ne + 2 * na: 2 * (ne + na), :] = (
            Inn - h3d6 * (4. * matmul3d(fxmid, dxmid_dxkp1) + fx[:, :, 1:])
    )
    # derivative rhs wrt zk+1
    block_ode_alg_algmid[:ne, 2 * (ne + na):, :] =- h3d6 * (4. * matmul3d(fxmid, dxmid_dzkp1) + fz[:, :, 1:])

    # derivative alg wrt xk
    block_ode_alg_algmid[ne:ne+na, :ne, :] = gx[:, :, :-1]
    # derivative alg wrt zk
    block_ode_alg_algmid[ne:ne + na, ne: ne + na, :] = gz[:, :, :-1]

    # derivative algmid wrt xk
    block_ode_alg_algmid[ne + na:, :ne, :] = matmul3d(gxmid, dxmid_dxk)
    # derivative algmid wrt zk
    block_ode_alg_algmid[ne + na:, ne: ne + na, :] = matmul3d(gxmid, dxmid_dzk)
    # derivative algmid wrt zmid
    block_ode_alg_algmid[ne + na:, ne + na: ne + 2 * na, :] = gzmid
    # derivative algmid wrt xk+1
    block_ode_alg_algmid[ne + na:, ne + 2 * na: 2 * (ne + na), :] = matmul3d(gxmid, dxmid_dxkp1)
    # derivative algmid wrt zk+1
    block_ode_alg_algmid[ne + na:, 2 * (ne + na):, :] = matmul3d(gxmid, dxmid_dzkp1)

    # Computing the Hessian of the boundary conditions
    jac_bc_x0, jac_bc_xT, jac_bc_z0, jac_bc_zT = ocp.bcjac(xp[:, 0], xp[:, -1], z[:, 0], z[:, -1])

    # Gathering values in a numpy array
    vals = np.concatenate((
        np.hstack((jac_bc_x0, jac_bc_z0, jac_bc_xT, jac_bc_zT)).ravel(order="F"),
        block_ode_alg_algmid.ravel(order="F"),
        np.hstack((gx[:, :, -1], gz[:, :, -1])).ravel(order="F")
    ))
    return vals


def row_col_jac_indices(time, ne, na):
    N = len(time)
    # indices for bcjac
    row_bcjac = np.tile(np.arange(ne), 2*(ne+na))
    col_bcjac = (np.tile(np.repeat(np.arange(ne+na), ne), 2)
                 + np.repeat(np.array([0., (N-1) * (ne+2*na)]), ne*(ne+na)))

    # indices for block_ode_alg_algmid
    row_block_ode_alg_algmid = (ne + np.tile(np.tile(np.arange(ne+2*na), 2*ne + 3*na), N-1)
                                + np.repeat(np.arange(N - 1) * (ne + 2 * na), (ne + 2 * na) * (2 * ne + 3 * na)))

    col_block_ode_alg_algmid = (np.tile(np.repeat(np.arange(2 * ne + 3 * na), ne + 2*na), N - 1)
                                + np.repeat(np.arange(N - 1) * (ne + 2 * na), (ne + 2 * na) * (2 * ne + 3 * na)))

    range_eqxend = (ne + (N - 1) * (ne + 2 * na), ne + na + (N - 1) * (ne + 2 * na))
    row_algend = np.tile(np.arange(range_eqxend[0], range_eqxend[1]), ne + na)
    col_algend = np.repeat(np.arange((N - 1) * (ne + 2 * na), (N - 1) * (ne + 2 * na) + ne+na), na)
    rowis = np.concatenate((row_bcjac, row_block_ode_alg_algmid, row_algend))
    colis = np.concatenate((col_bcjac, col_block_ode_alg_algmid, col_algend))
    shape_jac = ((N - 1) * (ne + 2 * na) + ne + na,
                          (N - 1) * (ne + 2 * na) + ne + na)
    Inn = repmat(np.eye(ne), (1, 1, N - 1))

    res_odeis = np.tile(np.arange(ne), N - 1) + np.repeat(np.arange(N - 1) * (ne + 2 * na), ne) + ne
    res_algis = np.concatenate((2 * ne + np.tile(np.arange(2 * na), N - 1) + np.repeat(np.arange(N - 1) * (ne + 2 * na), 2 * na),
                               np.arange(ne + (N-1) * (ne + 2 * na), ne + (N-1) * (ne + 2 * na)+na)))
    return rowis, colis, shape_jac, Inn, res_odeis, res_algis


def estimate_rms(time, xp, z, zmid, ocp, atol=1e-9, restol=1e-3):
    h = np.diff(time)
    tmid = time[:-1] + h / 2.
    if zmid is not None:
        aggregated_z = np.concatenate(
            (np.reshape(np.vstack((z[:, :-1], zmid)), (z.shape[0], 2 * (len(time)-1)), order="F"),
             z[:, -1:]), axis=1)
        aggregated_time = np.sort(np.concatenate((time, tmid)))
        fun_interp_z = interp1d(x=aggregated_time, y=aggregated_z)
    else:
        zmid = .5 * (z[:, :-1] + z[:, 1:])
        fun_interp_z = interp1d(x=time, y=z)
    threshold = atol / restol
    lob4 = (1. + np.sqrt(3. / 7.)) / 2.
    lob2 = (1. - np.sqrt(3. / 7.)) / 2.
    lobw24 = 49. / 90.
    lobw3 = 32. / 45.
    rhs = ocp.ode(time, xp, z)

    xpmid = (xp[:, :-1] + xp[:, 1:]) / 2. - (rhs[:, 1:] - rhs[:, :-1]) * h / 8.
    rhsmid = ocp.ode(tmid, xpmid, zmid)
    colloc_res = xp[:, 1:] - xp[:, :-1] - (rhs[:, 1:] + 4. * rhsmid + rhs[:, :-1]) * h / 6.

    hscale = 1.5 / h
    temp = colloc_res * hscale / np.fmax(np.abs(rhsmid), threshold)
    res = lobw3 * np.sum(temp ** 2, axis=0)

    # Lobatto 2 points
    tlob = time[:-1] + lob2 * h
    xplob, derxp_lob = interp_hermite(h, xp, rhs, lob2)
    zlob = fun_interp_z(tlob)
    rhslob = ocp.ode(tlob, xplob, zlob)
    temp = (derxp_lob - rhslob) / np.fmax(np.abs(rhslob), threshold)
    res += lobw24 * np.sum(temp ** 2, axis=0)

    # Lobatto 4 points
    tlob = time[:-1] + lob4 * h
    xplob, derxp_lob = interp_hermite(h, xp, rhs, lob4)
    zlob = fun_interp_z(tlob)
    rhslob = ocp.ode(tlob, xplob, zlob)
    temp = (derxp_lob - rhslob) / np.fmax(np.abs(rhslob), threshold)
    res += lobw24 * np.sum(temp ** 2, axis=0)

    return np.sqrt(np.abs(h/2.) * res), fun_interp_z


def interp_hermite(h, xp, rhs, lob):
    scal = 1. / h
    slope = (xp[:, 1:] - xp[:, :-1]) * scal
    c = (3. * slope - 2. * rhs[:, :-1] - rhs[:, 1:]) * scal
    d = (rhs[:, :-1] + rhs[:, 1:] - 2. * slope) * scal ** 2

    scal = lob * h
    d *= scal
    xplob = ((d + c) * scal + rhs[:, :-1]) * scal + xp[:, :-1]
    derxp_lob = (3. * d + 2. * c) * scal + rhs[:, :-1]
    return xplob, derxp_lob


def create_new_xp_z_zmid(time, xp, z, fun_interp_z, residuals, ocp, restol=1e-3, coeff_reduce_mesh=.5, nmax=10000,
                         authorize_reduction=True):
    n = xp.shape[0]
    T = len(time)
    new_T = T + np.sum(np.where(residuals > restol, 1, 0)) + np.sum(np.where(residuals > 100. * restol, 1, 0))
    new_time = np.zeros((new_T,))
    new_time[0] = time[0]
    new_xp = np.zeros((n, new_T))
    rhs = ocp.ode(time, xp, z)
    ti = 0
    nti = 0
    new_xp[:, 0] = xp[:, 0]
    h = np.diff(time)
    while ti <= T-2:
        if residuals[ti] > restol:
            if residuals[ti] > 100. * restol:
                ni = 2
            else:
                ni = 1
            hi = h[ti] / (ni + 1)
            inds = np.arange(1, ni + 1)
            new_time[nti+1: nti + ni+1] = new_time[nti] + hi * inds
            xinterp = ntrp3h(new_time[nti: nti+ni], time[ti], xp[:, ti],
                             time[ti+1], xp[:, ti+1], rhs[:, ti], rhs[:, ti+1], ni)
            new_xp[:, nti+1:nti+ni+1] = xinterp
            nti += ni
        elif authorize_reduction and ti <= T-4 and max(residuals[ti:ti+3]) < restol * coeff_reduce_mesh:
            hnew = (time[ti+3] - time[ti]) / 2.
            pred_res = residuals[ti] / (h[ti] / hnew) ** 3.5
            pred_res = max(pred_res, residuals[ti+1] / ((time[ti+2] - time[ti+1]) / hnew) ** 3.5)
            pred_res = max(pred_res, residuals[ti+2] / ((time[ti+3] - time[ti+2]) / hnew) ** 3.5)
            if pred_res < restol * coeff_reduce_mesh:
                new_time[nti + 1] = new_time[nti] + hnew
                xinterp = ntrp3h(new_time[nti + 1], time[ti], xp[:, ti], time[ti + 3], xp[:, ti + 3], rhs[:, ti],
                                 rhs[:, ti + 3], 1)
                new_xp[:, nti + 1] = xinterp[:, 0]
                nti += 1
                ti += 2
        new_time[nti + 1] = time[ti + 1]
        new_xp[:, nti + 1] = xp[:, ti + 1]
        nti += 1
        ti += 1
    time = new_time[:nti+1]
    xp = new_xp[:, :nti+1]
    z = fun_interp_z(time)
    tmid = time[:-1] + np.diff(time) / 2.
    zmid = fun_interp_z(tmid)
    too_much_nodes = len(time) > nmax
    return time, xp, z, zmid, too_much_nodes


def ntrp3h(newtime, tk, xk, tkp1, xkp1, rhsk, rhskp1, ni):
    h = tkp1 - tk
    slope = (xkp1 - xk) / h
    c = 3. * slope - 2. * rhsk - rhskp1
    d = rhsk + rhskp1 - 2. * slope
    s = (newtime - tk) / h
    s2 = s ** 2
    s3 = s * s2
    xinterp = np.zeros((len(xk), ni))
    if ni == 1:
        xinterp[:, 0] = xk + h * (d * s3 + c * s2 + rhsk * s)
    else:
        for col in range(ni):
            xinterp[:, col] = xk + h * (d * s3[col] + c * s2[col] + rhsk * s[col])
    return xinterp


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

