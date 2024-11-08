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
import sys
from nlse_funs import solve_newton, estimate_rms, create_new_xp_z_zmid, row_col_jac_indices
import casadi as cs
from inspect import ismethod


def _has_method(obj, meth):
    return hasattr(obj, meth) and ismethod(getattr(obj, meth))


class BVPSol:
    def __init__(self, time=None, xp=None, z=None, zmid=None, infos=None):
        self.time = time
        self.xp = xp
        self.z = z
        self.zmid = zmid
        self.infos = infos


class Infos:

    def __init__(self, success, NLSE_infos, DAE_infos):
        """
        This class contains the BVPDAE's numerical solving informations
        :param success: Boolean indicating if the problems is successfully solved
        :param NLSE_infos: class whose attributes gather informations on the Non Linear Equations Solver
        :param DAE_infos: classe whose attributes gather informations on the mesh refinement procedure
        """
        self.success = success
        self.NLSE_infos = NLSE_infos
        self.DAE_infos = DAE_infos


class DAEInfos:
    def __init__(self, success, rms_residual):
        """
        This class contains the BVPDAE's numerical solving informations
        :param success: Boolean indicating if the BVPDAE is solved with required discretization residual error
        :param rms_residual: numpy array with ODEs discretization residual error
        """
        self.success = success
        self.rms_residual = rms_residual


class BVPDAE:
    """
This is a Two Point Boundary Value Problem solver consisting in ODEs coupled with DAEs. The parameterization of the solver is done through an options dictionnary given at instanciation containing the following items
    - **display:** Verbosity of the algorithm (ranging from 0 to 2). Default is 0
    - **check_jacobian:** Boolean when True checks provided Jacobian. Default is False
    - **approx_prb:** Boolean when True algebraic variables are computed exactly at collocation points. If False, zmid is interpolated.
    - **control_odes_error:** Boolean when True time mesh is adapted to control residual error. Default is False
    - **max_mesh_point:** Maximum length of time array. Default is 1e5
    - **res_tol:**: residual relative error on ODEs. Default is 1e-3
    - **no_mesh_reduction:** Boolean when True mesh modification can only add points., Default is False
    - **threshold_mesh_reduction:** Real value in (0,1] such that mesh points are removed if ODE's residual error is <= threshold_mesh_reduction * res_tol on three consecutive time interval.
    - **max_NLSE_iter:** Maximum number of iteration for solving the NLSE. Default is 100 in case of mesh refinement 500 otherwise
    - **max_rms_control_iter:** Maximum number of mesh modification iterations. Default is 1000
    - **newton_tol:** relative tolerance of Newton scheme. Default is 1e-10
    - **abs_tol:**: residual absolute error on ODEs. Default is 1e-9
    - **coeff_damping:** Damping coefficient of damping Newton-Raphson scheme. Default is 2.
    - **opt_solver:** Numerical solver to be chosen among ls_newton, lsr_newton. Default is ls_newton
    - **linear_solver:** Linear solver to be chosen among lu or umfpack. Default is umfpack
    - **coeff_damping:** Damping coefficient of damping Newton scheme. Default is 2.
    - **max_probes:** Maximum number of damping operations for armijo step selection. Default is 6
    - **reg_hess:** Hessian regularization parameters. Default is (0., 1e-7)
    - **max_reg_hessian_probes:** Maximum probes for hessian regularization. Default is 10
    """
    def __init__(self, **kwargs):
        self.display = kwargs.get("display", 0)
        self.check_jacobian = kwargs.get("check_jacobian", False)
        self.approx_prb = kwargs.get("approx_prb", True)

        self.control_odes_error = kwargs.get("control_odes_error", False)
        self.max_mesh_point = kwargs.get("max_mesh_point", 100000)
        self.res_tol = kwargs.get("res_tol", 1e-3)
        self.no_mesh_reduction = kwargs.get("no_mesh_reduction", False)
        if self.no_mesh_reduction:
            max_NLSE_iter = 500
        else:
            max_NLSE_iter = 100
        self.max_NLSE_iter = kwargs.get("max_NLSE_iter", max_NLSE_iter)
        self.threshold_mesh_reduction = min(1., kwargs.get("threshold_mesh_reduction", .1))
        self.max_rms_control_iter = kwargs.get("max_rms_control_iter", 1000)

        self.newton_tol = kwargs.get("newton_tol", 1e-10)
        self.abs_tol = kwargs.get("abs_tol", 1e-9)
        self.coeff_damping = kwargs.get("coeff_damping", 2.)
        self.max_probes = kwargs.get("max_probes", 6)
        _HESS_REG = kwargs.get("hess_reg", (0., 1e-7))
        if len(_HESS_REG) != 2:
            self._HESS_REG = (0., 1e-7)
        else:
            self._HESS_REG = _HESS_REG
        self.max_reg_hessian_probes = kwargs.get("c", 10)
        opt_solver = kwargs.get("opt_solver", "ls_newton")
        if opt_solver not in ["ls_newton", "lsr_newton"]:
            opt_solver = "ls_newton"
        if opt_solver == "ls_newton":
            self.opt_solver = 0
        else:
            self.opt_solver = 1
        linear_solver = kwargs.get("linear_solver", "umfpack")
        if linear_solver not in ["umfpack", "lu"]:
            linear_solver = "umfpack"
        if linear_solver == "lu":
            self.linear_solver = 0
        else:
            self.linear_solver = 1

    def solve(self, bvp_sol, ocp):
        """
        Solve the Optimal Control Problem from OCP initializing the algorithm with (time, xp, z) values

        :param sol_bvp: BVPsol
        :param ocp: Object representing an indirect optimal control problem
        :return: bvp_sol, infos.
        """

        # All OCPs must have an ODE method
        assert _has_method(ocp, "ode"), "ocp must have a ode(time, ode_sol, alg_sol) method"

        # All OCPs must have an twobc method
        assert _has_method(ocp, "twobc"), \
            "ocp must have a twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T) method"

        # Tests to see if provided ocp has all jacobian
        provided_jacobians = (
                _has_method(ocp, "odejac") and _has_method(ocp, "algjac") and _has_method(ocp, "bcjac")
        )

        dae_infos = None
        time, xp, z, zmid = bvp_sol.time, bvp_sol.xp, bvp_sol.z, bvp_sol.zmid

        if not provided_jacobians:
            ocp = _csOCP(ocp, xp, z)

        ne, na = xp.shape[0], z.shape[0]
        rowis, colis, shape_jac, Inn, res_odeis, res_algis = row_col_jac_indices(time, ne, na)
        if zmid is None:
            zmid = .5 * (z[:, :-1] + z[:, 1:])
        if self.control_odes_error:
            rms_control_iter = 0
            # Begining of the iterative solving of the TPBVP
            while rms_control_iter < self.max_rms_control_iter:
                # Computing the solution of grad_dae = 0 through a Newton scheme
                (xpnew, znew, zmidnew, rhsmidnew, nlse_infos)\
                    = solve_newton(time, xp, z, zmid, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                                   res_tol=self.res_tol, max_iter=self.max_NLSE_iter, display=self.display,
                                   linear_solver=self.linear_solver, atol=self.newton_tol,
                                   coeff_damping=self.coeff_damping, max_probes=self.max_probes)

                if nlse_infos.ode_residual is None or nlse_infos.ae_residual is None or nlse_infos.bc_residual is None:
                    infos = Infos(False, nlse_infos, dae_infos)
                    break
                else:
                    xp, z, zmid, rhsmid = xpnew, znew, zmidnew, rhsmidnew

                rms_res, fun_interp_z = estimate_rms(time, xp, z, zmid, ocp, atol=self.abs_tol, restol=self.res_tol)
                max_rms_res = np.max(rms_res)
                if np.isnan(max_rms_res):
                    dae_infos = DAEInfos(False, rms_res)
                    infos = Infos(False, nlse_infos, dae_infos)
                    break

                if self.display >= 1:
                    print('     # Residual error = ' + str(max_rms_res) + ' with N = ' + str(len(time)))
                    if self.display >= 2:
                        print('     ')
                sys.stdout.flush()
                if max_rms_res < self.res_tol:
                    dae_infos = DAEInfos(True, rms_res)
                    infos = Infos(True, nlse_infos, dae_infos)
                    break
                else:
                    new_time, new_xp, new_z, new_zmid, too_much_nodes = create_new_xp_z_zmid(
                        time, xp, z, fun_interp_z, rms_res, ocp, restol=self.res_tol,
                        coeff_reduce_mesh=self.threshold_mesh_reduction, nmax=self.max_mesh_point,
                        authorize_reduction=not self.no_mesh_reduction
                    )

                    if not too_much_nodes:
                        time, xp, z, zmid = new_time, new_xp, new_z, new_zmid
                    else:
                        dae_infos = DAEInfos(False, rms_res)
                        infos = Infos(False, nlse_infos, dae_infos)
                        break
                    rowis, colis, shape_jac, Inn, res_odeis, res_algis = row_col_jac_indices(time, ne, na)
                rms_control_iter += 1
        else:
            (xp, z, zmid, rhsmid, nlse_infos) \
                = solve_newton(time, xp, z, zmid, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                               res_tol=self.res_tol, max_iter=self.max_NLSE_iter, display=self.display,
                               linear_solver=self.linear_solver, atol=self.newton_tol, coeff_damping=self.coeff_damping,
                               max_probes=self.max_probes)
            infos = Infos(nlse_infos.success, nlse_infos, dae_infos)
        if self.display > 0:
            print("Solving complete")
        return BVPSol(time=time, xp=xp, z=z, zmid=zmid), infos


def build_running_fun_and_jacobian(fun, ode_sol_val, alg_sol_val):
    # Initialize symbolic variables
    t = cs.MX.sym('t', 1)
    ode_sol = cs.MX.sym('ode_sol', ode_sol_val.shape[0])
    alg_sol = cs.MX.sym('alg_sol', alg_sol_val.shape[0])

    # create CASADi function
    cfun = fun(t, ode_sol, alg_sol)
    F = cs.Function('F', [t, ode_sol, alg_sol], [cfun])

    # Compute the Jacobian of f with respect to ode_sol
    jac_wrt_ode_sol = cs.jacobian(cfun, ode_sol)
    # Create a CASADi function for the Jacobian
    jac_wrt_ode_sol_func = cs.Function('jac_wrt_ode_sol_func', [t, ode_sol, alg_sol], [jac_wrt_ode_sol])

    # Compute the Jacobian of f with respect to alg_sol
    jac_wrt_alg_sol = cs.jacobian(cfun, alg_sol)
    # Create a CASADi function for the Jacobian
    jac_wrt_alg_sol_func = cs.Function('jac_wrt_alg_sol_func', [t, ode_sol, alg_sol], [jac_wrt_alg_sol])

    return F, jac_wrt_ode_sol_func, jac_wrt_alg_sol_func


def build_bc_fun_and_jacobian(twobc, ode_sol_0_val, ode_sol_T_val, alg_sol_0_val, alg_sol_T_val):
    # Initialize symbolic variables
    ode_sol_0 = cs.MX.sym('ode_sol_0', ode_sol_0_val.size)
    ode_sol_T = cs.MX.sym('ode_sol_T', ode_sol_T_val.size)
    alg_sol_0 = cs.MX.sym('alg_sol_0', alg_sol_0_val.size)
    alg_sol_T = cs.MX.sym('alg_sol_T', alg_sol_T_val.size)

    # create CASADi function
    ctwobc = twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)
    twobc_func = cs.Function('twobc_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [ctwobc])

    # Compute the Jacobian of f with respect to ode_sol_0
    jacbc_wrt_ode_sol_0 = cs.jacobian(ctwobc, ode_sol_0)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_ode_sol_0_func = cs.Function(
        'jacbc_wrt_ode_sol_0_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_ode_sol_0]
    )

    # Compute the Jacobian of f with respect to ode_sol_T
    jacbc_wrt_ode_sol_T = cs.jacobian(ctwobc, ode_sol_T)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_ode_sol_T_func = cs.Function(
        'jacbc_wrt_ode_sol_T_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_ode_sol_T]
    )

    # Compute the Jacobian of f with respect to alg_sol_0
    jacbc_wrt_alg_sol_0 = cs.jacobian(ctwobc, alg_sol_0)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_alg_sol_0_func = cs.Function(
        'jacbc_wrt_alg_sol_0_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_alg_sol_0]
    )

    # Compute the Jacobian of f with respect to alg_sol_T
    jacbc_wrt_alg_sol_T = cs.jacobian(ctwobc, alg_sol_T)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_alg_sol_T_func = cs.Function(
        'jacbc_wrt_alg_sol_T_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_alg_sol_T]
    )

    return (
        twobc_func, jacbc_wrt_ode_sol_0_func, jacbc_wrt_ode_sol_T_func, jacbc_wrt_alg_sol_0_func,
        jacbc_wrt_alg_sol_T_func
    )


class _csOCP:
    """This class links a TPBVP written in the Casadi framework into the NumPy framework to be solved using the BVPDAE class.

        :param ocp: ocp written in the Casadi framework, returning the right-hand side of the ODEs, the algebraic equations, and the boundary conditions.
        :param ode_sol: NumPy array containing the ODEs' solution.
        :param alg_sol: NumPy array containing the solution of the algebraic equations.
    """
    def __init__(self, ocp, ode_sol, alg_sol):

        self._ode, self._jacode_wrt_ode_sol, self._jacode_wrt_alg_sol = build_running_fun_and_jacobian(
            ocp.ode, ode_sol, alg_sol
        )

        self._algeq, self._jacalg_wrt_ode_sol, self._jacalg_wrt_alg_sol = build_running_fun_and_jacobian(
            ocp.algeq, ode_sol, alg_sol
        )

        self._twobc, self._jacbc_ode_sol_0, self._jacbc_ode_sol_T, self._jacbc_alg_sol_0, self._jacbc_alg_sol_T = (
            build_bc_fun_and_jacobian(ocp.twobc, ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1])
        )

    def ode(self, time, ode_sol, alg_sol):

        """This function returns the ODEs' right hand side in NumPy array format

            :param time: Numpy array of discretization time steps
            :param ode_sol: Numpy array of ODEs' solution at discretization points
            :param alg_sol: Numpy array of algebraic solutions at discretization points
            :return: **rhs_ode**: Numpy array of time derivative of ode_sol at discretization points
        """
        ode_vec = self._ode.map(len(time))
        return np.array(ode_vec(time, ode_sol, alg_sol)).reshape(ode_sol.shape, order="F")

    def algeq(self, time, ode_sol, alg_sol):
        """This function returns the residual of the algbreaic equations

            :param time:
                Numpy array of time steps at discretization points
            :param ode_sol:
                2d Numpy array of ODEs'solution at discretization points
            :param alg_sol:
                2d Numpy array of algebraic variables at discretization points
            :return:
                **res_alg_eq**: 2d Numpy array of residual of algebraic equations
        """
        algeq_vec = self._algeq.map(len(time))
        return np.array(algeq_vec(time, ode_sol, alg_sol)).reshape(alg_sol.shape, order="F")

    def twobc(self, ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T):

        return np.array(self._twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)).flatten()

    def odejac(self, time, ode_sol, alg_sol):
        """This function returns the Jacobians of the ODEs' right-hand side as three dimensional Numpy arrays

            :param time:
                Numpy array of discretization time steps
            :param ode_sol:
                Numpy array of ODEs' solution at discretization points
            :param alg_sol:
                Numpy array of algebraic solutions at discretization points
            :return:
                - **jac_ode_wrt_ode_sol**: 3D Numpy array of ODE's right hand side Jacobian with respect to ode_sol at discretization points
                - **jac_ode_wrt_ode_sol**: 3D Numpy array of ODE's right hand side Jacobian with respect to alg_sol at discretization points
        """
        jacode_wrt_ode_sol_vectorized = self._jacode_wrt_ode_sol.map(len(time))
        jacode_wrt_alg_sol_vectorized = self._jacode_wrt_alg_sol.map(len(time))
        return (
            np.array(jacode_wrt_ode_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                ode_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1], order="F"
            ),
            np.array(jacode_wrt_alg_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                ode_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1], order="F"
            )
        )

    def algjac(self, time, ode_sol, alg_sol):
        """This function returns the Jacobians of the algebraix equations as three dimensional Numpy arrays

            :param time: Numpy array of discretization time steps
            :param ode_sol: Numpy array of ODEs' solution at discretization points
            :param alg_sol: Numpy array of algebraic solutions at discretization points
            :return:
                - **jac_alg_wrt_ode_sol**: 3D Numpy array of algebraic equations Jacobian with respect to ode_sol at discretization points
                - **jac_alg_wrt_alg_sol**: 3D Numpy array of algebraic equations Jacobian with respect to alg_sol at discretization points
        """
        jacalg_wrt_ode_sol_vectorized = self._jacalg_wrt_ode_sol.map(len(time))
        jacalg_wrt_alg_sol_vectorized = self._jacalg_wrt_alg_sol.map(len(time))
        return (
            np.array(jacalg_wrt_ode_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                alg_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1], order="F"
            ),
            np.array(jacalg_wrt_alg_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                alg_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1], order="F"
            )
        )

    def bcjac(self, ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T):

        return (
            np.array(self._jacbc_ode_sol_0(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)),
            np.array(self._jacbc_ode_sol_T(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)),
            np.array(self._jacbc_alg_sol_0(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)),
            np.array(self._jacbc_alg_sol_T(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T))
        )





