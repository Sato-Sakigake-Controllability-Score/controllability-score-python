# src/cs/solvers/projected_gradient.py
from __future__ import annotations

from typing import Callable, Tuple, Optional, List
import numpy as np
import numpy.typing as npt

from ..options import PGSolverOptions
from ..results import CSResult, CSTraceItem


Array = npt.NDArray[np.float64]
Fun = Callable[[Array], Tuple[float, Array]]
Proj = Callable[[Array], Array]


class ProjectedGradientSolver:
    """
    Projected gradient method with Armijo backtracking (projection arc).

    Required objective signature:
        f, g = fun(p)
    where:
        - f: scalar objective value (Inf allowed)
        - g: gradient vector (NaN entries allowed to signal invalidity)

    Robustness rule (same as MATLAB):
        - Always evaluate (f,g) at trial points.
        - Reject a trial point if f is not finite OR g contains NaN.

    Stopping:
        - step_norm = ||p_new - p||_2 <= tol
        - backtracking fails if alpha < step_size_inf
    """

    def __init__(self, options: Optional[PGSolverOptions] = None) -> None:
        self.options = PGSolverOptions() if options is None else options

    @staticmethod
    def _as_1d_float(x: npt.ArrayLike, name: str) -> Array:
        a = np.asarray(x, dtype=float)
        if a.ndim != 1:
            raise ValueError(f"{name} must be a 1D real vector")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite")
        return a

    @staticmethod
    def _eval_fg(fun: Fun, p: Array) -> Tuple[float, Array]:
        f, g = fun(p)
        f = float(f)
        g = np.asarray(g, dtype=float).reshape(-1)
        return f, g

    def solve(self, fun: Fun, proj: Proj, p0: npt.ArrayLike) -> Tuple[Array, CSResult]:
        """
        Run projected gradient method with Armijo backtracking.

        Returns:
            p: final point
            info: CSResult diagnostics
        """
        opt = self.options
        func_count = 0

        # validate p0
        p0v = self._as_1d_float(p0, "p0")

        # initial projection
        p = np.asarray(proj(p0v), dtype=float).reshape(-1)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("proj(p0) must return a non-empty 1D vector")

        # evaluate at initial point
        fp, gp = self._eval_fg(fun, p)
        func_count += 1

        if (not np.isfinite(fp)) or np.any(np.isnan(gp)):
            info = CSResult(
                objective_value=fp,
                gradient=gp,
                grad_norm=np.nan if (gp.size == 0 or np.any(np.isnan(gp))) else float(np.linalg.norm(gp)),
                step_norm=np.nan,
                iterations=0,
                func_count=func_count,
                converged=False,
                exit_flag=-2,
                exit_message="Initial point evaluation failed.",
                algorithm="ProjectedGradient (Armijo, projection arc)",
                solver_options=opt,
                trace=None,
            )
            return p, info

        trace: Optional[List[CSTraceItem]] = [] if opt.store_trace else None

        converged = False
        exit_flag = 0
        exit_message = ""
        step_norm = float("inf")

        alpha0 = opt.step_size  # initial trial step size each iteration
        iters_done = 0

        for it in range(1, opt.max_iter + 1):
            iters_done = it
            alpha = alpha0
            accepted = False

            while alpha >= opt.step_size_inf:
                # trial point on projection arc
                p_tilde = np.asarray(proj(p - alpha * gp), dtype=float).reshape(-1)

                # always evaluate at trial point
                f_tilde, g_tilde = self._eval_fg(fun, p_tilde)
                func_count += 1

                # reject invalid trial point
                if (not np.isfinite(f_tilde)) or np.any(np.isnan(g_tilde)):
                    alpha = opt.rho * alpha
                    continue

                # Armijo condition
                rhs = fp + opt.sigma * float(gp @ (p_tilde - p))
                if f_tilde <= rhs:
                    accepted = True
                    break

                alpha = opt.rho * alpha

            if not accepted:
                exit_flag = -1
                exit_message = "Armijo backtracking failed (alpha < step_size_inf)."
                break

            # accept
            p_prev = p
            p = p_tilde
            fp = f_tilde
            gp = g_tilde

            step_norm = float(np.linalg.norm(p - p_prev, 2))

            if trace is not None:
                trace.append(
                    CSTraceItem(
                        iter=it,
                        objective=float(fp),
                        grad_norm=float(np.linalg.norm(gp)) if not np.any(np.isnan(gp)) else float("nan"),
                        step_norm=step_norm,
                        step_size=float(alpha),
                    )
                )

            if opt.verbose:
                print(f"Iter {it:4d}  f={fp: .6e}  step={step_norm: .3e}  alpha={alpha: .3e}")

            # stopping criterion
            if step_norm <= opt.tol:
                converged = True
                exit_flag = 1
                exit_message = "Step norm below tol."
                break

        if (not converged) and exit_flag == 0 and iters_done >= opt.max_iter:
            exit_flag = 0
            exit_message = "Maximum iterations reached."

        grad_norm = float("nan") if (gp.size == 0 or np.any(np.isnan(gp))) else float(np.linalg.norm(gp, 2))

        info = CSResult(
            objective_value=float(fp),
            gradient=gp,
            grad_norm=grad_norm,
            step_norm=float(step_norm),
            iterations=int(iters_done if iters_done is not None else 0),
            func_count=int(func_count),
            converged=bool(converged),
            exit_flag=int(exit_flag),
            exit_message=str(exit_message),
            algorithm="ProjectedGradient (Armijo, projection arc)",
            solver_options=opt,
            trace=trace,
        )
        return p, info
