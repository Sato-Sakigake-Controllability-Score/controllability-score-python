# src/cs/solvers/vcs_solver.py
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..problem import CSProblem
from ..results import CSResult
from ..projections.simplex import project_onto_simplex
from ..pg_solvers.project_gradient import ProjectedGradientSolver
from ..utils.assemble import assemble_w_block
from .solve_context import SolveContext


Array = npt.NDArray[np.float64]
Fun = Callable[[Array], Tuple[float, Array]]


def make_vcs_fun(problem: CSProblem) -> Fun:
    """
    Build VCS objective (f,g) from a CSProblem.

    MATLAB reference (evalVcs):
      f(p) = - sum_{k in VcsBlocks} logdet(W_k(p)),
      W_k(p) = sum_i p_i * W_{i,k}.

    logdet via Cholesky:
      Wk = R'R  => logdet(Wk) = 2*sum(log(diag(R))).

    Gradient:
      g_i = - sum_k trace( inv(Wk) * W_{i,k} ).

    Notes:
    - problem.vcs_blocks are 1-based (MATLAB style).
    - w_list[i][k0] stores W_{i,k} (k0 is 0-based).
    """
    n = int(problem.dimension)
    w_list = problem.wlist.w_list

    blocks_1based = np.asarray(problem.wlist.vcs_blocks, dtype=np.int64).reshape(-1)

    def fun(p: Array) -> Tuple[float, Array]:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if p.size != n:
            raise ValueError(f"p must have length {n}, got {p.size}")

        f = 0.0
        g = np.zeros(n, dtype=np.float64)

        for k1 in blocks_1based:
            # Assemble W_k(p)
            Wk = assemble_w_block(problem, p, int(k1))

            # SPD check + Cholesky
            try:
                L = np.linalg.cholesky(Wk)  # W = L L^T
            except np.linalg.LinAlgError:
                return np.inf, np.full(n, np.nan, dtype=np.float64)

            # f += -logdet(Wk) = -2*sum(log(diag(L)))
            d = np.diag(L)
            if np.any(d <= 0):
                return np.inf, np.full(n, np.nan, dtype=np.float64)

            f -= 2.0 * float(np.sum(np.log(d)))

            # Gradient:
            # Winv = inv(Wk) via solves (MATLAB: R \ (R' \ I))
            nk = Wk.shape[0]
            eye = np.eye(nk, dtype=Wk.dtype)
            X = np.linalg.solve(L, eye)   # L^{-1}
            Winv = X.T @ X              # L^{-T} L^{-1}

            for i in range(n):
                Bik = np.asarray(w_list[i][k1], dtype=np.float64)
                # g_i += -trace(Winv @ Bik)  == -sum(sum(Winv .* Bik.'))
                g[i] -= float(np.sum(Winv * Bik.T))

        if not np.isfinite(f):
            f = np.inf
        if np.any(~np.isfinite(g)):
            g[:] = np.nan

        return f, g

    return fun


def solve_vcs(
    problem: CSProblem,
    *,
    context: Optional[SolveContext] = None,
) -> Tuple[Array, CSResult]:
    """
    Solve vcs for a CSProblem using ProjectedGradientSolver.

    Parameters
    ----------
    problem : CSProblem
        Precomputed problem instance (validated & structured).
    solver_options : PGSolverOptions, optional
        Projected gradient solver options.
    initial_guess : array-like, optional
        Initial p. Defaults to problem.initial_guess.

    Returns
    -------
    p : ndarray, shape (n,)
        Solution.
    info : CSResult
        Diagnostics.
    """
    ctx = SolveContext() if context is None else context

    p0 = ctx.resolve_initial_guess(problem.dimension)

    if p0.size != problem.dimension:
        raise ValueError(f"InitialGuess must have length {problem.dimension}.")

    fun = make_vcs_fun(problem)
    proj = project_onto_simplex

    solver = ProjectedGradientSolver(ctx.solver_options)
    p, info = solver.solve(fun, proj, p0)


    return p, info
