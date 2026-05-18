# src/cs/solvers/aecs_solver.py
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..problem import CSProblem
from ..results import CSResult
from ..projections.simplex import project_onto_simplex
from ..pg_solvers.project_gradient import ProjectedGradientSolver
from .solve_context import SolveContext
from ..utils.assemble import assemble_w_block



Array = npt.NDArray[np.float64]
Fun = Callable[[Array], Tuple[float, Array]]


def make_aecs_fun(problem: CSProblem) -> Fun:
    """
    Build AECS objective (f,g) from a CSProblem.

    MATLAB reference (evalAecs):
      f(p) = sum_{k in AecsBlocks} trace( inv(W_k(p)) * S_kk ),
      where S = problem.aecs_matrix is dense on the AECS subspace and S_kk is the
      diagonal block corresponding to k under the AECS subspace ordering induced
      by problem.aecs_blocks.

    Gradient:
      g_i = - sum_k trace( (Winv*Skk*Winv) * W_{i,k} ).

    Notes:
    - problem.aecs_blocks are 0-based
    - problem.block_sizes is assumed to be per original block index .
    - w_list[i][k0] stores W_{i,k} (k0 is 0-based).
    """
    n = int(problem.dimension)
    w_list = problem.wlist.w_list

    blocks_1based = np.asarray(problem.wlist.aecs_blocks, dtype=np.int64).reshape(-1)
    S = problem.wlist.aecs_matrix



    def fun(p: Array) -> Tuple[float, Array]:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if p.size != n:
            raise ValueError(f"p must have length {n}, got {p.size}")

        f = 0.0
        g = np.zeros(n, dtype=np.float64)

        for k1 in blocks_1based:
            # Assemble AECS block W_k(p)
            Wk = assemble_w_block(problem, p, int(k1))

            # SPD check + Cholesky
            try:
                L = np.linalg.cholesky(Wk)  # W = L L^T
            except np.linalg.LinAlgError:
                return np.inf, np.full(n, np.nan, dtype=np.float64)
            nk = Wk.shape[0]

            # Winv = inv(Wk) via solves
            eye = np.eye(nk, dtype=Wk.dtype)
            X = np.linalg.solve(L, eye)   # L^{-1}
            Winv = X.T @ X                # L^{-T} L^{-1}
            if S is not None:

                # Diagonal block S_kk under AECS subspace ordering
                Skk = S[int(k1)]

                # f += trace(Winv * Skk) = sum(sum(Winv .* Skk.'))
                if Skk is None:
                    raise ValueError("aecs matrix must not null.")
                f += float(np.sum(Winv * Skk.T))

                # Gradient:
                # Gk = Winv * Skk * Winv
                Gk = Winv @ Skk @ Winv
            else:
                f += float(np.sum(Winv))
                Gk = Winv @ Winv

            for i in range(n):
                Bik = np.asarray(w_list[i][k1], dtype=np.float64)
                # g_i += -trace(Gk * Bik) = -sum(sum(Gk .* Bik.'))
                g[i] -= float(np.sum(Gk * Bik.T))

        if not np.isfinite(f):
            f = np.inf
        if np.any(~np.isfinite(g)):
            g[:] = np.nan

        return f, g

    return fun


def solve_aecs(
    problem: CSProblem,
    *,
    context: Optional[SolveContext] = None,
) -> Tuple[Array, CSResult]:
    """
    Solve AECS for a CSProblem using ProjectedGradientSolver.

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

    fun = make_aecs_fun(problem)
    proj = project_onto_simplex

    solver = ProjectedGradientSolver(ctx.solver_options)
    p, info = solver.solve(fun, proj, p0)


    return p, info
