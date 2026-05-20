# tests/solvers/test_aecs_basic.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from controllability_scoring.problem import CSProblem
from controllability_scoring.options import PGSolverOptions
from controllability_scoring.solvers.solve_aecs import solve_aecs, make_aecs_fun
from controllability_scoring.solvers.solve_context import SolveContext

Array = npt.NDArray[np.float64]


def finite_diff_grad(fun, p: Array, eps: float = 1e-6) -> Array:
    """Central difference (local test only; no projection)."""
    n = p.size
    g = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dp = np.zeros(n, dtype=np.float64)
        dp[i] = eps
        f1, _ = fun(p + dp)
        f2, _ = fun(p - dp)
        g[i] = (f1 - f2) / (2.0 * eps)
    return g


def assert_on_simplex(p: Array, atol: float = 1e-8):
    assert p.ndim == 1
    assert np.all(p >= -1e-12)
    assert abs(float(np.sum(p)) - 1.0) <= atol


@pytest.fixture
def tiny_problem() -> CSProblem:
    """
    Small, stable (Hurwitz) problem so the infinite-horizon Lyapunov Gramian exists.
    """
    A_pos = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ],
        dtype=np.float64,
    )
    A = -A_pos
    T = float("inf")
    return CSProblem(A, T)


def test_make_aecs_fun_gradient_matches_finite_diff(tiny_problem: CSProblem):
    """
    Analytic gradient from make_aecs_fun should match finite-difference gradient.
    Catches sign mistakes, transpose mistakes (Bik.T), and block indexing bugs.
    """
    prob = tiny_problem
    fun = make_aecs_fun(prob)
    n = int(prob.dimension)

    ctx = SolveContext()
    p0 = ctx.resolve_initial_guess(n)

    # Keep away from simplex corners (numerically safer for finite differences)
    p = 0.9 * p0 + 0.1 * (np.ones_like(p0) / n)

    f, g = fun(p)
    assert np.isfinite(f)
    assert np.all(np.isfinite(g))

    g_fd = finite_diff_grad(fun, p, eps=1e-6)

    # Finite-difference is noisy; use mild tolerances
    assert np.allclose(g, g_fd, rtol=1e-5, atol=1e-6)


def test_solve_aecs_returns_simplex_solution(tiny_problem: CSProblem):
    """
    solve_aecs should return a finite solution on the simplex.
    """
    prob = tiny_problem

    opt = PGSolverOptions()
    ctx = SolveContext(solver_options=opt)

    p, info = solve_aecs(prob, context=ctx)

    assert_on_simplex(p)
    assert np.all(np.isfinite(p))


def test_solve_aecs_does_not_increase_objective(tiny_problem: CSProblem):
    """
    AECS is a minimization problem. The returned objective should not be worse than
    the objective at the context-initialized starting point.
    """
    prob = tiny_problem
    fun = make_aecs_fun(prob)
    n = int(prob.dimension)

    opt = PGSolverOptions()
    ctx = SolveContext(solver_options=opt)

    p0 = ctx.resolve_initial_guess(n)
    f0, _ = fun(p0)
    assert np.isfinite(f0)

    p, _ = solve_aecs(prob, context=ctx)

    f1, _ = fun(p)
    assert np.isfinite(f1)

    # Allow tiny numerical slack
    assert f1 <= f0 + 1e-8
