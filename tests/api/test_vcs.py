# tests/test_vcs_centrality_validation.py
from __future__ import annotations

import math
import numpy as np

from cs.api import vcs
from cs.solvers.solve_vcs import make_vcs_fun
from cs.problem import CSProblem
from cs.options import WOptions, PGSolverOptions


def _mk_prob(A: np.ndarray, T: float, *, steps: int | None = None) -> CSProblem:
    A = np.asarray(A, dtype=float)
    wopt = WOptions.from_system(A, T)
    if steps is not None:
        wopt = wopt.with_steps(int(steps))
    return CSProblem(A, T, w_options=wopt)


def _grid_simplex_3(m: int = 40) -> np.ndarray:
    # All (i,j,k)/m with i+j+k=m
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            k = m - i - j
            pts.append([i / m, j / m, k / m])
    return np.asarray(pts, dtype=float)


def test_vcs_solution_is_near_grid_optimum_n3() -> None:
    # Stable, mildly asymmetric A (avoid near-symmetry so p is not exactly uniform)
    A = np.array(
        [
            [-2.0,  0.6,  0.0],
            [ 0.1, -1.3,  0.4],
            [ 0.0,  0.2, -1.6],
        ],
        dtype=float,
    )
    T = math.inf

    # Solve by your API
    sol_opt = PGSolverOptions(max_iter=2000, tol=1e-12)
    p, info, WList, P = vcs(A, T, solver_options=sol_opt)

    # Build the same CSProblem and objective for evaluation
    prob = _mk_prob(A, T)
    fun = make_vcs_fun(prob)

    f_sol, _ = fun(p)
    assert np.isfinite(f_sol)

    # Grid search reference (coarse)
    grid = _grid_simplex_3(m=50)
    f_best = math.inf
    p_best = None
    for pg in grid:
        fg, _ = fun(pg)
        if np.isfinite(fg) and fg < f_best:
            f_best = fg
            p_best = pg

    assert p_best is not None
    # Solver should be at least as good as (or close to) the coarse grid optimum.
    # Allow small slack because grid is coarse.
    assert f_sol <= f_best + 1e-3

def test_uniform_under_full_symmetry() -> None:
    n = 4
    alpha = 2.0
    beta = 0.3
    J = np.ones((n, n), dtype=float)
    A = -alpha * np.eye(n) + beta * (J - np.eye(n))  # stable & permutation-symmetric

    p, info, _, _ = vcs(A, math.inf, solver_options=PGSolverOptions(max_iter=2000, tol=1e-12))

    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert abs(float(np.sum(p)) - 1.0) < 1e-8
    # "そこまで大きく変わらない"前提に合わせて、かなり緩い閾値
    assert (np.max(p) - np.min(p)) <= 0.02

def test_permutation_equivariance() -> None:
    A = np.array(
        [
            [-3.0,  0.4,  0.1,  0.0],
            [ 0.2, -2.6,  0.2,  0.1],
            [ 0.1,  0.2, -2.8,  0.3],
            [ 0.0,  0.1,  0.3, -2.9],
        ],
        dtype=float,
    )

    opt = PGSolverOptions(max_iter=2500, tol=1e-12)
    p, _, _, _ = vcs(A, math.inf, solver_options=opt)

    perm = np.array([2, 1, 0, 3], dtype=int)
    Pm = np.eye(4)[perm]
    A2 = Pm @ A @ Pm.T

    p2, _, _, _ = vcs(A2, math.inf, solver_options=opt)

    assert np.allclose(p2, p[perm], rtol=5e-4, atol=5e-6)

def test_small_perturbation_changes_p_only_mildly() -> None:
    n = 4
    alpha = 2.0
    beta = 0.25
    J = np.ones((n, n), dtype=float)
    A = -alpha * np.eye(n) + beta * (J - np.eye(n))

    opt = PGSolverOptions(max_iter=2500, tol=1e-12)
    p1, _, _, _ = vcs(A, math.inf, solver_options=opt)

    # Small perturbation
    eps = 1e-2
    E = np.zeros_like(A)
    E[1, 0] = eps
    E[2, 0] = -eps / 2
    A2 = A + E

    p2, _, _, _ = vcs(A2, math.inf, solver_options=opt)

    # L1 distance should be small (loose threshold)
    assert float(np.sum(np.abs(p2 - p1))) <= 0.10

def test_wlist_satisfies_lyapunov_residual_infinite_horizon() -> None:
    A = np.array(
        [
            [-2.0, 0.2, 0.0],
            [0.1, -1.5, 0.3],
            [0.0, 0.2, -1.8],
        ],
        dtype=float,
    )

    # Force lyap method via WOptions (if your API supports passing it)
    wopt = WOptions.from_system(A, math.inf).with_method("lyap")
    p, info, WList, _ = vcs(A, math.inf, w_options=wopt)

    n = A.shape[0]
    for i in range(n):
        Wi = WList[i]
        ei = np.zeros((n, 1), dtype=float)
        ei[i, 0] = 1.0
        Q = ei @ ei.T
        R = A @ Wi + Wi @ A.T + Q
        assert np.linalg.norm(R, ord="fro") < 1e-8
