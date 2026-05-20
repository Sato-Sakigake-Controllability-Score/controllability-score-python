# tests/solvers/test_aecs_symmetry_qualitative.py
from __future__ import annotations

import math
import numpy as np

from controllability_scoring.problem import CSProblem
from controllability_scoring.solvers.solve_aecs import solve_aecs


def _mk_problem(A: np.ndarray, *, T: float = math.inf) -> CSProblem:
    return CSProblem(A=np.asarray(A, dtype=float), T=T)


def _solve_p(prob: CSProblem) -> np.ndarray:
    p, _info = solve_aecs(prob)
    return np.asarray(p, dtype=float).reshape(-1)


def test_p_is_uniform_under_permutation_symmetry() -> None:
    """
    If all states are dynamically symmetric, the optimizer should return uniform p.
    This should hold for AECS too (objective constructed from blocks should be permutation-invariant).
    """
    n = 4
    alpha = 2.0
    beta = 0.3
    J = np.ones((n, n), dtype=float)

    A = -alpha * np.eye(n) + beta * (J - np.eye(n))

    prob = _mk_problem(A, T=math.inf)
    p = _solve_p(prob)

    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert np.all(p >= -1e-12)
    assert abs(np.sum(p) - 1.0) < 1e-8

    # All entries should be equal (within tolerance)
    assert np.max(p) - np.min(p) < 5e-3


def test_mild_asymmetry_causes_only_mild_bias() -> None:
    """
    With a small asymmetry, p should tilt toward the favored state,
    but not by a huge margin.
    """
    n = 4
    alpha = 2.0
    beta = 0.3
    J = np.ones((n, n), dtype=float)
    A = -alpha * np.eye(n) + beta * (J - np.eye(n))

    eps = 0.05
    A = A.copy()
    A[1, 0] += eps
    A[2, 0] += eps
    A[3, 0] += eps

    p = _solve_p(_mk_problem(A))
    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert abs(np.sum(p) - 1.0) < 1e-8

    # Favored index should not become smaller than all others
    assert p[0] >= np.min(p[1:])

    # Bias should stay mild (very loose)
    assert float(np.max(p) / np.min(p)) <= 1.5


def test_permutation_equivariance_of_p() -> None:
    """
    If we permute state labels (A -> P A P^T), the solution p should permute the same way.
    """
    A = np.array(
        [
            [-3.0,  0.4,  0.1,  0.0],
            [ 0.2, -2.6,  0.2,  0.1],
            [ 0.1,  0.2, -2.8,  0.3],
            [ 0.0,  0.1,  0.3, -2.9],
        ],
        dtype=float,
    )

    p = _solve_p(_mk_problem(A))

    perm = np.array([2, 1, 0, 3], dtype=int)
    Pm = np.eye(4)[perm]
    A2 = Pm @ A @ Pm.T

    p2 = _solve_p(_mk_problem(A2))

    assert np.allclose(p2, p[perm], rtol=1e-4, atol=1e-6)
