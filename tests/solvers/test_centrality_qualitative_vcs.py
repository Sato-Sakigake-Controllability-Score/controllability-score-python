# tests/solvers/test_vcs_centrality_qualitative.py
from __future__ import annotations

import math
import numpy as np

from cs.problem import CSProblem
from cs.solvers.solve_vcs import solve_vcs


def _mk_problem(A: np.ndarray, *, T: float = math.inf) -> CSProblem:
    """
    Build a minimal CSProblem for VCS tests.
    Adjust fields if your CSProblem constructor differs.
    """
    return CSProblem(
        A=np.asarray(A, dtype=float),
        T=T
    )


def _solve_p(prob: CSProblem) -> np.ndarray:
    # Use stricter but stable solver settings for tests

    p, _info = solve_vcs(prob)
    p = np.asarray(p, dtype=float).reshape(-1)
    return p


def test_p_is_uniform_under_permutation_symmetry() -> None:
    """
    If all states are dynamically symmetric, the optimizer should return uniform p.
    """
    n = 4
    alpha = 2.0
    beta = 0.3
    J = np.ones((n, n), dtype=float)

    # Stable, fully symmetric: all nodes are equivalent
    A = -alpha * np.eye(n) + beta * (J - np.eye(n))

    prob = _mk_problem(A, T=math.inf)
    p = _solve_p(prob)

    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert np.all(p >= -1e-12)  # numerical slack
    assert abs(np.sum(p) - 1.0) < 1e-8

    # All entries should be equal (within tolerance)
    assert np.max(p) - np.min(p) < 5e-3



def test_mild_asymmetry_causes_only_mild_bias() -> None:
    """
    With a small asymmetry, p should tilt toward the favored state,
    but not by a huge margin (known to not change drastically).
    """
    # Start from symmetric stable A, then add a small perturbation that favors state 0
    n = 4
    alpha = 2.0
    beta = 0.3
    J = np.ones((n, n), dtype=float)
    A = -alpha * np.eye(n) + beta * (J - np.eye(n))

    # Mild asymmetry: slightly increase coupling from state 0 to others
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

    # But bias should stay mild: e.g., max/min <= 1.5 (very loose)
    assert float(np.max(p) / np.min(p)) <= 1.5


def test_permutation_equivariance_of_p() -> None:
    """
    If we permute the state labels (A -> P A P^T), the solution p should permute the same way.
    This is a qualitative structural property independent of magnitudes.
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

    # Permute states: swap 0 and 2
    perm = np.array([2, 1, 0, 3], dtype=int)
    Pm = np.eye(4)[perm]  # permutation matrix
    A2 = Pm @ A @ Pm.T

    p2 = _solve_p(_mk_problem(A2))

    # p2 should be p permuted in the same way (up to solver tolerance)
    assert np.allclose(p2, p[perm], rtol=1e-4, atol=1e-6)