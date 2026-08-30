# tests/solvers/test_vcs_graph_bias.py
from __future__ import annotations

import math
import numpy as np

from controllability_scoring.problem import CSProblem
from controllability_scoring.solvers.solve_vcs import solve_vcs


def _laplacian_from_weight(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    d = np.sum(W, axis=1)
    return np.diag(d) - W


def _stable_A_from_graph(W: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Build a Hurwitz matrix from an undirected weighted graph:
      A = -(L + alpha I)
    """
    n = W.shape[0]
    L = _laplacian_from_weight(W)
    return -(L + alpha * np.eye(n))


def _mk_problem(A: np.ndarray, *, T: float = math.inf) -> CSProblem:
    return CSProblem(
        A=np.asarray(A, dtype=float),
        T=T
    )


def _solve_p(prob: CSProblem) -> np.ndarray:
    p, _info = solve_vcs(prob)
    return np.asarray(p, dtype=float).reshape(-1)


def _range_ratio(p: np.ndarray) -> float:
    m = float(np.mean(p))
    return float((np.max(p) - np.min(p)) / (m if m != 0 else 1.0))


def test_star_graph_center_is_largest_and_leaves_are_equalish() -> None:
    """
    Star graph: center node is structurally special, leaves are symmetric.
    Expect:
      - p_center is the largest
      - leaves are (almost) equal
      - overall non-uniformity exists but is not huge
    """
    n = 6
    center = 0

    # Weighted star adjacency
    W = np.zeros((n, n), dtype=float)
    for j in range(1, n):
        W[center, j] = 1.0
        W[j, center] = 1.0

    A = _stable_A_from_graph(W, alpha=1.0)
    p = _solve_p(_mk_problem(A, T=math.inf))

    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert abs(np.sum(p) - 1.0) < 1e-8

    # Center should be largest (qualitative)
    assert p[center] + 1e-12 >= np.max(p[1:])

    # Leaves are symmetric -> nearly equal
    leaves = p[1:]
    assert float(np.max(leaves) - np.min(leaves)) <= 5e-3

    # Bias exists but should not be extreme (very loose)
    assert _range_ratio(p) <= 0.6


def test_two_cliques_size_bias_and_within_clique_symmetry() -> None:
    """
    Two disconnected cliques of different sizes.
    Because dynamics decouple, symmetry implies:
      - nodes within the same clique should have (almost) equal p
      - larger clique tends to receive slightly more total weight (mild bias)
    """
    n1, n2 = 5, 3
    n = n1 + n2

    # Two cliques adjacency (no inter edges)
    W = np.zeros((n, n), dtype=float)

    # Clique 1
    for i in range(n1):
        for j in range(n1):
            if i != j:
                W[i, j] = 1.0

    # Clique 2
    offset = n1
    for i in range(offset, n):
        for j in range(offset, n):
            if i != j:
                W[i, j] = 1.0

    A = _stable_A_from_graph(W, alpha=1.0)
    p = _solve_p(_mk_problem(A, T=math.inf))

    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert abs(np.sum(p) - 1.0) < 1e-8

    p1 = p[:n1]
    p2 = p[n1:]

    # Within-clique symmetry
    assert float(np.max(p1) - np.min(p1)) <= 5e-3
    assert float(np.max(p2) - np.min(p2)) <= 5e-3

    # Mild size bias: total weight of larger clique should not be smaller
    # (Do not demand a big margin)
    assert float(np.sum(p1)) >= float(np.sum(p2))


def test_ring_graph_is_nearly_uniform() -> None:
    """
    Ring/cycle graph is vertex-transitive -> p should be near-uniform.
    """
    n = 8
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        W[i, j] = 1.0
        W[j, i] = 1.0

    A = _stable_A_from_graph(W, alpha=1.0)
    p = _solve_p(_mk_problem(A, T=math.inf))

    assert p.shape == (n,)
    assert np.all(np.isfinite(p))
    assert abs(np.sum(p) - 1.0) < 1e-8

    # Very mild non-uniformity threshold
    assert _range_ratio(p) <= 0.12
