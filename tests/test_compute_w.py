# tests/test_compute_w.py
from __future__ import annotations

import math
from typing import  Dict, List

import numpy as np
import scipy.linalg as sla

from controllability_scoring.gramian.compute_w import compute_w
from controllability_scoring.options import WOptions


def _numeric_finite_gramian(A: np.ndarray, Q: np.ndarray, T: float, steps: int = 20000) -> np.ndarray:
    """
    Numerical reference for finite-horizon Gramian using trapezoidal integration:
        W(T) = ∫_0^T exp(A t) Q exp(A^T t) dt
    This is used only for testing.
    """
    n = A.shape[0]
    dt = T / steps
    W = np.zeros((n, n), dtype=float)

    for k in range(steps + 1):
        t = k * dt
        E = sla.expm(A * t)
        integrand = E @ Q @ E.T
        weight = 0.5 if (k == 0 or k == steps) else 1.0
        W += weight * integrand

    W *= dt
    return 0.5 * (W + W.T)


def _assert_structure(
    WList: List[Dict[str, List[np.ndarray]]],
    block_sizes: np.ndarray,
    vcs_blocks: np.ndarray,
    aecs_blocks: np.ndarray,
    aecs_matrix: List[np.ndarray],
    P: np.ndarray,
    n: int,
) -> None:
    assert isinstance(WList, list)
    assert len(WList) == n

    assert np.array_equal(block_sizes, np.array([n], dtype=np.int64))
    assert np.array_equal(vcs_blocks, np.array([0], dtype=np.int64))
    assert np.array_equal(aecs_blocks, np.array([0], dtype=np.int64))

    assert P.shape == (n, n)
    assert np.allclose(P, np.eye(n))

    assert isinstance(aecs_matrix, list)
    assert len(aecs_matrix) == 1      # 今の compute_w は単一ブロック想定
    assert aecs_matrix[0].shape == (n, n)
    assert np.allclose(aecs_matrix[0], np.eye(n))

    for i in range(n):
        assert isinstance(WList[i], list)
        assert len(WList[i]) == 1
        Wi = WList[i][0]
        assert isinstance(Wi, np.ndarray)
        assert Wi.shape == (n, n)
        # Symmetry check
        assert np.allclose(Wi, Wi.T, atol=1e-10, rtol=1e-10)


def test_compute_w_finite_horizon_matches_numeric_integration() -> None:
    # Stable A (not required for finite horizon, but makes values well-behaved)
    A = np.array([[-1.0, 0.2],
                  [0.0, -0.5]], dtype=float)
    T = 1.0
    wopt = WOptions.from_system(A,T)

    WList = compute_w(A, T, wopt)
    n = A.shape[0]
    _assert_structure(WList.w_list, WList.block_sizes, WList.vcs_blocks, WList.aecs_blocks, WList.aecs_matrix, WList.transform_matrix, n)

    # Compare each Wi against a numerical reference integral
    for i in range(n):
        B = np.zeros((n, 1), dtype=float)
        B[i, 0] = 1.0
        Q = B @ B.T

        W_ref = _numeric_finite_gramian(A, Q, T, steps=20000)
        Wi = WList.w_list[i][0]

        # Relative/absolute tolerances chosen to be robust across platforms
        assert np.allclose(Wi, W_ref, rtol=5e-6, atol=5e-8)


def test_compute_w_infinite_horizon_solves_lyapunov_residual() -> None:
    # Must be Hurwitz (stable) for infinite-horizon Gramian to exist
    A = np.array([[-1.2, 0.3],
                  [-0.1, -0.7]], dtype=float)
    T = math.inf
    wopt = WOptions.from_system(A,T)

    WList = compute_w(A, T, wopt)
    n = A.shape[0]
    _assert_structure(WList.w_list, WList.block_sizes,WList.vcs_blocks, WList.aecs_blocks, WList.aecs_matrix, WList.transform_matrix, n)

    # Check Lyapunov residual: A W + W A^T + Q ≈ 0 for each i (Q = e_i e_i^T)
    for i in range(n):
        B = np.zeros((n, 1), dtype=float)
        B[i, 0] = 1.0
        Q = B @ B.T

        Wi = WList.w_list[i][0]
        residual = A @ Wi + Wi @ A.T + Q
        assert np.linalg.norm(residual, ord="fro") < 1e-9


def test_compute_w_rejects_negative_T() -> None:
    A = np.array([[-1.0, 0.0],
                  [0.0, -2.0]], dtype=float)
    T = -1
    try:
        wopt = WOptions.from_system(A,T)
        compute_w(A, T, wopt)
        assert False, "Expected ValueError for negative T"
    except ValueError:
        pass


def test_compute_w_rejects_nonsquare_A() -> None:
    A = np.zeros((3, 2), dtype=float)
    T = 1.0

    try:
        wopt = WOptions.from_system(A,T)
        compute_w(A, T, wopt)
        assert False, "Expected ValueError for non-square A"
    except ValueError:
        pass
