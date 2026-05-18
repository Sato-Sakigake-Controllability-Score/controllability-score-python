# tests/test_gramian.py
#
# Purpose:
#   - Stable / unstable systems
#   - Short-time / long-time
#   - Infinite horizon
#   - Noscale / scale variants
#
# Strategy:
#   (1) Finite horizon: verify Sylvester identity
#         A W + W A^T = -(e^{TA} E_i e^{TA^T} - E_i)
#       (valid even if A is unstable)
#
#   (2) Infinite horizon (stable only): verify Lyapunov identity
#         A W + W A^T = -E_i
#
#   (3) Compare against analytic solution for diagonal A
#
#   (4) Compare fin_integral_noscale vs fin_lyap_noscale
#
#   (5) For scale version: unscale and back-transform to original
#       coordinates and compare with noscale result.
#
# Usage:
#   pytest -q
#
# NOTE:
#   Adjust import paths to match your project structure.

import numpy as np
import pytest
from scipy.linalg import expm, block_diag


from cs.gramian.gramian import (
    fin_lyap_noscale,
    fin_lyap_scale,
    inf_lyap_noscale,
)
from cs.gramian.block_diagonalization import block_diagonalization
# =============================================================


# ---- Minimal wopts stub (absorbs attribute name inconsistencies) ----
class WOpts:
    def __init__(self, steps=400, method="lyap", eigtol=1e-10):
        self.Steps = steps
        self.steps = steps
        self.Method = method
        self.method = method
        self.eigtol = eigtol
        self.EigTol = eigtol


def _Ei(n: int, i: int, dtype=float):
    E = np.zeros((n, n), dtype=dtype)
    E[i, i] = 1
    return E


def _assert_symmetric_psd(W, psd_tol=1e-10):
    # Check symmetry
    np.testing.assert_allclose(W, W.T.conj(), atol=1e-10, rtol=1e-10)

    # Check positive semidefiniteness (allow small numerical error)
    ev = np.linalg.eigvalsh((W + W.T.conj()) * 0.5)
    assert np.min(ev) >= -psd_tol


def _assert_finite_time_identity(A, T, Wi, i, atol=5e-8, rtol=5e-7):
    """
    Finite-horizon identity:
        A W + W A^T = -(e^{TA} E_i e^{TA^T} - E_i)
    """
    n = A.shape[0]
    Ei = _Ei(n, i, dtype=Wi.dtype)
    eAT = expm(T * A)
    rhs = eAT @ Ei @ eAT.T.conj() - Ei
    lhs = A @ Wi + Wi @ A.T.conj()
    np.testing.assert_allclose(lhs, rhs, atol=atol, rtol=rtol)


def _assert_infinite_time_identity(A, Wi, i, atol=5e-8, rtol=5e-7):
    """
    Infinite-horizon identity (stable case):
        A W + W A^T = -E_i
    """
    n = A.shape[0]
    Ei = _Ei(n, i, dtype=Wi.dtype)
    lhs = A @ Wi + Wi @ A.T.conj()
    rhs = -Ei
    np.testing.assert_allclose(lhs, rhs, atol=atol, rtol=rtol)


def _make_random_stable_A(rng, n, shift=1.0, scale=0.1):
    M = rng.standard_normal((n, n))
    return -shift * np.eye(n) + scale * M


def _make_random_unstable_A(rng, n, shift=1.0, scale=0.1):
    M = rng.standard_normal((n, n))
    return shift * np.eye(n) + scale * M


# ===== Scale utilities =====
def _rebuild_DinvFull_and_Q(A, T, wopts):
    # Recompute the same DinvFull used inside fin_lyap_scale
    blocks, block_sizes, _, Q, Qinv = block_diagonalization(A, wopts)
    nS, nI, nU = map(int, block_sizes)

    emAUT = expm(-T * blocks[2]) if nU > 0 else np.eye(0, dtype=A.dtype)
    sqrtT = np.sqrt(T)

    DinvFull = block_diag(
        np.eye(nS, dtype=A.dtype),
        (np.eye(nI, dtype=A.dtype) / sqrtT) if nI > 0 else np.eye(0, dtype=A.dtype),
        emAUT,
    )
    return DinvFull, Q


def _unscale_and_backtransform(Wi_scaled, DinvFull, Q):
    # DinvFull = D^{-1}
    D = np.linalg.inv(DinvFull)
    W_block = D @ Wi_scaled @ D.T.conj()
    W_orig = Q @ W_block @ Q.T.conj()
    return W_orig


# =========================================================
# 1) Basic structural checks (finite horizon)
# =========================================================
@pytest.mark.parametrize("T", [1e-3, 0.2, 5.0])
def test_finite_time_outputs_are_symmetric_psd_for_stable(T):
    rng = np.random.default_rng(10)
    n = 5
    A = _make_random_stable_A(rng, n)
    wopts = WOpts()

    WL = fin_lyap_noscale(A, T, wopts)
    for i in range(n):
        _assert_symmetric_psd(WL.w_list[i][0], psd_tol=1e-8)


@pytest.mark.parametrize("T", [1e-3, 0.2, 5.0])
def test_finite_time_outputs_are_symmetric_psd_for_unstable(T):
    rng = np.random.default_rng(11)
    n = 5
    A = _make_random_unstable_A(rng, n)
    wopts = WOpts()

    WL = fin_lyap_noscale(A, T, wopts)
    for i in range(n):
        _assert_symmetric_psd(WL.w_list[i][0], psd_tol=1e-8)


# =========================================================
# 2) Analytic diagonal case
# =========================================================
@pytest.mark.parametrize("T", [1e-3, 0.5, 10.0])
def test_finite_time_matches_diagonal_analytic(T):
    lam = np.array([-1.0, -2.0, -0.1], dtype=float)
    A = np.diag(lam)
    n = A.shape[0]
    wopts = WOpts()

    WL = fin_lyap_noscale(A, T, wopts)

    for i in range(n):
        Wi = WL.w_list[i][0]
        expected = np.zeros((n, n), dtype=float)
        expected[i, i] = (np.exp(2.0 * lam[i] * T) - 1.0) / (2.0 * lam[i])
        np.testing.assert_allclose(Wi, expected, atol=1e-10, rtol=1e-9)


def test_infinite_time_matches_diagonal_analytic():
    lam = np.array([-1.0, -2.0, -0.1], dtype=float)
    A = np.diag(lam)
    n = A.shape[0]
    wopts = WOpts(eigtol=1e-12)

    WL = inf_lyap_noscale(A, wopts)

    for i in range(n):
        Wi = WL.w_list[i][0]
        expected = np.zeros((n, n), dtype=float)
        expected[i, i] = -1.0 / (2.0 * lam[i])
        np.testing.assert_allclose(Wi, expected, atol=1e-10, rtol=1e-9)


# =========================================================
# 3) Finite-time identity (stable and unstable)
# =========================================================
@pytest.mark.parametrize("T", [1e-3, 0.2, 5.0])
def test_finite_time_identity_holds_random_stable(T):
    rng = np.random.default_rng(20)
    n = 6
    A = _make_random_stable_A(rng, n)
    wopts = WOpts()

    WL = fin_lyap_noscale(A, T, wopts)
    for i in range(n):
        _assert_finite_time_identity(A, T, WL.w_list[i][0], i)


@pytest.mark.parametrize("T", [1e-3, 0.2, 5.0])
def test_finite_time_identity_holds_random_unstable(T):
    rng = np.random.default_rng(21)
    n = 6
    A = _make_random_unstable_A(rng, n)
    wopts = WOpts()

    WL = fin_lyap_noscale(A, T, wopts)
    for i in range(n):
        _assert_finite_time_identity(A, T, WL.w_list[i][0], i)


# =========================================================
# 4) Infinite horizon behavior
# =========================================================
def test_infinite_time_raises_on_unstable():
    A = np.diag([0.1, -1.0, -2.0])
    wopts = WOpts(eigtol=1e-12)
    with pytest.raises(ValueError):
        inf_lyap_noscale(A, wopts)


def test_infinite_time_identity_holds_random_stable():
    rng = np.random.default_rng(40)
    n = 6
    A = _make_random_stable_A(rng, n, shift=1.5)
    wopts = WOpts(eigtol=1e-10)

    WL = inf_lyap_noscale(A, wopts)
    for i in range(n):
        Wi = WL.w_list[i][0]
        _assert_infinite_time_identity(A, Wi, i)
        _assert_symmetric_psd(Wi, psd_tol=1e-8)


# =========================================================
# 5) Scale vs noscale consistency (finite horizon)
# =========================================================
@pytest.mark.parametrize("T", [0.2, 2.0])
def test_finite_scale_matches_noscale_after_unscale(T):
    rng = np.random.default_rng(50)
    n = 5
    A = _make_random_stable_A(rng, n)
    wopts = WOpts()

    WL0 = fin_lyap_noscale(A, T, wopts)
    WLs = fin_lyap_scale(A, T, wopts)

    DinvFull, Q = _rebuild_DinvFull_and_Q(A, T, wopts)

    for i in range(n):
        Wi_back = _unscale_and_backtransform(
            WLs.w_list[i][0], DinvFull, Q
        )
        np.testing.assert_allclose(Wi_back, WL0.w_list[i][0], atol=1e-7, rtol=1e-6)


# =========================================================
# 6) Scale version: Sa consistency
# =========================================================
def test_finite_scale_Sa_is_symmetric_psd_and_consistent():
    rng = np.random.default_rng(60)
    n = 5
    T = 1.0
    A = _make_random_stable_A(rng, n)
    wopts = WOpts()

    WLs = fin_lyap_scale(A, T, wopts)
    Sa = WLs.aecs_matrix[0]

    _assert_symmetric_psd(Sa, psd_tol=1e-8)

    blocks, block_sizes, _, Q, Qinv = block_diagonalization(A, wopts)
    DinvFull, _ = _rebuild_DinvFull_and_Q(A, T, wopts)

    Sa_re = DinvFull @ (Qinv @ Qinv.T.conj()) @ DinvFull.T.conj()
    np.testing.assert_allclose(Sa, Sa_re, atol=1e-8, rtol=1e-7)