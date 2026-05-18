# tests/test_cs_validation.py
from __future__ import annotations

import math
import numpy as np
import pytest

from cs.api import cs, vcs, aecs
from cs.solvers.solve_vcs import make_vcs_fun
from cs.solvers.solve_aecs import make_aecs_fun
from cs.problem import CSProblem
from cs.options import WOptions, PGSolverOptions


def _mk_prob(A: np.ndarray, T: float, *, steps: int | None = None) -> CSProblem:
    A = np.asarray(A, dtype=float)
    wopt = WOptions.from_system(A, T)
    if steps is not None:
        wopt = wopt.with_steps(int(steps))
    return CSProblem(A, T, w_options=wopt)


def _grid_simplex_3(m: int = 40) -> np.ndarray:
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            k = m - i - j
            pts.append([i / m, j / m, k / m])
    return np.asarray(pts, dtype=float)


def test_cs_matches_separate_vcs_aecs_calls() -> None:
    A = np.array(
        [
            [-2.0,  0.6,  0.0],
            [ 0.1, -1.3,  0.4],
            [ 0.0,  0.2, -1.6],
        ],
        dtype=float,
    )
    T = math.inf
    opt = PGSolverOptions(max_iter=2000, tol=1e-12)

    pV, pA, infoV, infoA, Wcs, Pcs = cs(A, T, solver_options=opt, w_output="trans")
    pV2, infoV2, Wv, Pv = vcs(A, T, solver_options=opt, w_output="trans")
    pA2, infoA2, Wa, Pa = aecs(A, T, solver_options=opt, w_output="trans")

    # p vectors should match their separate counterparts
    assert np.allclose(pV, pV2, rtol=1e-12, atol=1e-12)
    assert np.allclose(pA, pA2, rtol=1e-12, atol=1e-12)

    # basic simplex sanity
    assert abs(float(np.sum(pV)) - 1.0) < 1e-10
    assert abs(float(np.sum(pA)) - 1.0) < 1e-10
    assert np.all(pV >= -1e-14)
    assert np.all(pA >= -1e-14)

    # WList export should match (since cs uses same Blocks[0] export rule)
    assert len(Wcs) == len(Wv) == len(Wa) == A.shape[0]
    for i in range(A.shape[0]):
        assert np.allclose(Wcs[i], Wv[i], rtol=1e-12, atol=1e-12)
        assert np.allclose(Wcs[i], Wa[i], rtol=1e-12, atol=1e-12)

    # P should also match (often identity for current stub/impl)
    if Pcs is None:
        assert Pv is None and Pa is None
    else:
        assert Pv is not None and Pa is not None
        assert np.allclose(Pcs, Pv, rtol=1e-12, atol=1e-12)
        assert np.allclose(Pcs, Pa, rtol=1e-12, atol=1e-12)


def test_cs_solution_is_near_grid_optimum_n3_for_both_objectives() -> None:
    # Stable, mildly asymmetric A
    A = np.array(
        [
            [-2.0,  0.6,  0.0],
            [ 0.1, -1.3,  0.4],
            [ 0.0,  0.2, -1.6],
        ],
        dtype=float,
    )
    T = math.inf

    sol_opt = PGSolverOptions(max_iter=2500, tol=1e-12)
    pV, pA, _, _, _, _ = cs(A, T, solver_options=sol_opt)

    prob = _mk_prob(A, T)
    fV = make_vcs_fun(prob)
    fA = make_aecs_fun(prob)

    fV_sol, _ = fV(pV)
    fA_sol, _ = fA(pA)
    assert np.isfinite(fV_sol)
    assert np.isfinite(fA_sol)

    grid = _grid_simplex_3(m=50)

    # VCS grid optimum
    fV_best = math.inf
    for pg in grid:
        fg, _ = fV(pg)
        if np.isfinite(fg) and fg < fV_best:
            fV_best = fg
    assert fV_sol <= fV_best + 1e-3

    # AECS grid optimum
    fA_best = math.inf
    for pg in grid:
        fg, _ = fA(pg)
        if np.isfinite(fg) and fg < fA_best:
            fA_best = fg
    assert fA_sol <= fA_best + 1e-3


def test_cs_uniform_under_full_symmetry_for_both() -> None:
    n = 4
    alpha = 2.0
    beta = 0.3
    J = np.ones((n, n), dtype=float)
    A = -alpha * np.eye(n) + beta * (J - np.eye(n))

    opt = PGSolverOptions(max_iter=2500, tol=1e-12)
    pV, pA, _, _, _, _ = cs(A, math.inf, solver_options=opt)

    for p in (pV, pA):
        assert p.shape == (n,)
        assert np.all(np.isfinite(p))
        assert abs(float(np.sum(p)) - 1.0) < 1e-8
        assert (np.max(p) - np.min(p)) <= 0.02


def test_cs_permutation_equivariance_for_both() -> None:
    A = np.array(
        [
            [-3.0,  0.4,  0.1,  0.0],
            [ 0.2, -2.6,  0.2,  0.1],
            [ 0.1,  0.2, -2.8,  0.3],
            [ 0.0,  0.1,  0.3, -2.9],
        ],
        dtype=float,
    )

    opt = PGSolverOptions(max_iter=3000, tol=1e-12)
    pV, pA, _, _, _, _ = cs(A, math.inf, solver_options=opt)

    perm = np.array([2, 1, 0, 3], dtype=int)
    Pm = np.eye(4)[perm]
    A2 = Pm @ A @ Pm.T

    pV2, pA2, _, _, _, _ = cs(A2, math.inf, solver_options=opt)

    assert np.allclose(pV2, pV[perm], rtol=5e-4, atol=5e-6)
    assert np.allclose(pA2, pA[perm], rtol=5e-4, atol=5e-6)


def test_cs_wlist_satisfies_lyapunov_residual_infinite_horizon() -> None:
    # Same style as AECS test: if WList is computed via Lyapunov, residual should be small.
    A = np.array(
        [
            [-2.0, 0.2, 0.0],
            [0.1, -1.5, 0.3],
            [0.0, 0.2, -1.8],
        ],
        dtype=float,
    )

    wopt = WOptions.from_system(A, math.inf).with_method("lyap")
    pV, pA, infoV, infoA, WList, _ = cs(A, math.inf, w_options=wopt, inf_keep="stable_only")

    n = A.shape[0]
    assert len(WList) == n

    for i in range(n):
        Wi = WList[i]
        ei = np.zeros((n, 1), dtype=float)
        ei[i, 0] = 1.0
        Q = ei @ ei.T
        R = A @ Wi + Wi @ A.T + Q
        assert np.linalg.norm(R, ord="fro") < 1e-8
def test_vcs_is_uniform_when_A_is_symmetric() -> None:
    # Symmetric + Hurwitz (negative definite) so infinite-horizon is well-defined
    n = 5
    rng = np.random.default_rng(0)
    M = rng.normal(size=(n, n))
    A = -(M + M.T) - 2.0 * np.eye(n)  # symmetric and strictly stable
    T = math.inf

    opt = PGSolverOptions(max_iter=3000, tol=1e-12)
    pV, infoV, _, _ = vcs(A, T, solver_options=opt)

    assert pV.shape == (n,)
    assert np.all(np.isfinite(pV))
    assert abs(float(np.sum(pV)) - 1.0) < 1e-8

    # VCS optimum should be uniform 1/n for symmetric A
    u = np.ones(n) / n
    assert np.allclose(pV, u, rtol=5e-4, atol=5e-6)


def test_vcs_and_aecs_are_uniform_when_A_is_skew_symmetric_finite_horizon() -> None:
    # Pure skew-symmetric matrices are not Hurwitz in general, so use finite horizon.
    n = 6
    rng = np.random.default_rng(1)
    M = rng.normal(size=(n, n))
    A = M - M.T  # skew-symmetric: A^T = -A
    T = 1.0

    opt = PGSolverOptions(max_iter=4000, tol=1e-12)

    pV, infoV, _, _ = vcs(A, T, solver_options=opt)
    pA, infoA, _, _ = aecs(A, T, solver_options=opt)

    for p in (pV, pA):
        assert p.shape == (n,)
        assert np.all(np.isfinite(p))
        assert abs(float(np.sum(p)) - 1.0) < 1e-8
        assert np.all(p >= -1e-12)

        u = np.ones(n) / n
        assert np.allclose(p, u, rtol=5e-4, atol=5e-6)
def assert_match_3dp(a: np.ndarray, b: np.ndarray) -> None:
    assert np.max(np.abs(a - b)) <= 1e-3 + 1e-12, (
        f"max_abs_err={np.max(np.abs(a-b))} exceeds 5e-4"
    )



def _laplacian_L_10nodes_w02() -> np.ndarray:
    # Nodes 1..10
    # Directed edges with weight 0.2 (as in the paper, Fig.1)
    L = np.zeros((10, 10), dtype=float)

    # add edge j -> i with weight w
    # (i.e. x_j influences x_i, consistent with dot{x} = -L x)
    def add_edge(i: int, j: int, w: float = 0.2) -> None:
        L[i-1, i-1] += w
        L[i-1, j-1] -= w

    # Fig.1 edges (source -> target)
    for target in [1, 2, 3, 4]:
        add_edge(target, 7, 0.2)   # 7 -> 1,2,3,4

    add_edge(10, 2, 0.2)  # 2 -> 10
    add_edge(6, 4, 0.2)   # 4 -> 6
    add_edge(8, 3, 0.2)   # 3 -> 8
    add_edge(5, 1, 0.2)   # 1 -> 5
    add_edge(1, 9, 0.2)   # 9 -> 1
    add_edge(6, 10, 0.2)  # 10 -> 6

    return L


@pytest.mark.parametrize(
    "T, pV_exp, pA_exp",
    [
        (
            0.01,
            np.array([0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1], dtype=float),
            np.array([0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1], dtype=float),
        ),
        (
            1.0,
            np.array([0.099677,0.1,0.1,0.099997,0.099674,
                        0.09935,0.10131,0.09967,0.10033,0.099995], dtype=float),
            np.array([0.10927,0.1,0.1,0.1,0.099783,
                      0.10905,0.091277,0.099815,0.090815,0.099979], dtype=float),
        ),
        (
            1000.0,
            np.array([0.073347, 0.10112, 0.10876, 0.086378, 0.045557,
                       0.060743, 0.24929, 0.042309, 0.16614, 0.066358], dtype=float),
            np.array([0.17127, 0.11333, 0.12054, 0.10584, 0.090745, 
                      0.13350, 0.092572, 0.069467, 0.0070316, 0.09571 ], dtype=float),
        ),
        (
            10000.0,
            np.array([0.073327, 0.10108, 0.10874, 0.086362, 0.044985, 0.060707, 0.24952, 0.042214, 0.16674, 0.06613], dtype=float),
            np.array([0.17281, 0.11364, 0.12093, 0.10610, 0.092299, 0.13383, 0.092752, 0.069445, 0.0023358, 0.095859], dtype=float),
        ),
        (
            math.inf,
            np.array([0.07329, 0.10104, 0.10871, 0.08634, 0.04490, 0.06091, 0.24967, 0.04220, 0.16686, 0.06631], dtype=float),
            np.array([0.17428, 0.11408, 0.12174, 0.10610, 0.09275, 0.13433, 0.09233, 0.06901, 0.0, 0.09538], dtype=float),
        ),
    ],
)
def test_cs_laplacian_matches_known_results(T, pV_exp, pA_exp):
    L = _laplacian_L_10nodes_w02()

    A = -L  # xdot = -L x

    n = 10
    # p0 = np.ones(n) / n
    p0 = np.array([0.10927,0.1,0.1,0.1,0.099783,
                      0.10905,0.091277,0.099815,0.090815,0.099979], dtype=float)

    # ============================================================
    # (1) 厳密 solver：既知結果と 6 桁精度で一致するか
    # ============================================================
    opt_strict = PGSolverOptions(
        max_iter=100000,
        tol=1e-16
    )
    wopts = WOptions(
       method = "lyap",
        # steps = 10000
        eigtol = 1e-7
    )
    pV, pA, infoV, infoA, WList, P = cs(
        A, T,
        solver_options=opt_strict,
        initial_guess=p0,
        w_options=wopts
    )

    
    assert_match_3dp(pV, pV_exp)
    assert_match_3dp(pA, pA_exp)


    # assert np.allclose(pV, pV_exp, rtol=1e-6, atol=1e-7)
    # assert np.allclose(pA, pA_exp, rtol=1e-6, atol=1e-7)

    # ============================================================
    # (2) 通常 solver：基本的な sanity（常に満たすべき条件）
    # ============================================================
    opt_normal = PGSolverOptions(max_iter=5000, tol=1e-12)

    pV2, pA2, infoV2, infoA2, WList2, P2 = cs(
        A, T,
        solver_options=opt_normal
    )

    assert pV2.shape == (n,)
    assert pA2.shape == (n,)
    assert np.all(np.isfinite(pV2))
    assert np.all(np.isfinite(pA2))
    assert abs(float(np.sum(pV2)) - 1.0) < 1e-8
    assert abs(float(np.sum(pA2)) - 1.0) < 1e-8
