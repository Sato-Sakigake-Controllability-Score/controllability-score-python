# tests/solvers/test_projected_gradient_numeric.py
import numpy as np
from dataclasses import replace

from controllability_scoring.pg_solvers.project_gradient import ProjectedGradientSolver
from controllability_scoring.options import PGSolverOptions


def project_onto_simplex(p: np.ndarray) -> np.ndarray:
    """Euclidean projection onto simplex {x>=0, sum x = 1}."""
    p = np.asarray(p, dtype=float).reshape(-1)
    n = p.size
    if n == 0:
        raise ValueError("Input must be a nonempty vector.")
    u = np.sort(p)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, n + 1)
    t = (cssv - 1.0) / j
    idx = np.where(u - t > 0)[0]
    if idx.size == 0:
        return np.ones(n) / n
    theta = t[idx[-1]]
    x = np.maximum(p - theta, 0.0)
    s = float(x.sum())
    return x / s if s > 0 else np.ones(n) / n


def sample_simplex(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    # Dirichlet(1,...,1) は simplex 上のサンプル（実装が短く、参照用に便利）
    return rng.dirichlet(np.ones(n), size=m)


def test_quadratic_over_simplex_matches_known_solution():
    """
    min 0.5||p-q||^2 s.t. p in simplex
    の真の解は proj_simplex(q)。
    solver結果がこれに近いことを検証。
    """
    rng = np.random.default_rng(0)
    n = 12
    q = rng.normal(size=n)

    p_star = project_onto_simplex(q)

    def fun(p):
        r = p - q
        f = 0.5 * float(r @ r)
        g = r
        return f, g

    opt = replace(
        PGSolverOptions(),
        max_iter=800,
        tol=1e-12,
        step_size=1.0,
        rho=0.5,
        sigma=1e-4,
        step_size_inf=1e-16,
        store_trace=False,
        verbose=False,
    )
    solver = ProjectedGradientSolver(opt)

    p0 = sample_simplex(n, 1, rng=rng).reshape(-1)
    p, info = solver.solve(fun, project_onto_simplex, p0)

    # 可行性（simplex）
    assert np.all(p >= -1e-12)
    assert np.isclose(p.sum(), 1.0, atol=1e-10)

    # 解析解との一致
    assert np.linalg.norm(p - p_star) <= 1e-6


def test_quadratic_over_simplex_beats_random_reference():
    """
    数値解としての検証：
    simplex 上のランダム点より solver の目的値が十分良いことを確認。
    """
    rng = np.random.default_rng(1)
    n = 10
    q = rng.normal(size=n)

    def fun(p):
        r = p - q
        f = 0.5 * float(r @ r)
        g = r
        return f, g

    opt = replace(
        PGSolverOptions(),
        max_iter=600,
        tol=1e-12,
        step_size=1.0,
        rho=0.5,
        sigma=1e-4,
        step_size_inf=1e-16,
        store_trace=False,
        verbose=False,
    )
    solver = ProjectedGradientSolver(opt)

    p0 = sample_simplex(n, 1, rng=rng).reshape(-1)
    p, info = solver.solve(fun, project_onto_simplex, p0)

    f_solver = float(info.objective_value)

    # 参照：ランダムサンプルでの最小値（重すぎない範囲）
    samples = sample_simplex(n, 10000, rng=rng)
    f_samples = 0.5 * np.sum((samples - q) ** 2, axis=1)
    f_ref = float(f_samples.min())

    assert f_solver <= f_ref + 1e-8


def test_linear_over_simplex_is_vertex_solution():
    """
    min c^T p s.t. p in simplex の解は argmin(c) への one-hot。
    """
    n = 7
    c = np.array([2.0, -1.0, 0.5, 3.0, 0.1, 0.2, 1.5])  # 最小は index=1
    j = int(np.argmin(c))

    def fun(p):
        return float(c @ p), c.copy()

    opt = replace(
        PGSolverOptions(),
        max_iter=400,
        tol=1e-12,
        step_size=1.0,
        rho=0.5,
        sigma=1e-4,
        step_size_inf=1e-16,
        store_trace=False,
        verbose=False,
    )
    solver = ProjectedGradientSolver(opt)

    p0 = np.ones(n) / n
    p, info = solver.solve(fun, project_onto_simplex, p0)

    assert np.all(p >= -1e-12)
    assert np.isclose(p.sum(), 1.0, atol=1e-10)

    # ほぼone-hot（厳密に1.0を要求すると不安定なので緩め）
    assert p[j] >= 0.999
    assert np.all(np.delete(p, j) <= 1e-3)
