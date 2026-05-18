# tests/solvers/test_projected_gradient_integration.py
import numpy as np
from dataclasses import replace

from cs.pg_solvers.project_gradient import ProjectedGradientSolver
from cs.options import PGSolverOptions


def identity_proj(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def project_onto_simplex(p: np.ndarray) -> np.ndarray:
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


def test_runs_with_default_options_identity_proj():
    # 既定オプションで「落ちない」ことが主目的
    solver = ProjectedGradientSolver()

    def fun(p):
        f = 0.5 * float(p @ p)
        g = p
        return f, g

    p, info = solver.solve(fun, identity_proj, np.array([1.0, -2.0, 3.0]))

    assert np.all(np.isfinite(p))
    assert np.isfinite(info.objective_value)
    assert info.func_count >= 1
    assert info.iterations >= 1
    # exit_flag は収束(1)か最大反復(0)のどちらかが通常
    assert info.exit_flag in (0, 1)


def test_runs_with_trace_on_and_off():
    def fun(p):
        f = 0.5 * float(p @ p)
        g = p
        return f, g

    base = PGSolverOptions()

    # trace ON
    opt_on = replace(base, store_trace=True, verbose=False, max_iter=50, tol=1e-12)
    s_on = ProjectedGradientSolver(opt_on)
    p_on, info_on = s_on.solve(fun, identity_proj, np.array([2.0, -1.0]))
    assert info_on.trace is not None
    assert len(info_on.trace) == info_on.iterations

    # trace OFF
    opt_off = replace(base, store_trace=False, verbose=False, max_iter=50, tol=1e-12)
    s_off = ProjectedGradientSolver(opt_off)
    p_off, info_off = s_off.solve(fun, identity_proj, np.array([2.0, -1.0]))
    assert info_off.trace is None


def test_end_to_end_with_simplex_projection():
    """
    solver + simplex proj の組み合わせが普通に動くこと。
    （詳細な最適解検証は numeric 側でやる）
    """
    opt = replace(PGSolverOptions(), max_iter=300, tol=1e-10, verbose=False, store_trace=False)
    solver = ProjectedGradientSolver(opt)

    rng = np.random.default_rng(0)
    n = 8
    q = rng.normal(size=n)

    def fun(p):
        r = p - q
        f = 0.5 * float(r @ r)
        g = r
        return f, g

    p0 = np.ones(n) / n
    p, info = solver.solve(fun, project_onto_simplex, p0)

    # 出力の性質（落ちない、可行、数値が変でない）
    assert np.all(np.isfinite(p))
    assert np.all(p >= -1e-12)
    assert np.isclose(p.sum(), 1.0, atol=1e-10)
    assert np.isfinite(info.objective_value)
    assert info.exit_flag in (0, 1, -1)  # 研究コードでは設定次第で-1もあり得る
