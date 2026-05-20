# tests/solvers/test_projected_gradient_unit.py
import numpy as np
import pytest
from dataclasses import replace

from controllability_scoring.pg_solvers.project_gradient import ProjectedGradientSolver
from controllability_scoring.options import PGSolverOptions


def identity_proj(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def test_p0_must_be_1d_vector():
    solver = ProjectedGradientSolver()
    def fun(p):
        return 0.0, np.zeros_like(p)
    with pytest.raises(ValueError, match="p0 must be a 1D real vector"):
        solver.solve(fun, identity_proj, np.zeros((2, 2)))


def test_p0_must_be_finite():
    solver = ProjectedGradientSolver()
    def fun(p):
        return 0.0, np.zeros_like(p)
    with pytest.raises(ValueError, match="p0 must be finite"):
        solver.solve(fun, identity_proj, np.array([0.0, np.inf]))


def test_proj_must_return_non_empty_1d():
    solver = ProjectedGradientSolver()
    def fun(p):
        return 0.0, np.zeros_like(p)

    def bad_proj(_p):
        return np.array([])

    with pytest.raises(ValueError, match=r"proj\(p0\) must return a non-empty 1D vector"):
        solver.solve(fun, bad_proj, np.array([1.0, 2.0]))


def test_initial_eval_fail_when_f_not_finite_sets_exit_minus2():
    opt = replace(PGSolverOptions(), max_iter=5)
    solver = ProjectedGradientSolver(opt)

    def fun(_p):
        return (np.inf, np.array([0.0, 0.0]))

    p, info = solver.solve(fun, identity_proj, np.array([0.2, 0.8]))
    assert info.exit_flag == -2
    assert info.converged is False
    assert info.iterations == 0
    assert info.func_count == 1
    assert "Initial point evaluation failed" in info.exit_message


def test_initial_eval_fail_when_grad_has_nan_sets_exit_minus2():
    opt = replace(PGSolverOptions(), max_iter=5)
    solver = ProjectedGradientSolver(opt)

    def fun(_p):
        return (0.0, np.array([np.nan, 0.0]))

    p, info = solver.solve(fun, identity_proj, np.array([0.2, 0.8]))
    assert info.exit_flag == -2
    assert info.converged is False
    assert info.iterations == 0
    assert info.func_count == 1


def test_armijo_backtracking_failure_sets_exit_minus1():
    # Armijoが絶対に通らないようなfunを作る
    opt = replace(
        PGSolverOptions(),
        max_iter=3,
        tol=1e-12,
        step_size=1.0,
        rho=0.5,
        sigma=1e-4,
        step_size_inf=1e-12,
        store_trace=False,
        verbose=False,
    )
    solver = ProjectedGradientSolver(opt)

    state = {"fp0": None}

    def fun(p):
        # 初回のfpは0.0、以降は必ず fp0+1.0 を返して改善しない
        if state["fp0"] is None:
            state["fp0"] = 0.0
            f = 0.0
        else:
            f = state["fp0"] + 1.0
        g = np.ones_like(p)
        return float(f), g

    p, info = solver.solve(fun, identity_proj, np.array([0.0, 0.0]))
    assert info.exit_flag == -1
    assert info.converged is False
    assert "Armijo" in info.exit_message


def test_trace_length_matches_iterations_when_store_trace_true():
    opt = replace(
        PGSolverOptions(),
        max_iter=50,
        tol=1e-12,
        step_size=1.0,
        rho=0.5,
        sigma=1e-4,
        step_size_inf=1e-16,
        store_trace=True,
        verbose=False,
    )
    solver = ProjectedGradientSolver(opt)

    def fun(p):
        f = 0.5 * float(p @ p)
        g = p
        return f, g

    p, info = solver.solve(fun, identity_proj, np.array([2.0, -1.0, 0.5]))

    assert info.trace is not None
    assert len(info.trace) == info.iterations
    # acceptedしたiterが1..iterationsで並ぶこと
    assert [t.iter for t in info.trace] == list(range(1, info.iterations + 1))
    # func_countは少なくとも初期評価+各iterで1回以上評価される
    assert info.func_count >= 1 + info.iterations
