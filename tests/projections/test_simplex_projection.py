# tests/projections/test_simplex_projection.py
import numpy as np
import pytest

from controllability_scoring.projections.simplex import project_onto_simplex


def _is_simplex(x: np.ndarray, atol: float = 1e-12) -> bool:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.all(x >= -atol) and np.isclose(float(x.sum()), 1.0, atol=atol)


def test_projection_returns_1d_float_array():
    p = np.array([[0.2, -1.0, 3.0]])  # 2DでもOK（内部で1D化）
    x = project_onto_simplex(p)
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1
    assert x.dtype.kind == "f"
    assert x.size == 3


def test_projection_feasibility_nonneg_and_sum1():
    p = np.array([0.2, -1.0, 3.0])
    x = project_onto_simplex(p)
    assert _is_simplex(x, atol=1e-10)


def test_projection_idempotent():
    rng = np.random.default_rng(0)
    p = rng.normal(size=20)
    x = project_onto_simplex(p)
    xx = project_onto_simplex(x)
    assert np.allclose(x, xx, atol=1e-12, rtol=0.0)


def test_projection_keeps_simplex_points_unchanged():
    # すでに simplex 上の点はそのまま（数値誤差込みで）
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert _is_simplex(x0)

    x = project_onto_simplex(x0)
    assert np.allclose(x, x0, atol=1e-12, rtol=0.0)


def test_projection_vertex_solution_for_large_coordinate():
    # 1成分が非常に大きい場合は、その成分にほぼ全質量が集まる
    p = np.array([-10.0, -10.0, 1000.0, -10.0])
    x = project_onto_simplex(p)
    assert _is_simplex(x, atol=1e-10)
    assert x[2] >= 1.0 - 1e-12
    assert np.all(np.delete(x, 2) <= 1e-12)


def test_projection_uniform_when_all_equal():
    p = np.array([5.0, 5.0, 5.0, 5.0])
    x = project_onto_simplex(p)
    assert _is_simplex(x, atol=1e-12)
    assert np.allclose(x, np.ones(4) / 4.0, atol=1e-12, rtol=0.0)


def test_reject_empty_input():
    with pytest.raises(ValueError, match="nonempty"):
        project_onto_simplex(np.array([]))


def test_reject_non_finite_input():
    with pytest.raises(ValueError, match="finite"):
        project_onto_simplex([0.1, np.inf, 0.9])

    with pytest.raises(ValueError, match="finite"):
        project_onto_simplex([0.1, np.nan, 0.9])
