import numpy as np
import pytest

from controllability_scoring.options import PGSolverOptions


def test_pgsolver_defaults_are_valid():
    opt = PGSolverOptions()
    assert opt.step_size == 0.1
    assert opt.step_size_inf == 1e-12
    assert opt.max_iter == 1000
    assert opt.tol == 1e-8
    assert opt.rho == 0.5
    assert opt.sigma == 1e-4
    assert opt.verbose is False
    assert opt.store_trace is False


@pytest.mark.parametrize("field,value", [
    ("step_size", 0.0),
    ("step_size", -1.0),
    ("step_size_inf", 0.0),
    ("step_size_inf", -1.0),
    ("tol", 0.0),
    ("tol", -1.0),
])
def test_positive_float_fields_reject_nonpositive(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        PGSolverOptions(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("step_size", np.inf),
    ("step_size", np.nan),
    ("step_size_inf", np.inf),
    ("step_size_inf", np.nan),
    ("tol", np.inf),
    ("tol", np.nan),
    ("rho", np.inf),
    ("rho", np.nan),
    ("sigma", np.inf),
    ("sigma", np.nan),
])
def test_float_fields_reject_nonfinite(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        PGSolverOptions(**kwargs)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_open01_fields_reject_outside(value):
    with pytest.raises(ValueError):
        PGSolverOptions(rho=value)
    with pytest.raises(ValueError):
        PGSolverOptions(sigma=value)


def test_max_iter_must_be_positive_int():
    with pytest.raises(ValueError):
        PGSolverOptions(max_iter=0)
    with pytest.raises(ValueError):
        PGSolverOptions(max_iter=-1)
    with pytest.raises(TypeError):
        PGSolverOptions(max_iter=1.2)


def test_max_iter_rejects_bool():
    # bool is subclass of int; must be rejected explicitly
    with pytest.raises(TypeError):
        PGSolverOptions(max_iter=True)


def test_bool_like_fields_are_coerced_to_bool():
    opt = PGSolverOptions(verbose=np.bool_(True), store_trace=np.bool_(False))
    assert opt.verbose is True
    assert opt.store_trace is False


def test_with_methods_return_new_instance_and_validate():
    opt = PGSolverOptions()

    opt2 = opt.with_tol(1e-10).with_max_iter(2000).with_verbose(True)
    assert opt2 is not opt
    assert opt2.tol == 1e-10
    assert opt2.max_iter == 2000
    assert opt2.verbose is True

    # validation should still happen on replace
    with pytest.raises(ValueError):
        opt.with_rho(1.0)
