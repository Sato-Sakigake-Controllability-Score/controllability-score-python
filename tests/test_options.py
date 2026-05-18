import numpy as np
import pytest

from cs.options import WOptions


def test_woptions_defaults_inf():
    opt = WOptions.from_system(T=np.inf)
    assert opt.method == "lyap"
    assert opt.steps == 0
    assert opt.use_scaling is True


def test_woptions_defaults_finite():
    opt = WOptions.from_system(T=10.0)
    assert opt.method == "integral"
    assert opt.steps == 50
    assert opt.use_scaling is True


def test_woptions_lyap_forces_steps_zero():
    opt = WOptions(method="lyap", steps=999, use_scaling=False)
    assert opt.method == "lyap"
    assert opt.steps == 0
    assert opt.use_scaling is False


def test_woptions_non_lyap_requires_steps_ge_1():
    with pytest.raises(Exception):
        WOptions(method="integral", steps=0)


def test_woptions_from_system_method_override_sets_steps_50_if_missing():
    # MATLAB intent: lyap(default for T=inf) -> non-lyap => steps auto 50
    A = np.eye(3)
    opt = WOptions.from_system(A, np.inf, method="trapezoidal")
    assert opt.method == "trapezoidal"
    assert opt.steps == 50


def test_woptions_with_method_transition_lyap_to_non_lyap_sets_50():
    opt = WOptions(method="lyap", steps=0)
    opt2 = opt.with_method("simpson")
    assert opt2.method == "simpson"
    assert opt2.steps == 50


def test_woptions_with_steps_keeps_constraints():
    opt = WOptions.from_system(T=10.0)  # integral, steps 50
    opt2 = opt.with_steps(100)
    assert opt2.method == "integral"
    assert opt2.steps == 100

    # If method is lyap, steps is forced to 0 even if user tries otherwise
    opt3 = WOptions(method="lyap", steps=0).with_steps(100)
    assert opt3.method == "lyap"
    assert opt3.steps == 0
