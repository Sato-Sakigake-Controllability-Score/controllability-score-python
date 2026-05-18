import numpy as np

from cs.results import (
    CSResult,
    CSTraceItem,
    VCSResults,
    AECSResults,
    CSResults,
)
from cs.options import PGSolverOptions


def test_csresult_defaults():
    r = CSResult()
    assert np.isnan(r.objective_value)
    assert r.gradient.shape == (0,)
    assert np.isnan(r.grad_norm)
    assert np.isnan(r.step_norm)

    assert r.iterations == 0
    assert r.func_count == 0
    assert r.converged is False
    assert r.exit_flag == 0
    assert r.exit_message == ""
    assert r.algorithm == ""
    assert r.solver_options is None
    assert r.problem_info == {}
    assert r.trace is None


def test_csresult_can_store_options_problem_info_and_trace():
    opt = PGSolverOptions(max_iter=10)
    trace = [
        CSTraceItem(iter=0, objective=1.0, grad_norm=2.0, step_norm=0.5, step_size=0.1),
        CSTraceItem(iter=1, objective=0.8, grad_norm=1.5, step_norm=0.2, step_size=0.05),
    ]
    r = CSResult(
        objective_value=0.8,
        gradient=np.array([1.0, 2.0], dtype=float),
        grad_norm=1.5,
        step_norm=0.2,
        iterations=2,
        func_count=5,
        converged=True,
        exit_flag=1,
        exit_message="converged",
        algorithm="Projected gradient with Armijo line search",
        solver_options=opt,
        problem_info={"Objective": "VCS", "Dimension": 2},
        trace=trace,
    )
    assert r.solver_options.max_iter == 10
    assert r.problem_info["Objective"] == "VCS"
    assert r.trace is not None
    assert len(r.trace) == 2
    assert r.trace[0].iter == 0


def test_vcsresults_basic_shape():
    p = np.array([0.2, 0.8], dtype=float)
    info = CSResult(iterations=3, converged=True)
    res = VCSResults(p=p, info=info)

    assert res.p.shape == (2,)
    assert res.info.converged is True
    assert res.w_list is None
    assert res.transform_matrix is None
    assert res.w_output is None


def test_aecsresults_basic_shape():
    p = np.array([0.3, 0.7], dtype=float)
    info = CSResult(iterations=4, converged=False, exit_flag=0)
    res = AECSResults(p=p, info=info)

    assert res.p.shape == (2,)
    assert res.info.iterations == 4
    assert res.w_list is None
    assert res.transform_matrix is None
    assert res.w_output is None


def test_csresults_bundles_vcs_and_aecs():
    pV = np.array([0.1, 0.9], dtype=float)
    pA = np.array([0.4, 0.6], dtype=float)
    infoV = CSResult(converged=True, exit_flag=1)
    infoA = CSResult(converged=True, exit_flag=1)

    vcs = VCSResults(p=pV, info=infoV)
    aecs = AECSResults(p=pA, info=infoA)

    both = CSResults(vcs=vcs, aecs=aecs)

    assert both.vcs.p.shape == (2,)
    assert both.aecs.p.shape == (2,)
    assert both.w_list is None
    assert both.transform_matrix is None
    assert both.w_output is None


def test_optional_w_list_and_transform_matrix_can_be_attached():
    n = 3
    p = np.ones(n) / n
    info = CSResult()
    WList = [np.eye(n, dtype=float) for _ in range(n)]
    P = np.eye(n, dtype=float)

    res = VCSResults(p=p, info=info, w_list=WList, transform_matrix=P, w_output="trans")

    assert res.w_list is not None
    assert len(res.w_list) == n
    assert res.w_list[0].shape == (n, n)
    assert res.transform_matrix is not None
    assert res.transform_matrix.shape == (n, n)
    assert res.w_output == "trans"
