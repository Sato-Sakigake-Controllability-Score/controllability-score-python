# import numpy as np
# import pytest
# from dataclasses import replace

# from cs.problem import CSProblem
# from cs.options import PGSolverOptions
# from cs.solvers.solve_aecs import solve_aecs


# def _assert_simplex(p: np.ndarray, atol: float = 1e-10) -> None:
#     p = np.asarray(p, dtype=float).reshape(-1)
#     assert np.all(p >= -1e-12)
#     assert np.isclose(float(p.sum()), 1.0, atol=atol)


# def test_solve_aecs_smoke_returns_p_and_info():
#     # Small stable system; should work even with stub compute_w
#     A = np.array([[-1.0, 0.0],
#                   [0.0, -2.0]], dtype=float)
#     prob = CSProblem(A=A, T=np.inf)

#     p, info = solve_aecs(prob)

#     assert isinstance(p, np.ndarray)
#     assert p.ndim == 1
#     assert p.size == prob.dimension

#     _assert_simplex(p)

#     # Basic CSResult fields should be populated
#     assert hasattr(info, "objective_value")
#     assert np.isfinite(float(info.objective_value))
#     assert hasattr(info, "iterations")
#     assert hasattr(info, "func_count")
#     assert hasattr(info, "exit_flag")
#     assert hasattr(info, "converged")


# def test_solve_aecs_accepts_initial_guess_override():
#     A = np.array([[-1.0, 0.0],
#                   [0.0, -2.0]], dtype=float)
#     prob = CSProblem(A=A, T=np.inf)

#     p0 = np.array([0.9, 0.1], dtype=float)
#     p, info = solve_aecs(prob, initial_guess=p0)

#     assert p.shape == (2,)
#     _assert_simplex(p)


# def test_solve_aecs_rejects_wrong_initial_guess_length():
#     A = np.array([[-1.0, 0.0],
#                   [0.0, -2.0]], dtype=float)
#     prob = CSProblem(A=A, T=np.inf)

#     with pytest.raises(ValueError, match=r"InitialGuess must have length"):
#         solve_aecs(prob, initial_guess=np.array([1.0, 0.0, 0.0], dtype=float))


# def test_solve_aecs_respects_solver_options_max_iter_small():
#     A = np.array([[-1.0, 0.0],
#                   [0.0, -2.0]], dtype=float)
#     prob = CSProblem(A=A, T=np.inf)

#     opt = replace(
#         PGSolverOptions(),
#         max_iter=2,
#         tol=1e-30,         # Make convergence difficult to hit max_iter
#         step_size=1.0,
#         step_size_inf=1e-16,
#         rho=0.5,
#         sigma=1e-4,
#         store_trace=False,
#         verbose=False,
#     )

#     p, info = solve_aecs(prob, solver_options=opt)

#     _assert_simplex(p)

#     # Solver should not exceed max_iter
#     assert int(info.iterations) <= 2
#     assert int(info.func_count) >= 1


# def test_solve_aecs_default_solution_is_uniform_for_identity_aecs_matrix():
#     """
#     The stub compute_w returns aecs_matrix = I, so the objective in
#     make_aecs_fun becomes 0.5 * ||p||^2 over the simplex.
#     The unique minimizer is the uniform distribution.

#     This test fixes a meaningful numerical behavior of solve_aecs
#     in the stub environment.
#     """
#     A = np.array([[-1.0, 0.0],
#                   [0.0, -2.0]], dtype=float)
#     prob = CSProblem(A=A, T=np.inf)

#     opt = replace(
#         PGSolverOptions(),
#         max_iter=500,
#         tol=1e-12,
#         step_size=1.0,
#         store_trace=False,
#         verbose=False,
#     )
#     p, info = solve_aecs(prob, solver_options=opt)

#     u = np.ones(prob.dimension) / prob.dimension
#     assert np.linalg.norm(p - u) <= 1e-6
