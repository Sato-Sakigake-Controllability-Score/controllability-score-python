# src/controllability_scoring/api/vcs.py
from __future__ import annotations

from typing import Optional, Tuple, List, Literal
import math
import numpy as np
import numpy.typing as npt

from .problem import CSProblem
from .options import WOptions, PGSolverOptions
from .results import CSResult
from .solvers.solve_vcs import solve_vcs
from .solvers.solve_aecs import solve_aecs
from .solvers.solve_context import SolveContext

Array = npt.NDArray[np.float64]
WOutput = Literal["orig", "trans"]

def vcs(
    A: npt.ArrayLike,
    T: float = math.inf,
    *,
    w_options: Optional[WOptions] = None,
    steps: Optional[int] = None,
    solver_options: Optional[PGSolverOptions] = None,
    initial_guess: Optional[npt.ArrayLike] = None,
    w_output: WOutput = "orig",
    inf_keep:str = "all"
) -> Tuple[Array, CSResult, List[Array], Optional[Array]]:
    A_mat = np.asarray(A, dtype=np.float64)
    if A_mat.ndim != 2 or A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError("A must be a square matrix.")
    T_val = float(T) if T is not None else math.inf

    wopt = w_options if w_options is not None else WOptions.from_system(A_mat, T_val)
    if steps is not None:
        wopt = wopt.with_steps(int(steps))

    prob = CSProblem(A_mat, T_val, w_options=wopt)

    # Build internal context (not exposed in API)
    ctx = SolveContext(
        initial_guess=initial_guess,
        solver_options=PGSolverOptions() if solver_options is None else solver_options,
    )

    p, info = solve_vcs(prob, context=ctx)
    WList = prob.wlist.export_matrices(
        T=prob.T,
        use_scaling=bool(prob.w_options.use_scaling),
        w_output=w_output,
        inf_keep=inf_keep,
    )
    # WList: List[Array] = [
    #     np.asarray(prob.wlist.w_list[i][0], dtype=np.float64)
    #     for i in range(prob.dimension)
    # ]

    Pm = prob.wlist.transform_matrix
    # P: Optional[Array] = None
    # if Pm is not None and np.size(Pm) != 0:
    #     P = np.asarray(Pm, dtype=np.float64)
    # if w_output == "orig" and P is not None:
    #     WList = [P @ W @ P.T for W in WList]

    return p, info, WList, Pm

def aecs(
    A: npt.ArrayLike,
    T: float = math.inf,
    *,
    w_options: Optional[WOptions] = None,
    steps: Optional[int] = None,
    solver_options: Optional[PGSolverOptions] = None,
    initial_guess: Optional[npt.ArrayLike] = None,
    w_output: WOutput = "orig",
    inf_keep:str = "all"
) -> Tuple[Array, CSResult, List[Array], Optional[Array]]:
    A_mat = np.asarray(A, dtype=np.float64)
    if A_mat.ndim != 2 or A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError("A must be a square matrix.")
    T_val = float(T) if T is not None else math.inf

    wopt = w_options if w_options is not None else WOptions.from_system(A_mat, T_val)
    if steps is not None:
        wopt = wopt.with_steps(int(steps))

    prob = CSProblem(A_mat, T_val, w_options=wopt)

    # Build internal context (not exposed in API)
    ctx = SolveContext(
        initial_guess=initial_guess,
        solver_options=PGSolverOptions() if solver_options is None else solver_options,
    )

    p, info = solve_aecs(prob, context=ctx)
    WList = prob.wlist.export_matrices(
        T=prob.T,
        use_scaling=bool(prob.w_options.use_scaling),
        w_output=w_output,
        inf_keep=inf_keep,
    )

    # WList: List[Array] = [
    #     np.asarray(prob.wlist.w_list[i][0], dtype=np.float64)
    #     for i in range(prob.dimension)
    # ]
    Pm = prob.wlist.transform_matrix
    # Pm = getattr(prob, "transform_matrix", None)
    # P: Optional[Array] = None
    # if Pm is not None and np.size(Pm) != 0:
    #     P = np.asarray(Pm, dtype=np.float64)

    # if w_output == "orig" and P is not None:
    #     WList = [P @ W @ P.T for W in WList]

    return p, info, WList, Pm


def cs(
    A: npt.ArrayLike,
    T: float = math.inf,
    *,
    w_options: Optional[WOptions] = None,
    steps: Optional[int] = None,
    solver_options: Optional[PGSolverOptions] = None,
    initial_guess: Optional[npt.ArrayLike] = None,
    w_output: WOutput = "orig",
    inf_keep:str = "all"
) -> Tuple[
    Array,            # pV
    Array,            # pA
    CSResult,         # infoV
    CSResult,         # infoA
    List[Array],      # WList (exported)
    Optional[Array],  # P
]:
    """
    Solve both VCS and AECS from the same CSProblem instance.

    MATLAB reference (cs):
      prob = CSProblem(A,T,"WOptions", wopt)
      [pV, infoV] = prob.solveVcs(...)
      [pA, infoA] = prob.solveAecs(...)
      if opt.WOutput == "orig": WList = P*W*P'
      and returns P if requested.

    Python adaptation:
    - always returns (pV, pA, infoV, infoA, WList, P)
    - WList uses Blocks[0] like existing vcs()/aecs() API
    """
    A_mat = np.asarray(A, dtype=np.float64)
    if A_mat.ndim != 2 or A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError("A must be a square matrix.")
    T_val = float(T) if T is not None else math.inf

    # Build WOptions
    wopt = w_options if w_options is not None else WOptions.from_system(A_mat, T_val)
    if steps is not None:
        wopt = wopt.with_steps(int(steps))

    # Single shared problem instance
    prob = CSProblem(A_mat, T_val, w_options=wopt)

    # Shared internal context (initial guess + solver options)
    ctx = SolveContext(
        initial_guess=initial_guess,
        solver_options=PGSolverOptions() if solver_options is None else solver_options,
    )

    # Solve both objectives
    pV, infoV = solve_vcs(prob, context=ctx)
    pA, infoA = solve_aecs(prob, context=ctx)

    # Export WList (match existing api/vcs.py behavior: first block only)
    WList = prob.wlist.export_matrices(
        T=prob.T,
        use_scaling=bool(prob.w_options.use_scaling),
        w_output=w_output,
        inf_keep=inf_keep,
    )

    # WList: List[Array] = [
    #     np.asarray(prob.wlist.w_list[i][0], dtype=np.float64)
    #     for i in range(prob.dimension)
    # ]

    # Export transform matrix P if present
    Pm = prob.wlist.transform_matrix
    # Pm = getattr(prob, "transform_matrix", None)
    # P: Optional[Array] = None
    # if Pm is not None and np.size(Pm) != 0:
    #     P = np.asarray(Pm, dtype=np.float64)

    # Map back to original coordinates if requested
    # if w_output == "orig" and P is not None:
    #     WList = [P @ W @ P.T for W in WList]

    return pV, pA, infoV, infoA, WList, Pm
