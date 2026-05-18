from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
import numpy as np
import numpy.typing as npt

from .options import PGSolverOptions  # あなたの options.py にある想定


ExitFlag = int


@dataclass(frozen=True, slots=True)
class CSTraceItem:
    """
    One iteration record (stored only when store_trace=True).
    Mirrors the typical MATLAB Trace struct fields.
    """
    iter: int
    objective: float
    grad_norm: float
    step_norm: float
    step_size: float


@dataclass(frozen=True, slots=True)
class CSResult:
    """
    Result/diagnostics of an optimization (Projected Gradient Solver) run.
    Python counterpart of MATLAB CSResult.
    """

    # --- final point diagnostics ---
    objective_value: float = np.nan
    gradient: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0,), dtype=float)
    )
    grad_norm: float = np.nan
    step_norm: float = np.nan

    # --- counters ---
    iterations: int = 0
    func_count: int = 0

    # --- termination ---
    converged: bool = False
    exit_flag: ExitFlag = 0
    exit_message: str = ""
    algorithm: str = ""

    # --- provenance ---
    solver_options: Optional[PGSolverOptions] = None
    problem_info: Dict[str, Any] = field(default_factory=dict)

    # --- optional trace ---
    # None: not stored, []: stored but empty
    trace: Optional[List[CSTraceItem]] = None


WOutput = Literal["orig", "trans"]


@dataclass(frozen=True, slots=True)
class VCSResults:
    """
    High-level result returned by vcs(A, T, ...).

    MATLAB outputs:
      p
      [p, info]
      [p, info, WList]
      [p, info, WList, P]
    """
    p: npt.NDArray[np.float64]
    info: CSResult

    # Optional exports (only if requested / computed)
    w_list: Optional[list[npt.NDArray[np.float64]]] = None  # full n-by-n matrices (length n)
    transform_matrix: Optional[npt.NDArray[np.float64]] = None  # P (n-by-n)
    # w_output は WList を返す場合に「どの座標系か」を明示したいときだけ有効
    w_output: Optional[WOutput] = None


@dataclass(frozen=True, slots=True)
class AECSResults:
    """
    High-level result returned by aecs(A, T, ...).

    MATLAB outputs:
      p
      [p, info]
      [p, info, WList]
      [p, info, WList, P]
    """
    p: npt.NDArray[np.float64]
    info: CSResult

    w_list: Optional[list[npt.NDArray[np.float64]]] = None
    transform_matrix: Optional[npt.NDArray[np.float64]] = None
    w_output: Optional[WOutput] = None


@dataclass(frozen=True, slots=True)
class CSResults:
    """
    High-level result returned by cs(A, T, ...), solving both VCS and AECS
    from the same CSProblem instance (MATLAB cs).

    MATLAB outputs:
      [pV, pA]
      [pV, pA, infoV, infoA]
      [pV, pA, infoV, infoA, WList]
      [pV, pA, infoV, infoA, WList, P]
    """
    vcs: VCSResults
    aecs: AECSResults

    # Optional shared exports from the underlying problem (if requested)
    w_list: Optional[list[npt.NDArray[np.float64]]] = None
    transform_matrix: Optional[npt.NDArray[np.float64]] = None
    w_output: Optional[WOutput] = None
