# src/controllability_scoring/solve_context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt

from ..options import PGSolverOptions
from ..utils import validate as v

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SolveContext:
    """
    Solver-side context:
    - how to initialize p
    - solver hyperparameters
    """
    initial_guess: Optional[npt.ArrayLike] = None
    solver_options: PGSolverOptions = PGSolverOptions()

    def resolve_initial_guess(self, n: int) -> Array:
        """
        Resolve initial guess for a problem of dimension n.

        Rules:
        - If initial_guess is None: use uniform simplex (1/n).
        - Otherwise: validate user-provided guess.
        """
        ig = self.initial_guess
        if ig is None:
            return np.ones((n,), dtype=float) / n

        ig_arr = np.asarray(ig, dtype=float).reshape(-1)
        v.validate_initial_guess(ig_arr, n)
        return ig_arr

    # Optional convenience builders
    def with_initial_guess(self, ig: npt.ArrayLike) -> "SolveContext":
        return SolveContext(
            initial_guess=ig,
            solver_options=self.solver_options,
        )

    def with_solver_options(self, opt: PGSolverOptions) -> "SolveContext":
        return SolveContext(
            initial_guess=self.initial_guess,
            solver_options=opt,
        )
