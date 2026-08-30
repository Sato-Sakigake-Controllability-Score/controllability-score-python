from dataclasses import dataclass, field
from typing import  Optional, Literal
import numpy as np
import numpy.typing as npt

from .options import WOptions
from .gramian.wlist import WList
from .gramian.compute_w import compute_w
from .utils import validate as v
WOutput = Literal["orig", "trans"]


@dataclass(frozen=True, slots=True)
class CSProblem:
    """
    CSProblem
    =========
    Definition of a controllability-score optimization problem.

    This class encapsulates all data and structural information required
    to evaluate and solve controllability-score–based optimization problems
    (VCS / AECS).  The problem is fully constructed at initialization time
    and remains immutable thereafter.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix of the linear dynamical system.

    T : float, optional
        Time horizon. Must be positive or np.inf.
        Defaults to np.inf.

    w_options : WOptions, optional
        Options controlling the computation of controllability Gramians.
        If not provided, a default WOptions(A, T) is constructed.

    Notes
    -----
    - All structural quantities (Gramian blocks, block sizes, block indices,
      coordinate transforms, and AECS matrices) are computed during
      initialization via ``gramian.compute_w``.
    - The resulting block-diagonal structure is immutable after construction.
    - After a CSProblem instance is created, all solvers may assume that
      dimensional consistency and structural validity have already been
      verified.

    Block-Diagonal Representation
    ------------------------------
    The controllability Gramians are stored in a block-diagonal form:

    - ``w_list`` is a list of length n (``dimension``).
    - ``w_list[i]`` is a dictionary containing the key ``"Blocks"``.
    - ``w_list[i]`` is a list of length B, where B is the number
      of blocks.
    - The k-th block has shape ``(block_sizes[k], block_sizes[k])``.

    Block Configuration
    -------------------
    - ``block_sizes`` is a 1D array of positive integers whose sum equals n.
    - ``vcs_blocks`` specifies which block indices are used in VCS
      optimization.
    - ``aecs_blocks`` specifies which block indices are used in AECS
      optimization.

    The block configuration is determined by ``gramian.compute_w`` and
    cannot be modified after construction.

    Design Philosophy
    -----------------
    This class follows a "validate-once, trust-forever" design:

    - All input validation and structural checks are performed during
      initialization.
    - Solver implementations are not required to re-check dimensions or
      consistency.
    - This design mirrors the behavior of the original MATLAB implementation
      while adopting a more explicit and immutable Python interface.

    """

    A: npt.NDArray[np.float64]
    T: float = np.inf
    w_options: Optional[WOptions] = None

    # computed / immutable after construction
    wlist: WList = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        v.validate_A(A)
        T = float(self.T)
        v.validate_T(T)

        n = A.shape[0]
        wopt = self.w_options or WOptions.from_system(A, T)
        wlist = compute_w(A, T, wopt)

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "w_options", wopt)
        object.__setattr__(self, "dimension", n)

        object.__setattr__(self, "wlist", wlist)
    
    def W_original_list(
        self,
        *,
        w_output: WOutput = "trans",
        inf_keep: str = "stable_only",
    ) -> list[np.ndarray]:
        """
        Return the list of all W_i in original coordinates.

        Mode decision is based on:
            - w_options.use_scaling
            - finite vs infinite horizon (self.T)

        Parameters
        ----------
        inf_keep : {'stable_only', 'all'}
            For infinite-horizon scaled case:
              - 'stable_only': keep only stable (S) block
              - 'all': keep all blocks

        Returns
        -------
        W_list : list of (n,n) ndarray
        """
        return self.wlist.export_matrices(
            T=self.T,
            use_scaling=bool(getattr(self.w_options, "use_scaling", False)),
            w_output=w_output,
            inf_keep=inf_keep,
        )
