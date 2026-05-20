
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..problem import CSProblem

Array = npt.NDArray[np.float64]


def assemble_w_block(problem: "CSProblem", p: npt.ArrayLike, k1: int) -> Array:
    """
    Assemble W_k(p) for a CSProblem.

    MATLAB:
      Wk = sum_{i=1}^n p(i) * WList{i}.Blocks{k};
      Wk = 0.5 * (Wk + Wk.');

    Parameters
    ----------
    problem : CSProblem
        Must have: dimension, block_sizes, w_list
        where w_list[i][k0] is block matrix.
    p : array-like, shape (n,)
        Weight vector.
    k1 : int
        Block index.

    Returns
    -------
    Wk : ndarray, shape (nk, nk)
        Symmetrized assembled block.
    """
    return problem.wlist.assemble_block(p, k1)
