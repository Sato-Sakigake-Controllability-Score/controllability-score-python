
import numpy as np
import numpy.typing as npt
from ..problem import CSProblem

Array = npt.NDArray[np.float64]


def assemble_w_block(problem:CSProblem, p: npt.ArrayLike, k1: int) -> Array:
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
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    n = int(problem.dimension)
    if p.size != n:
        raise ValueError(f"p must have length equal to dimension ({n}). Got {p.size}.")

    B = int(np.asarray(problem.wlist.block_sizes).size)
    if k1 < 0 or k1 >= B:
        raise ValueError(f"Block index k out of range. Got k={k1}, valid 1..{B}.")

    nk = int(problem.wlist.block_sizes[k1])

    # Start from zeros like MATLAB
    Wk = np.zeros((nk, nk), dtype=np.float64)

    # Add p_i * W_{i,k} (skip if p_i == 0 like MATLAB)
    w_list = problem.wlist.w_list
    for i in range(n):
        pi = float(p[i])
        if pi != 0.0:
            Wk += pi * np.asarray(w_list[i][k1], dtype=np.float64)

    # Symmetrize (guards against numerical asymmetry)
    Wk = 0.5 * (Wk + Wk.T)
    return Wk
