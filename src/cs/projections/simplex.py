# src/cs/projections/simplex.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import cast


Array = npt.NDArray[np.float64]


def project_onto_simplex(p: npt.ArrayLike) -> Array:
    """
    Project a vector onto the probability simplex (Euclidean projection):

        { x : x >= 0, sum(x) = 1 }.

    Parameters
    ----------
    p : array-like
        Input vector (any shape convertible to 1D).

    Returns
    -------
    x : ndarray, shape (n,)
        Projected vector on the simplex.

    Notes
    -----
    This implements the classic O(n log n) algorithm:
    - Sort p in descending order
    - Find rho = max { j : u_j - (1/j)(sum_{i=1}^j u_i - 1) > 0 }
    - theta = (sum_{i=1}^rho u_i - 1) / rho
    - x = max(p - theta, 0)
    - Final renormalization to enforce sum(x)=1 (numerical guard)
    """
    x_in = np.asarray(p, dtype=np.float64).reshape(-1)

    if x_in.size == 0:
        raise ValueError("Input must be a nonempty vector.")
    if not np.all(np.isfinite(x_in)):
        raise ValueError("Input must be finite.")

    n = x_in.size

    # Sort in descending order
    u = np.sort(x_in)[::-1]

    # Cumulative sum
    cssv = np.cumsum(u)

    # t_j = (sum_{i=1}^j u_i - 1) / j
    j = np.arange(1, n + 1, dtype=np.float64)
    t = (cssv - 1.0) / j

    # rho = last index where u - t > 0
    idx = np.nonzero(u - t > 0.0)[0]
    if idx.size == 0:
        # Should not happen for finite input, but keep it safe.
        return cast(Array, np.ones(n, dtype=np.float64) / np.float64(n))

    rho = idx[-1]
    theta = t[rho]

    # Projection
    x = np.maximum(x_in - theta, np.float64(0.0))

    # Numerical guard: enforce exact sum=1 if close / avoid divide-by-zero
    s = float(x.sum())
    if s > 0.0:
        x /= s
    else:
        x[:] = 1.0 / float(n)

    return cast(Array, x)
