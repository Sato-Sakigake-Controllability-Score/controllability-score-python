import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional

def validate_A(A: np.ndarray) -> None:
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy array")
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.isrealobj(A):
        raise ValueError("A must be real-valued")

def validate_T(T: float) -> float:
    """
    Validate time horizon T and return it as float.

    Accepts np.inf, rejects NaN, requires positive finite values.
    """
    try:
        t = float(T)
    except (TypeError, ValueError) as e:
        raise TypeError("T must be a real scalar (float-compatible).") from e

    if np.isnan(t):
        raise ValueError("T must not be NaN.")
    if t <= 0.0:
        raise ValueError("T must be positive (or np.inf).")
    return t

def validate_block_sizes(block_sizes: np.ndarray, n: int) -> None:
    if block_sizes.ndim != 1:
        raise ValueError("block_sizes must be a vector")
    if not np.all(block_sizes > 0):
        raise ValueError("block_sizes must be positive")
    if not np.all(block_sizes.astype(int) == block_sizes):
        raise ValueError("block_sizes must be integer-valued")
    if block_sizes.sum() != n:
        raise ValueError("sum(block_sizes) must equal n")

def validate_initial_guess(p: np.ndarray, n: int) -> None:
    if p.ndim != 1:
        raise ValueError("InitialGuess must be a vector")
    if p.size != n:
        raise ValueError(f"InitialGuess must have length {n}")
    if not np.isrealobj(p):
        raise ValueError("InitialGuess must be real")



def validate_block_indices(
    vcs_blocks: npt.NDArray[np.int64],
    aecs_blocks: npt.NDArray[np.int64],
    B: int,
) -> None:
    if vcs_blocks.ndim != 1 or vcs_blocks.size < 1:
        raise ValueError("vcs_blocks must be a non-empty 1D array.")
    if aecs_blocks.ndim != 1 or aecs_blocks.size < 1:
        raise ValueError("aecs_blocks must be a non-empty 1D array.")
    if np.any(vcs_blocks < 0) or np.any(vcs_blocks >= B):
        raise ValueError(f"vcs_blocks must be in [0, {B-1}].")
    if np.any(aecs_blocks < 0) or np.any(aecs_blocks >= B):
        raise ValueError(f"aecs_blocks must be in [0, {B-1}].")





def validate_w_list(w_list: Any, n: int, block_sizes: npt.NDArray[np.int64]) -> None:
    if not isinstance(w_list, list) or len(w_list) != n:
        raise ValueError("w_list must be a list of length n.")
    B = int(block_sizes.size)
    for i in range(n):
        entry = w_list[i]
        blocks = entry.get("Blocks") if isinstance(entry, dict) else entry
        if not isinstance(blocks, list) or len(blocks) != B:
            raise ValueError("Each w_list entry must contain B blocks.")
        for k in range(B):
            nk = int(block_sizes[k])
            blk = blocks[k]
            if not isinstance(blk, np.ndarray) or blk.shape != (nk, nk):
                raise ValueError("Each block must be an ndarray of shape (nk, nk).")



def validate_blocks(blocks: npt.NDArray[np.int64], nb: int, name: str) -> None:
    """Non-empty 1D int array, finite, in [1..nb], no duplicates."""
    if blocks.ndim != 1 or blocks.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    if not np.all(np.isfinite(blocks)):
        raise ValueError(f"{name} must contain finite integers.")
    if np.any(blocks != np.floor(blocks)):
        raise ValueError(f"{name} must contain integers.")
    if np.any(blocks < 0) or np.any(blocks >= nb):
        raise ValueError(f"{name} must be in [0, {nb-1}].")
    if np.unique(blocks).size != blocks.size:
        raise ValueError(f"{name} contains duplicate indices.")


def validate_w_list_blocks(
    w_list: List[List[npt.NDArray[np.float64]]],
    block_sizes: npt.NDArray[np.int64],
) -> None:
    """Validate W[i][b] shapes and real numeric type. W blocks must be non-empty."""
    nb = int(block_sizes.size)
    n = int(block_sizes.sum())

    if not isinstance(w_list, list) or len(w_list) != n:
        raise ValueError(f"w_list must be a list of length n={n}.")

    for i, Wi in enumerate(w_list):
        if not isinstance(Wi, list) or len(Wi) != nb:
            raise ValueError(f"w_list[{i}] must have nb={nb} blocks.")

        for b in range(nb):
            sb = int(block_sizes[b])
            blk = Wi[b]

            if not isinstance(blk, np.ndarray):
                raise TypeError(f"w_list[{i}][{b}] must be a numpy array.")
            if blk.ndim != 2 or blk.shape != (sb, sb):
                raise ValueError(f"w_list[{i}][{b}] must be shape ({sb}, {sb}), got {blk.shape}.")
            if blk.size == 0:
                raise ValueError(f"w_list[{i}][{b}] must be non-empty.")
            if not np.issubdtype(blk.dtype, np.number) or np.iscomplexobj(blk):
                raise TypeError(f"w_list[{i}][{b}] must be real numeric.")


def validate_transform_matrix(Q: Optional[npt.NDArray[np.float64]], n: int) -> None:
    if Q is None:
        return
    if not isinstance(Q, np.ndarray):
        raise TypeError("transform_matrix must be a numpy array or None.")
    if Q.ndim != 2 or Q.shape != (n, n):
        raise ValueError(f"transform_matrix must be shape ({n}, {n}), got {Q.shape}.")
    if not np.issubdtype(Q.dtype, np.number) or np.iscomplexobj(Q):
        raise TypeError("transform_matrix must be real numeric.")


def validate_aecs_matrix(
    Sa: Optional[List[Optional[npt.NDArray[np.float64]]]],
    block_sizes: npt.NDArray[np.int64],
) -> None:
    """
    Pythonic:
      - Sa can be None (unused).
      - If provided: list length nb; each block is None or (sb,sb) real numeric.
    """
    if Sa is None:
        return
    if not isinstance(Sa, list):
        raise TypeError("aecs_matrix must be a list or None.")

    nb = int(block_sizes.size)
    if len(Sa) != nb:
        raise ValueError(f"aecs_matrix must have length nb={nb}, got {len(Sa)}.")

    for b in range(nb):
        sb = int(block_sizes[b])
        Sb = Sa[b]
        if Sb is None:
            continue
        if not isinstance(Sb, np.ndarray):
            raise TypeError(f"aecs_matrix[{b}] must be a numpy array or None.")
        if Sb.ndim != 2 or Sb.shape != (sb, sb):
            raise ValueError(f"aecs_matrix[{b}] must be shape ({sb}, {sb}), got {Sb.shape}.")
        if not np.issubdtype(Sb.dtype, np.number) or np.iscomplexobj(Sb):
            raise TypeError(f"aecs_matrix[{b}] must be real numeric.")
