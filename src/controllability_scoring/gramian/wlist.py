from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Union, cast
import numpy as np
from numpy.typing import NDArray

import numpy.typing as npt
from scipy.linalg import block_diag, lu_factor, lu_solve

from ..options import WOptions
from ..utils.validate import (
    validate_blocks,
    validate_w_list_blocks,
    validate_transform_matrix,
    validate_aecs_matrix,
)

WOutput = Literal["orig", "trans"]


def _is_identity_matrix(A: np.ndarray | None) -> bool:
    if A is None or np.size(A) == 0:
        return True
    A = np.asarray(A, dtype=np.float64)
    return A.ndim == 2 and A.shape[0] == A.shape[1] and np.allclose(A, np.eye(A.shape[0]))


@dataclass(slots=True)
class WList:
    """
    Pythonic WList container.

    Required:
      - w_list: list length n, each is list length nb of (sb,sb) blocks

    Optional:
      - transform_matrix (Q): None if unused
      - aecs_matrix (Sa): None if unused
      - vcs_blocks / aecs_blocks: if None -> defaults (vcs=0..nb, aecs=[0])
      - w_options: stored

    Derived (auto):
      - block_sizes, num_blocks, dimension
    """

    # required
    w_list: List[List[npt.NDArray[np.float64]]]

    # optional
    transform_matrix: Optional[npt.NDArray[np.float64]] = None
    aecs_matrix: Optional[List[Optional[npt.NDArray[np.float64]]]] = None
    w_options: Optional[WOptions] = None
    DinvFull: Optional[np.ndarray] = None

    # optional user inputs (normalized to np.int64 arrays)
    vcs_blocks: Optional[Union[Sequence[int], npt.NDArray[np.int64]]] = None
    aecs_blocks: Optional[Union[Sequence[int], npt.NDArray[np.int64]]] = None

    # derived
    block_sizes: npt.NDArray[np.int64] = field(init=False)
    num_blocks: int = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        self._normalize_w_list_inplace()
        self._infer_structure()
        self._normalize_blocks_inplace()
        self._normalize_optional_matrices_inplace()
        self._validate()

    # --------------------
    # normalize / infer
    # --------------------
    def _normalize_w_list_inplace(self) -> None:
        if not isinstance(self.w_list, list) or len(self.w_list) == 0:
            raise ValueError("w_list must be a non-empty list.")

        for i, Wi in enumerate(self.w_list):
            if not isinstance(Wi, list) or len(Wi) == 0:
                raise ValueError(f"w_list[{i}] must be a non-empty list of blocks.")
            for b, blk in enumerate(Wi):
                arr = np.asarray(blk)
                if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
                    raise TypeError(f"w_list[{i}][{b}] must be real numeric.")
                self.w_list[i][b] = np.asarray(arr, dtype=np.float64)

    def _infer_structure(self) -> None:
        self.num_blocks = len(self.w_list[0])
        self.block_sizes = np.array(
            [int(self.w_list[0][b].shape[0]) for b in range(self.num_blocks)],
            dtype=np.int64,
        )
        self.dimension = int(self.block_sizes.sum())

    def _normalize_blocks_inplace(self) -> None:
        nb = self.num_blocks

        # vcs default: 0..nb
        if self.vcs_blocks is None or np.asarray(self.vcs_blocks, dtype=np.int64).size == 0:
            self.vcs_blocks = np.arange(0, nb, dtype=np.int64)
        else:
            self.vcs_blocks = np.unique(np.sort(np.asarray(self.vcs_blocks, dtype=np.int64).reshape(-1)))

        # aecs default: [0]
        if self.aecs_blocks is None or np.asarray(self.aecs_blocks, dtype=np.int64).size == 0:
            self.aecs_blocks = np.array([0], dtype=np.int64)
        else:
            self.aecs_blocks = np.unique(np.sort(np.asarray(self.aecs_blocks, dtype=np.int64).reshape(-1)))

    def _normalize_optional_matrices_inplace(self) -> None:
        if self.transform_matrix is not None:
            Q = np.asarray(self.transform_matrix)
            if not np.issubdtype(Q.dtype, np.number) or np.iscomplexobj(Q):
                raise TypeError("transform_matrix must be real numeric.")
            self.transform_matrix = np.asarray(Q, dtype=np.float64)

        if self.aecs_matrix is not None:
            if not isinstance(self.aecs_matrix, list):
                raise TypeError("aecs_matrix must be a list or None.")
            for b, Sb in enumerate(self.aecs_matrix):
                if Sb is None:
                    continue
                S = np.asarray(Sb)
                if not np.issubdtype(S.dtype, np.number) or np.iscomplexobj(S):
                    raise TypeError(f"aecs_matrix[{b}] must be real numeric.")
                self.aecs_matrix[b] = np.asarray(S, dtype=np.float64)

    def assemble_block(self, p: npt.ArrayLike, k: int) -> np.ndarray:
        """
        Assemble W_k(p) = sum_i p_i W_{i,k}.

        Parameters
        ----------
        p:
            Weight vector of length ``dimension``.
        k:
            Zero-based block index.
        """
        p_arr = np.asarray(p, dtype=np.float64).reshape(-1)
        n = int(self.dimension)
        if p_arr.size != n:
            raise ValueError(f"p must have length equal to dimension ({n}). Got {p_arr.size}.")

        if k < 0 or k >= self.num_blocks:
            raise ValueError(f"Block index k out of range. Got k={k}, valid 0..{self.num_blocks - 1}.")

        nk = int(self.block_sizes[k])
        Wk = np.zeros((nk, nk), dtype=np.float64)

        for i in range(n):
            pi = float(p_arr[i])
            if pi != 0.0:
                Wk += pi * np.asarray(self.w_list[i][k], dtype=np.float64)

        return 0.5 * (Wk + Wk.T)

    def wi_block_full(self, i: int) -> np.ndarray:
        """
        Return W_i as full (dimension x dimension) matrix
        in block coordinates.
        """

        if i < 0 or i >= len(self.w_list):
            raise IndexError("i out of range")

        blocks = self.w_list[i]

        # Case 1: already stored as single full matrix
        if len(blocks) == 1:
            return np.asarray(blocks[0], dtype=np.float64)
        result =  block_diag(*[np.asarray(B, dtype=np.float64) for B in blocks])

        # Case 2: stored as multiple blocks (S, I, U)
        return cast(NDArray[np.float64], result)

    def full_matrix(self, i: int) -> np.ndarray:
        """Return W_i as a full matrix in block/transformed coordinates."""
        return self.wi_block_full(i)

    def to_full_matrices(self) -> list[np.ndarray]:
        """Return all W_i as full matrices in block/transformed coordinates."""
        return [self.full_matrix(i) for i in range(self.dimension)]

    def export_matrices(
        self,
        *,
        T: float,
        use_scaling: bool,
        w_output: WOutput = "orig",
        inf_keep: str = "stable_only",
    ) -> list[np.ndarray]:
        """
        Export all W_i matrices in transformed or original coordinates.

        ``w_output="trans"`` returns block/transformed-coordinate matrices.
        ``w_output="orig"`` maps them back to original coordinates when a
        transform is available.
        """
        if w_output not in ("orig", "trans"):
            raise ValueError("w_output must be 'orig' or 'trans'.")
        if inf_keep not in ("stable_only", "all"):
            raise ValueError("inf_keep must be 'stable_only' or 'all'.")

        if w_output == "trans":
            return self.to_full_matrices()

        T_val = float(T)
        Q = self.transform_matrix
        Dinv = self.DinvFull
        Dinv_lu = None
        if use_scaling and np.isfinite(T_val) and Dinv is not None:
            Dinv_lu = lu_factor(Dinv)
        W_out: list[np.ndarray] = []

        for i in range(self.dimension):
            Wb = self.full_matrix(i)

            if not use_scaling:
                if Q is None or np.size(Q) == 0:
                    W_out.append(Wb)
                else:
                    W_out.append(Q @ Wb @ Q.T)
                continue

            if np.isfinite(T_val):
                if Dinv_lu is not None:
                    Wb = lu_solve(Dinv_lu, Wb)
                    Wb = lu_solve(Dinv_lu, Wb.T).T
                elif use_scaling and not _is_identity_matrix(Q):
                    raise ValueError(
                        "Cannot export finite-horizon scaled matrices to original coordinates: "
                        "WList.DinvFull is missing."
                    )

                if Q is None or np.size(Q) == 0:
                    W_out.append(Wb)
                else:
                    W_out.append(Q @ Wb @ Q.T)
                continue

            if inf_keep == "stable_only":
                nS = int(self.block_sizes[0]) if self.block_sizes.size > 0 else 0
                Wkeep = np.zeros((self.dimension, self.dimension), dtype=np.float64)
                if nS > 0:
                    Wkeep[:nS, :nS] = Wb[:nS, :nS]
                Wb = Wkeep

            if Q is None or np.size(Q) == 0:
                W_out.append(Wb)
            else:
                W_out.append(Q @ Wb @ Q.T)

        return W_out

    def to_original_matrices(
        self,
        *,
        T: float,
        use_scaling: bool,
        inf_keep: str = "stable_only",
    ) -> list[np.ndarray]:
        """Return all W_i matrices in original coordinates."""
        return self.export_matrices(
            T=T,
            use_scaling=use_scaling,
            w_output="orig",
            inf_keep=inf_keep,
        )

    # --------------------
    # validate (delegated)
    # --------------------
    def _validate(self) -> None:
        validate_w_list_blocks(self.w_list, self.block_sizes)

        # after normalization, these are guaranteed to be ndarray[int64]
        validate_blocks(np.asarray(self.vcs_blocks, dtype=np.int64), self.num_blocks, "vcs_blocks")
        validate_blocks(np.asarray(self.aecs_blocks, dtype=np.int64), self.num_blocks, "aecs_blocks")

        validate_transform_matrix(self.transform_matrix, self.dimension)
        validate_aecs_matrix(self.aecs_matrix, self.block_sizes)
