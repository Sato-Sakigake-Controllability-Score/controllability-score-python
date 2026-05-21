import numpy as np
import numpy.typing as npt

from scipy.linalg import expm, solve_continuous_lyapunov, eigvals, block_diag

from .wlist import WList 

from .block_diagonalization import block_diagonalization  
from ..options import WOptions


def fin_integral_noscale(A, T, wopts:WOptions) -> WList:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    steps = int(wopts.steps)
    if steps % 2 != 0:
        steps += 1
    dt = T / steps

    W = [[np.zeros((n, n), dtype=np.float64)] for _ in range(n)]

    eAT = expm(T * A)
    for i in range(n):
        Wi = W[i][0]
        Wi[i, i] += (1.0 / 3.0) * dt 
        colT = eAT[:, i:i+1]
        Wi += (1.0 / 3.0) * dt * (colT @ colT.T)

    for k in range(1, steps):
        t = k * dt
        eAt = expm(t * A)
        coeff = (4.0 / 3.0) * dt if (k % 2 == 1) else (2.0 / 3.0) * dt

        for i in range(n):
            col = eAt[:, i:i+1]
            W[i][0] += coeff * (col @ col.T)
    Id = np.eye(n, dtype=np.float64)
    return WList(
        w_list=W,
        transform_matrix=Id,
        aecs_matrix=[Id],
        w_options=wopts,
        vcs_blocks=[0],
        aecs_blocks=[0],
    )

def fin_integral_scale(A, T, wopts:WOptions) -> WList:
    """
    Python port of MATLAB: gramian/finIntegralScale_.m

    This computes the finite-time, scaled Gramian basis by Simpson integration
    in the block coordinates returned by block_diagonalization().
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    blocks, block_sizes, _, Q, Qinv = block_diagonalization(A, wopts)

    # MATLAB appears to recurse here, but the intended fallback is the unscaled
    # integral when no block transform is needed.
    if Q is None or Qinv is None or (isinstance(Q, np.ndarray) and Q.size == 0):
        return fin_integral_noscale(A, T, wopts)

    nS, nI, nU = (int(block_sizes[0]), int(block_sizes[1]), int(block_sizes[2]))

    idxS = np.arange(0, nS, dtype=int)
    idxI = np.arange(nS, nS + nI, dtype=int)
    idxU = np.arange(nS + nI, n, dtype=int)

    steps = int(wopts.steps)
    if steps % 2 != 0:
        steps += 1
    dt = T / steps
    sqrtT = np.sqrt(T)

    W = [[np.zeros((n, n), dtype=np.float64)] for _ in range(n)]

    emAUT = expm(-T * blocks[2]) if nU > 0 else np.eye(0, dtype=np.float64)

    for k in range(steps + 1):
        t = k * dt
        if k == 0 or k == steps:
            weight = (1.0 / 3.0) * dt
        elif k % 2 == 0:
            weight = (2.0 / 3.0) * dt
        else:
            weight = (4.0 / 3.0) * dt

        eASt = expm(t * blocks[0]) if nS > 0 else None
        eAIt = (expm(t * blocks[1]) / sqrtT) if nI > 0 else None
        emAUt = expm(-(T - t) * blocks[2]) if nU > 0 else None

        for i in range(n):
            Wi = W[i][0]
            Qinvi = Qinv[:, i]

            eAStQinviS = None
            eAItQinviI = None
            emAUtQinviU = None

            if nS > 0:
                QinviS = Qinvi[idxS].reshape(-1, 1)
                eAStQinviS = eASt @ QinviS
                Wi[np.ix_(idxS, idxS)] += weight * (eAStQinviS @ eAStQinviS.T)

            if nI > 0:
                QinviI = Qinvi[idxI].reshape(-1, 1)
                eAItQinviI = eAIt @ QinviI
                Wi[np.ix_(idxI, idxI)] += weight * (eAItQinviI @ eAItQinviI.T)

            if nU > 0:
                QinviU = Qinvi[idxU].reshape(-1, 1)
                emAUtQinviU = emAUt @ QinviU
                Wi[np.ix_(idxU, idxU)] += weight * (emAUtQinviU @ emAUtQinviU.T)

            if nS > 0 and nI > 0:
                Wi[np.ix_(idxS, idxI)] += weight * (eAStQinviS @ eAItQinviI.T)

            if nI > 0 and nU > 0:
                Wi[np.ix_(idxI, idxU)] += weight * (eAItQinviI @ emAUtQinviU.T)

            if nS > 0 and nU > 0:
                Wi[np.ix_(idxS, idxU)] += weight * (eAStQinviS @ emAUtQinviU.T)

    for i in range(n):
        Wi = W[i][0]
        if nS > 0 and nI > 0:
            Wi[np.ix_(idxI, idxS)] = Wi[np.ix_(idxS, idxI)].T
        if nI > 0 and nU > 0:
            Wi[np.ix_(idxU, idxI)] = Wi[np.ix_(idxI, idxU)].T
        if nS > 0 and nU > 0:
            Wi[np.ix_(idxU, idxS)] = Wi[np.ix_(idxS, idxU)].T

    DinvFull = block_diag(
        np.eye(nS, dtype=np.float64),
        (np.eye(nI, dtype=np.float64) / sqrtT) if nI > 0 else np.eye(0, dtype=np.float64),
        emAUT,
    )
    Sa0 = DinvFull @ (Qinv @ Qinv.T) @ DinvFull.T

    return WList(
        w_list=W,
        transform_matrix=np.asarray(Q, dtype=np.float64),
        aecs_matrix=[Sa0],
        DinvFull=DinvFull,
        w_options=wopts,
        vcs_blocks=[0],
        aecs_blocks=[0],
    )

def fin_lyap_noscale(A, T, wopts:WOptions) -> WList:
    """
    Finite-time Gramian computation using a Lyapunov equation (no scaling).

    For each i = 1,...,n, we compute

        W_i = ∫_0^T e^{tA} e_i e_i^T e^{tA^T} dt

    Instead of numerical integration, we use the identity that W_i satisfies
    a continuous Lyapunov equation.
    """

    # Ensure A is a real NumPy array
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    # Compute the matrix exponential e^{T A}
    eAT = expm(T * A)

    W = [[np.zeros((n, n), dtype=np.float64)] for _ in range(n)]
    
    method = wopts.method

    if method == "lyap":
        for i in range(n):
            eATi = eAT[:, i:i+1]
            rhs = -(eATi @ eATi.T)
            rhs[i, i] += 1.0
            Wi = solve_continuous_lyapunov(A, -rhs)
            W[i][0] = Wi

    elif method == "adi":
        # Placeholder for future ADI implementation
        raise NotImplementedError("ADI method is not implemented yet.")

    else:
        raise ValueError(f'Unknown Method "{method}".')
    Id = np.eye(n, dtype=np.float64)
    return WList(
        w_list=W,
        transform_matrix=Id,
        aecs_matrix=[Id],
        w_options=wopts,
        vcs_blocks=[0],
        aecs_blocks=[0],
    )


def fin_lyap_scale(A, T, wopts:WOptions) -> WList:
    """
    Python port of MATLAB: gramian/finLyapScale_.m

    This computes a finite-time Gramian basis W_i using a block-diagonalization
    (stable / imaginary-axis / unstable split) and special formulas per block,
    then stores the result in a WList along with the transform matrix Q and
    an AECS constant matrix Sa.

    Notes:
      - WList 'w_list' is a list of length n.
      - Each entry is a list of blocks; here we use nb=1 (no block structure in storage),
        so each entry is [Wi] where Wi is (n,n).
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    # Decompose A into blocks and a similarity transform:
    #   J = Q^{-1} A Q  (block diagonal, grouped by spectrum)
    # Expected outputs (matching MATLAB intent):
    #   blocks: [A_S, A_I, A_U] (may include empty blocks)
    #   block_sizes: [nS, nI, nU]
    #   Q: (n,n) or None
    #   Qinv: (n,n) or None
    blocks, block_sizes, _, Q, Qinv = block_diagonalization(A, wopts)

    # If no transform is available, fall back to the non-scaled Lyapunov method
    if Q is None or (isinstance(Q, np.ndarray) and Q.size == 0):
        return fin_lyap_noscale(A, T, wopts)

    nS, nI, nU = (int(block_sizes[0]), int(block_sizes[1]), int(block_sizes[2]))

    # Indices for each spectral region in the block-coordinates
    idxS = np.arange(0, nS, dtype=int)
    idxI = np.arange(nS, nS + nI, dtype=int)
    idxU = np.arange(nS + nI, n, dtype=int)

    # Allocate WList storage: n entries, each with one (n,n) block matrix
    W = [[np.zeros((n, n), dtype=np.float64)] for _ in range(n)]

    # Precompute exponentials that do not depend on i (when blocks exist)
    eAST = expm(T * blocks[0]) if nS > 0 else None
    emAUT = expm(-T * blocks[2]) if nU > 0 else np.eye(0, dtype=np.float64)

    sqrtT = np.sqrt(T)

    for i in range(n):
        Wi = W[i][0]  # the single block (n,n) matrix we fill

        # Qinvi is the i-th column of Qinv (block-coordinates)
        Qinvi = Qinv[:, i]

        # -------------------------
        # Left half-plane (stable)
        # -------------------------
        if nS > 0:
            QinviS = Qinvi[idxS]                      # (nS,)
            QinviS = QinviS.reshape(-1, 1)            # (nS,1)
            eASTQinviS = eAST @ QinviS                # (nS,1)

            # MATLAB:
            #   WSS = lyap(AS, QinviS*QinviS' - (e^{T AS}QinviS)(...)')
            rhsS = (QinviS @ QinviS.T) - (eASTQinviS @ eASTQinviS.T)

            # solve_continuous_lyapunov(AS, -rhsS) solves:
            #   AS X + X AS^T = -rhsS
            # which matches MATLAB lyap(AS, rhsS) in the convention:
            #   AS X + X AS^T + rhsS = 0
            Wi[np.ix_(idxS, idxS)] = solve_continuous_lyapunov(blocks[0], -rhsS)

        # --------------------------------
        # Imaginary axis (marginally stable)
        # --------------------------------
        eAIT = None  # used later for cross-terms involving the imaginary-axis block
        if nI > 0:
            QinviI = Qinvi[idxI].reshape(-1, 1)       # (nI,1)

            # Build the 2nI-by-2nI block matrix:
            #   CII = [ -AI,  QinviI*QinviI';
            #           0 ,   AI'          ]
            AI = blocks[1]
            CII = np.block([
                [-AI,                QinviI @ QinviI.T],
                [np.zeros((nI, nI)), AI.T]
            ])

            eCIIT = expm(T * CII)

            # MATLAB:
            #   eAIT = eCIIT(nI+1:2nI, nI+1:2nI).'
            eAIT = eCIIT[nI:2 * nI, nI:2 * nI].T

            # MATLAB:
            #   WII = eAIT * eCIIT(1:nI, nI+1:2nI) / T
            Wi[np.ix_(idxI, idxI)] = (eAIT @ eCIIT[0:nI, nI:2 * nI]) / T

        # --------------------------
        # Right half-plane (unstable)
        # --------------------------
        if nU > 0:
            QinviU = Qinvi[idxU].reshape(-1, 1)       # (nU,1)
            emAUTQinviU = emAUT @ QinviU              # (nU,1)

            # MATLAB:
            #   WUU = lyap(-AU, QinviU*QinviU' - (e^{-T AU}QinviU)(...)')
            rhsU = (QinviU @ QinviU.T) - (emAUTQinviU @ emAUTQinviU.T)
            Wi[np.ix_(idxU, idxU)] = solve_continuous_lyapunov(-blocks[2], -rhsU)

        # -------------
        # Cross terms
        # -------------

        # Stable (S) <-> Imaginary (I)
        if nS > 0 and nI > 0:
            QinviS = Qinvi[idxS].reshape(-1, 1)
            QinviI = Qinvi[idxI].reshape(-1, 1)

            # CIS = [ -AI,  QinviI*QinviS';
            #         0 ,   AS'         ]
            CIS = np.block([
                [-blocks[1],                      QinviI @ QinviS.T],
                [np.zeros((nS, nI), dtype=np.float64), blocks[0].T]
            ])
            eCIST = expm(T * CIS)

            # WIS = eAIT * eCIST(1:nI, nI+1:nI+nS) / sqrt(T)
            WIS = (eAIT @ eCIST[0:nI, nI:nI + nS]) / sqrtT
            Wi[np.ix_(idxI, idxS)] = WIS
            Wi[np.ix_(idxS, idxI)] = WIS.T

        # Stable (S) <-> Unstable (U)
        if nS > 0 and nU > 0:
            QinviS = Qinvi[idxS].reshape(-1, 1)
            QinviU = Qinvi[idxU].reshape(-1, 1)

            # CUS = [ -AU,  QinviU*QinviS';
            #         0 ,   AS'         ]
            CUS = np.block([
                [-blocks[2],                      QinviU @ QinviS.T],
                [np.zeros((nS, nU), dtype=np.float64), blocks[0].T]
            ])
            eCUST = expm(T * CUS)

            # WUS = eCUST(1:nU, nU+1:nU+nS)
            WUS = eCUST[0:nU, nU:nU + nS]
            Wi[np.ix_(idxU, idxS)] = WUS
            Wi[np.ix_(idxS, idxU)] = WUS.T

        # Imaginary (I) <-> Unstable (U)
        if nI > 0 and nU > 0:
            QinviI = Qinvi[idxI].reshape(-1, 1)
            QinviU = Qinvi[idxU].reshape(-1, 1)

            # CUI = [ -AU,  QinviU*QinviI';
            #         0 ,   AI'         ]
            CUI = np.block([
                [-blocks[2],                      QinviU @ QinviI.T],
                [np.zeros((nI, nU), dtype=np.float64), blocks[1].T]
            ])
            eCUIT = expm(T * CUI)

            # WUI = eCUIT(1:nU, nU+1:nU+nI) / sqrt(T)
            WUI = eCUIT[0:nU, nU:nU + nI] / sqrtT
            Wi[np.ix_(idxU, idxI)] = WUI
            Wi[np.ix_(idxI, idxU)] = WUI.T

    # Build the scaling matrix used for AECS constant Sa (in block-coordinates):
    #   DinvFull = diag( I_S, I_I / sqrt(T), expm(-T A_U) )
    DinvFull = block_diag(
        np.eye(nS, dtype=np.float64),
        (np.eye(nI, dtype=np.float64) / sqrtT) if nI > 0 else np.eye(0, dtype=np.float64),
        emAUT
    )

    # MATLAB:
    #   Sa = { DinvFull * (Qinv*Qinv') * DinvFull' }
    # In Python WList expects aecs_matrix as a list (one per block index).
    Sa0 = DinvFull @ (Qinv @ Qinv.T) @ DinvFull.T
    aecs_matrix = [Sa0]

    return WList(
        w_list=W,
        transform_matrix=np.asarray(Q, dtype=np.float64),
        aecs_matrix=aecs_matrix,
        DinvFull=DinvFull,
        w_options=wopts,
        vcs_blocks=[0],   # let WList default to "all blocks"
        aecs_blocks=[0],  # let WList default to "[0]" in your Python convention
    )



def inf_lyap_noscale(A, wopts:WOptions) -> WList:
    """
    Python port of MATLAB: gramian/infLyapNoscale_.m

    Infinite-horizon (T=inf) Gramian basis computation without scaling.

    Preconditions:
      - This path requires A to be (strictly) stable:
            Re(eig(A)) < -wopts.EigTol
        Otherwise the infinite-horizon Gramian diverges.

    For each i = 0,...,n-1 we solve the continuous Lyapunov equation:
        A W_i + W_i A^T + E_i = 0
    where E_i has a 1 at (i,i) and zeros elsewhere.

    Returns:
      - WList with nb=1 (no block structure), and no transform / AECS matrix.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    tol = wopts.eigtol

    # Check stability: all eigenvalues must satisfy Re(lambda) < -tol
    eigA = eigvals(A)
    if np.any(np.real(eigA) > -tol):
        raise ValueError(
            "computeGramian: A is unstable. "
            "T=inf without scaling requires A is stable."
        )

    # WList expects: list length n, each entry is a list of blocks.
    # Here nb=1, so each entry is [Wi] with Wi shape (n,n).
    W = [[np.zeros((n, n), dtype=np.float64)] for _ in range(n)]

    method = wopts.method

    if method == "lyap":
        for i in range(n):
            # E_i is the matrix with a single 1 at (i,i)
            Ei = np.zeros((n, n), dtype=np.float64)
            Ei[i, i] = 1.0
            Wi = solve_continuous_lyapunov(A, -Ei)
            W[i][0] = Wi

    elif method == "adi":
        raise NotImplementedError("ADI method is not implemented yet.")

    else:
        raise ValueError(f'Unknown Method "{method}".')
    Id = np.eye(n, dtype=np.float64)
    return WList(
        w_list=W,
        transform_matrix=Id,
        aecs_matrix=[Id],
        w_options=wopts,
        vcs_blocks=[0],   # let WList default to "all blocks"
        aecs_blocks=[0],  # let WList default to "[0]" in your Python convention
    )



def inf_lyap_scale(A, wopts:WOptions) -> WList:
    """
    Python port of MATLAB: gramian/infLyapScale_.m

    Infinite-horizon Gramian basis computation WITH scaling (stable/imag/unstable split).

    It block-diagonalizes A into (A_S, A_I, A_U) via a similarity transform:
        J = Q^{-1} A Q
    and for each i builds blocks using Lyapunov solves in the corresponding subspaces.

    Notes:
      - Unlike the 'noscale' version, w_list uses multiple blocks (nb can be 1..3)
        depending on which spectral regions exist.
      - WList in your project uses 0-based block indices internally, but we pass
        vcs_blocks/aecs_blocks as None so defaults apply safely.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    blocks, block_sizes, _, Q, Qinv = block_diagonalization(A, wopts)
    if Q is None or Qinv is None or (isinstance(Q, np.ndarray) and Q.size == 0):
        wopts.with_use_scaling(False)
        return inf_lyap_noscale(A, wopts)
    nS, nI, nU = (int(block_sizes[0]), int(block_sizes[1]), int(block_sizes[2]))

    # MATLAB uses 1-based index ranges; Python uses 0-based slices
    idxS = slice(0, nS)
    idxI = slice(nS, nS + nI)
    idxU = slice(nS + nI, nS + nI + nU)

    AS = blocks[0]  # (nS,nS) or (0,0)
    AI = blocks[1]  # (nI,nI) or (0,0)
    AU = blocks[2]  # (nU,nU) or (0,0)

    # Build W: list length n, each entry is a list of blocks.
    # Number of blocks per i depends on which idx ranges are non-empty.
    W: list[list[npt.NDArray[np.float64]]] = [[] for _ in range(n)]

    # MATLAB hack: lyap(blocks{2} - 1e-8*I, ...)
    # This is a numerical regularization to avoid singular Lyapunov on the imaginary axis.
    imag_shift = 1e-8

    for i in range(n):
        Qinvi = Qinv[:, i]  # block-coordinates column vector (length n)

        # ---- left half plane (stable) ----
        if nS > 0:
            QinviS = Qinvi[idxS].reshape(-1, 1)
            rhsS = QinviS @ QinviS.T

            # Solve: AS X + X AS^T + rhsS = 0
            W[i].append(solve_continuous_lyapunov(AS,- rhsS))

        # ---- imaginary axis ----
        if nI > 0:
            QinviI = Qinvi[idxI].reshape(-1, 1)
            rhsI = QinviI @ QinviI.T

            # Pure imaginary eigenvalues can make the Lyapunov operator singular.
            # MATLAB code applies a small negative real shift to stabilize the solve.
            AI_reg = AI - imag_shift * np.eye(nI, dtype=np.float64)

            # Solve: AI_reg X + X AI_reg^T + rhsI = 0
            W[i].append(solve_continuous_lyapunov(AI_reg,- rhsI))

            # If you want to match the commented MATLAB alternative:
            # W[i].append(rhsI)

        # ---- right half plane (unstable) ----
        if nU > 0:
            QinviU = Qinvi[idxU].reshape(-1, 1)
            rhsU = QinviU @ QinviU.T

            # Use -AU to make it stable for the Lyapunov solve:
            # (-AU) X + X (-AU)^T + rhsU = 0
            W[i].append(solve_continuous_lyapunov(-AU, -rhsU))

    # MATLAB: Sa = cell(size(W{1})) -> a list of Nones (same length as blocks)
    # Your WList expects aecs_matrix as Optional[List[Optional[np.ndarray]]]
    G = Qinv @ Qinv.T

    aecs_matrix = []
    if nS > 0:
        aecs_matrix.append(G[idxS, idxS])                    # S block
    if nI > 0:
        aecs_matrix.append(np.zeros((nI, nI), dtype=float))  # I block = 0
    if nU > 0:
        aecs_matrix.append(np.zeros((nU, nU), dtype=float))  # U block = 0
    num_blocks = len(W[0])
    vcs_blocks = list(range(num_blocks))
    return WList(
        w_list=W,
        transform_matrix=np.asarray(Q, dtype=np.float64) if Q is not None else None,
        aecs_matrix=aecs_matrix,
        w_options=wopts,
        vcs_blocks=vcs_blocks,   # defaults to all blocks in your Python WList
        aecs_blocks=[0],  # defaults to [0] in your Python WList
    )
