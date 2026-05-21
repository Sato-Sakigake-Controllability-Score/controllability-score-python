import numpy as np
from scipy.linalg import schur, rsf2csf, solve_sylvester, block_diag


def block_diagonalization(A, wopts):
    """
    Block-diagonalize A (up to similarity) into stable / imaginary-axis / unstable parts.

    This version avoids ordschur (not available in some SciPy versions) by using
    scipy.linalg.schur(sort=...) in two stages to achieve 3-way ordering:
      1) [stable|imag] vs [unstable]
      2) within [stable|imag], [stable] vs [imag]
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a real square 2D array.")
    n = A.shape[0]

    tol = float(wopts.eigtol)
    if tol <= 0:
        raise ValueError("wopts.eigtol must be positive.")

    # --- Real Schur of A.T: A.T = U T U^T ---
    # If sort is not None, SciPy returns (T, U, sdim) or (T, U) depending on version.
    # We handle both patterns below.
    def _schur_real_sorted(M, selector):
        out = schur(M, output="real", sort=selector)
        if len(out) == 3:
            Tm, Um, sdim = out
        else:
            # Older SciPy: may return only (T, U); but then no sorting happened.
            # In that case you must upgrade SciPy. We fail loudly.
            raise ImportError(
                "Your SciPy schur() does not support 'sort'. "
                "Please upgrade SciPy (schur(sort=...) is required)."
            )
        return Tm, Um, int(sdim)

    # Use complex Schur to classify eigenvalues robustly
    # U0, T0 = schur(A.T, output="real")  # unsorted, always available
    T0, U0 = schur(A.T, output="real")  # unsorted, always available

    Tc0, _ = rsf2csf(T0, U0)

    eig0 = np.diag(Tc0)

    isS = np.real(eig0) < -tol
    isU = np.real(eig0) > tol
    isI = ~(isS | isU)

    nS = int(np.count_nonzero(isS))
    nI = int(np.count_nonzero(isI))
    nU = int(np.count_nonzero(isU))

    if nS + nI + nU != n:
        raise RuntimeError("block_diagonalization: eigenvalue classification failed.")

    # --- If strictly stable, keep MATLAB behavior: no transform ---
    if nS == n:
        Ls = A
        Li = np.zeros((0, 0), dtype=np.float64)
        Lu = np.zeros((0, 0), dtype=np.float64)
        blocks = [Ls, Li, Lu]
        block_sizes = np.array([nS, 0, 0], dtype=np.int64)
        eigA = eig0
        Q = None
        Qinv = None
        return blocks, block_sizes, eigA, Q, Qinv

    # --- Stage 1: sort [stable|imag] first (i.e., "not unstable") ---
    def sel_stable_or_imag(x):
        r = np.real(x)
        return (r <= tol)  # <= tol means stable or imag region, since unstable is r > tol

    T1, U1, sdim1 = _schur_real_sorted(A.T, sel_stable_or_imag)
    # After stage1, leading block has size sdim1 = nS+nI (in exact arithmetic)
    # but numerical ties near tol can shift it; still we treat it as sdim1.
    k = sdim1  # size of [S|I] cluster

    # --- Stage 2: within the leading kxk block, sort stable first ---
    if k > 0:
        T11 = T1[:k, :k]

        def sel_stable(x):
            return np.real(x) < -tol

        T11s, U11s, sdimS = _schur_real_sorted(T11, sel_stable)

        # Embed the second transform into the full space:
        # U = U1 @ block_diag(U11s, I)
        U = U1 @ block_diag(U11s, np.eye(n - k, dtype=np.float64))

        # Apply similarity to T1:
        # T = G^T T1 G, where G = block_diag(U11s, I)
        G = block_diag(U11s, np.eye(n - k, dtype=np.float64))
        T = G.T @ T1 @ G
    else:
        U, T = U1, T1

    # Eigenvalues of reordered Schur form
    Tc, _ = rsf2csf(T, U) 
    eigA = np.diag(Tc)

    # Recompute counts from final eigA (more consistent with numeric sorting)
    isS = np.real(eigA) < -tol
    isU = np.real(eigA) > tol
    isI = ~(isS | isU)
    nS = int(np.count_nonzero(isS))
    nI = int(np.count_nonzero(isI))
    nU = int(np.count_nonzero(isU))

    # --- Make quasi-lower triangular (MATLAB: L = L.' ) ---
    L = T.T

    # --- Indices (0-based) ---
    idxS = slice(0, nS)
    idxI = slice(nS, nS + nI)
    idxU = slice(nS + nI, n)

    # --- Extract diagonal blocks ---
    Ls = L[idxS, idxS] if nS > 0 else np.zeros((0, 0), dtype=np.float64)
    Li = L[idxI, idxI] if nI > 0 else np.zeros((0, 0), dtype=np.float64)
    Lu = L[idxU, idxU] if nU > 0 else np.zeros((0, 0), dtype=np.float64)

    # --- Coupling blocks (sub-diagonal in L) ---
    Lis = L[idxI, idxS] if (nI > 0 and nS > 0) else np.zeros((nI, nS), dtype=np.float64)
    Lus = L[idxU, idxS] if (nU > 0 and nS > 0) else np.zeros((nU, nS), dtype=np.float64)
    Lui = L[idxU, idxI] if (nU > 0 and nI > 0) else np.zeros((nU, nI), dtype=np.float64)

    # --- Solve Sylvester equations ---
    Sis = np.zeros((nI, nS), dtype=np.float64)
    Sui = np.zeros((nU, nI), dtype=np.float64)
    Sus = np.zeros((nU, nS), dtype=np.float64)

    if nI > 0 and nS > 0:
        Sis = solve_sylvester(Li, -Ls, -Lis)

    if nU > 0 and nI > 0:
        Sui = solve_sylvester(Lu, -Li, -Lui)

    if nU > 0 and nS > 0:
        rhs = -Lus if nI == 0 else -(Lus + Lui @ Sis)
        Sus = solve_sylvester(Lu, -Ls, rhs)

    # --- Build S (lower block unit triangular) ---
    S = np.eye(n, dtype=np.float64)
    if nI > 0 and nS > 0:
        S[idxI, idxS] = Sis
    if nU > 0 and nS > 0:
        S[idxU, idxS] = Sus
    if nU > 0 and nI > 0:
        S[idxU, idxI] = Sui

    Sinv = np.linalg.solve(S, np.eye(n, dtype=np.float64))

    blocks = [Ls, Li, Lu]
    block_sizes = np.array([nS, nI, nU], dtype=np.int64)
    Q = U @ S
    Qinv = Sinv @ U.T

    return blocks, block_sizes, eigA, Q, Qinv