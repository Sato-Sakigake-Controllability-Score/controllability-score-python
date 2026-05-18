import numpy as np

from cs.gramian import compute_w
from cs.options import WOptions


def test_compute_w_stub_contract_shapes():
    A = np.eye(4)
    T = np.inf
    wopt = WOptions.from_system(A, T)

    WList = compute_w(A, T, wopt)

    n = A.shape[0]

    # BlockSizes
    assert WList.block_sizes.ndim == 1
    assert int(WList.block_sizes.sum()) == n
    B = int(WList.block_sizes.size)
    assert B >= 1

    # Indices (MATLAB-style: 1-based)
    assert WList.vcs_blocks.ndim == 1 and WList.vcs_blocks.size >= 1
    assert WList.aecs_blocks.ndim == 1 and WList.aecs_blocks.size >= 1
    assert np.all((WList.vcs_blocks >= 0) & (WList.vcs_blocks < B))
    assert np.all((WList.aecs_blocks >= 0) & (WList.aecs_blocks < B))

    # Transform matrix P
    assert WList.transform_matrix.shape == (n, n)

    # AECS matrix shape
    nA = int(WList.block_sizes[(WList.aecs_blocks - 1)].sum())
    assert isinstance(WList.aecs_matrix, list)
    assert sum(M.shape[0] for M in WList.aecs_matrix) == nA
    for M in WList.aecs_matrix:
        assert isinstance(M, np.ndarray)
        assert M.ndim == 2
        assert M.shape[0] == M.shape[1]

    # WList: length n, each entry is a list of B blocks
    assert isinstance(WList.w_list, list)
    assert len(WList.w_list) == n

    for i in range(n):
        wi = WList.w_list[i]
        assert isinstance(wi, list)

        blocks = wi
        assert isinstance(blocks, list)
        assert len(blocks) == B

        for k in range(B):
            nk = int(WList.block_sizes[k])
            assert np.asarray(blocks[k]).shape == (nk, nk)

