import numpy as np

from controllability_scoring.gramian.wlist import WList


def test_assemble_block_matches_weighted_sum():
    W = [
        [np.array([[1.0, 0.2], [0.2, 2.0]])],
        [np.array([[3.0, 0.4], [0.1, 4.0]])],
    ]
    wlist = WList(W)

    p = np.array([0.25, 0.75])
    Wk = wlist.assemble_block(p, 0)

    expected = p[0] * W[0][0] + p[1] * W[1][0]
    expected = 0.5 * (expected + expected.T)
    assert np.allclose(Wk, expected)


def test_to_full_matrices_builds_block_diagonal_matrices():
    W = [
        [np.array([[1.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])],
        [np.array([[4.0]]), np.array([[5.0, 0.0], [0.0, 6.0]])],
        [np.array([[7.0]]), np.array([[8.0, 0.0], [0.0, 9.0]])],
    ]
    wlist = WList(W)

    full = wlist.to_full_matrices()

    assert len(full) == 3
    assert np.allclose(full[0], np.diag([1.0, 2.0, 3.0]))
    assert np.allclose(full[1], np.diag([4.0, 5.0, 6.0]))
    assert np.allclose(full[2], np.diag([7.0, 8.0, 9.0]))


def test_export_matrices_trans_returns_block_coordinate_full_matrices():
    W = [
        [np.array([[1.0]]), np.array([[2.0]])],
        [np.array([[3.0]]), np.array([[4.0]])],
    ]
    wlist = WList(W, transform_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]))

    exported = wlist.export_matrices(T=1.0, use_scaling=False, w_output="trans")

    assert np.allclose(exported[0], np.diag([1.0, 2.0]))
    assert np.allclose(exported[1], np.diag([3.0, 4.0]))


def test_export_matrices_orig_applies_transform_without_scaling():
    W = [
        [np.array([[1.0]]), np.array([[2.0]])],
        [np.array([[3.0]]), np.array([[4.0]])],
    ]
    Q = np.array([[0.0, 1.0], [1.0, 0.0]])
    wlist = WList(W, transform_matrix=Q)

    exported = wlist.export_matrices(T=1.0, use_scaling=False, w_output="orig")

    assert np.allclose(exported[0], Q @ np.diag([1.0, 2.0]) @ Q.T)
    assert np.allclose(exported[1], Q @ np.diag([3.0, 4.0]) @ Q.T)


def test_export_matrices_orig_applies_scaling_via_dinv():
    W = [
        [np.array([[1.0, 0.2], [0.2, 2.0]])],
        [np.array([[3.0, 0.1], [0.1, 4.0]])],
    ]
    Q = np.array([[1.0, 0.5], [0.0, 1.0]])
    Dinv = np.array([[2.0, 0.3], [0.0, 4.0]])
    wlist = WList(W, transform_matrix=Q, DinvFull=Dinv)

    exported = wlist.export_matrices(T=1.0, use_scaling=True, w_output="orig")

    D = np.linalg.inv(Dinv)
    assert np.allclose(exported[0], Q @ (D @ W[0][0] @ D.T) @ Q.T)
    assert np.allclose(exported[1], Q @ (D @ W[1][0] @ D.T) @ Q.T)


def test_export_matrices_orig_rejects_missing_dinv_for_transformed_finite_scaling():
    W = [
        [np.array([[1.0, 0.0], [0.0, 2.0]])],
        [np.array([[3.0, 0.0], [0.0, 4.0]])],
    ]
    Q = np.array([[1.0, 0.5], [0.0, 1.0]])
    wlist = WList(W, transform_matrix=Q)

    try:
        wlist.export_matrices(T=1.0, use_scaling=True, w_output="orig")
        assert False, "Expected ValueError for missing DinvFull"
    except ValueError:
        pass


def test_export_matrices_orig_allows_identity_fallback_without_dinv():
    W = [
        [np.array([[1.0, 0.0], [0.0, 2.0]])],
        [np.array([[3.0, 0.0], [0.0, 4.0]])],
    ]
    wlist = WList(W, transform_matrix=np.eye(2))

    exported = wlist.export_matrices(T=1.0, use_scaling=True, w_output="orig")

    assert np.allclose(exported[0], W[0][0])
    assert np.allclose(exported[1], W[1][0])


def test_export_matrices_rejects_invalid_options():
    W = [
        [np.array([[1.0]])],
    ]
    wlist = WList(W)

    try:
        wlist.export_matrices(T=1.0, use_scaling=False, w_output="bad")
        assert False, "Expected ValueError for invalid w_output"
    except ValueError:
        pass

    try:
        wlist.export_matrices(T=np.inf, use_scaling=True, inf_keep="bad")
        assert False, "Expected ValueError for invalid inf_keep"
    except ValueError:
        pass
