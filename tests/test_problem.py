import numpy as np

from cs.problem import CSProblem
from cs.options import WOptions


def test_csproblem_constructs_with_defaults():
    A = np.eye(3)
    prob = CSProblem(A)

    assert prob.dimension == 3
    assert prob.T == float("inf")

    # assert prob.initial_guess.shape == (3,)
    # assert np.allclose(prob.initial_guess, np.ones(3) / 3)

    # computed fields exist
    assert int(prob.wlist.block_sizes.sum()) == prob.dimension
    assert prob.wlist.transform_matrix.shape == (3, 3)


def test_csproblem_accepts_woptions():
    A = np.eye(5)
    wopt = WOptions.from_system(A, np.inf, method="lyap")
    prob = CSProblem(A, w_options=wopt)
    assert prob.w_options.method == "lyap"


def test_csproblem_wlist_contract():
    A = np.eye(3)
    prob = CSProblem(A)

    n = prob.dimension
    B = int(prob.wlist.block_sizes.size)

    assert isinstance(prob.wlist.w_list, list)
    assert len(prob.wlist.w_list) == n

    for i in range(n):
        wi = prob.wlist.w_list[i]
        assert isinstance(wi, list), f"w_list[{i}] must be dict"

        blocks = wi
        assert isinstance(blocks, list)
        assert len(blocks) == B

        for k in range(B):
            blk = np.asarray(blocks[k])
            assert blk.shape == (prob.wlist.block_sizes[k], prob.wlist.block_sizes[k])
