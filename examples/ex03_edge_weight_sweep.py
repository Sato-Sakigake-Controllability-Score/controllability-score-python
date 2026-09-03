"""Score changes when the weight of one edge is varied.

This script varies one entry of the system matrix and records score changes.
"""

import numpy as np

from controllability_scoring import cs
from controllability_scoring.options import WOptions


def main() -> None:
    n = 5
    A0 = np.array(
        [
            [-0.8, 0.0, 0.1, 0.0, 0.0],
            [0.2, -0.7, 0.0, 0.0, 0.0],
            [0.0, 0.3, -0.5, 0.2, 0.0],
            [0.0, 0.0, 0.0, -0.4, 0.3],
            [0.1, 0.0, 0.0, 0.0, -0.6],
        ],
        dtype=float,
    )

    edge = (2, 4)
    weights = np.linspace(0.0, 1.5, 40)
    T = 2.0

    p_v_history = np.zeros((n, weights.size))
    p_a_history = np.zeros((n, weights.size))
    w_options = WOptions.from_system(A0, T, use_scaling=False)

    for k, weight in enumerate(weights):
        A = A0.copy()
        A[edge] = weight
        p_v_history[:, k], p_a_history[:, k], _, _, _, _ = cs(
            A,
            T,
            w_options=w_options,
        )

    print("Edge-weight sweep for A[2, 4]")
    print("weights:", np.round(weights, 4).tolist())
    print("Final VCS weights:", np.round(p_v_history[:, -1], 6).tolist())
    print("Final AECS weights:", np.round(p_a_history[:, -1], 6).tolist())


if __name__ == "__main__":
    main()
