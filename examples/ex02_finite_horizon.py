"""Compare VCS and AECS for several finite time horizons T.

This script compares how scores change with the terminal time.
"""

import numpy as np

from controllability_scoring import cs


def print_weight_table(title: str, values: np.ndarray, horizons: np.ndarray) -> None:
    print(title)
    header = "node " + " ".join(f"T_{t:.1f}".rjust(10) for t in horizons)
    print(header)
    for i, row in enumerate(values, start=1):
        print(f"{i:4d} " + " ".join(f"{x:10.6f}" for x in row))


def main() -> None:
    A = np.array(
        [
            [-0.6, 0.2, 0.0, 0.0],
            [0.0, -0.9, 0.3, 0.0],
            [0.1, 0.0, 0.2, 0.4],
            [0.0, 0.0, 0.1, 0.7],
        ],
        dtype=float,
    )

    horizons = np.array([0.5, 2.0, 5.0])
    p_v_all = np.zeros((A.shape[0], horizons.size))
    p_a_all = np.zeros((A.shape[0], horizons.size))

    for k, horizon in enumerate(horizons):
        p_v_all[:, k], p_a_all[:, k], _, _, _, _ = cs(A, horizon)

    print_weight_table("Finite-horizon VCS weights:", p_v_all, horizons)
    print()
    print_weight_table("Finite-horizon AECS weights:", p_a_all, horizons)


if __name__ == "__main__":
    main()
