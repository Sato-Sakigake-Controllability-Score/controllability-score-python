"""Minimal graph-style example for VCS and AECS.

This script uses an adjacency matrix directly as a system matrix.
"""

import numpy as np

from controllability_scoring import cs


def main() -> None:
    adjacency = np.array(
        [
            [0.0, 0.8, 0.3, 0.0],
            [0.0, 0.3, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.6],
            [0.3, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    p_v, p_a, _, _, _, _ = cs(adjacency)

    print("Adjacency matrix:")
    print(adjacency)

    print("VCS weights:")
    print(p_v)

    print("AECS weights:")
    print(p_a)


if __name__ == "__main__":
    main()
