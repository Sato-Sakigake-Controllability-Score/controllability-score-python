"""Basic example for computing VCS and AECS with the Python API."""

import numpy as np

from controllability_scoring import aecs, cs, vcs


def print_weights(name: str, weights: np.ndarray) -> None:
    print(f"{name} weights:")
    print(np.array2string(weights, precision=6, suppress_small=True))
    print(f"{name} simplex sum: {weights.sum():.12f}")
    print()


def print_solver_summary(info_v, info_a) -> None:
    print("Solver summary:")
    print(f"{'Score':<6} {'ObjectiveValue':>16} {'Iterations':>10} {'Converged':>10} {'ExitFlag':>8}")
    for name, info in (("VCS", info_v), ("AECS", info_a)):
        print(
            f"{name:<6} "
            f"{info.objective_value:16.8e} "
            f"{info.iterations:10d} "
            f"{str(info.converged):>10} "
            f"{info.exit_flag:8d}"
        )


def main() -> None:
    A = np.diag([-1.0, -2.0, 0.5, 1.2])

    p_v, p_a, info_v, info_a, _, _ = cs(A)
    p_v_separate, _, _, _ = vcs(A)
    p_a_separate, _, _, _ = aecs(A)

    print("System matrix A:")
    print(A)
    print()

    print_weights("VCS", p_v)
    print_weights("AECS", p_a)

    print_solver_summary(info_v, info_a)
    print()

    print("Consistency check:")
    print(f"  ||cs(A).p_v - vcs(A).p||  = {np.linalg.norm(p_v - p_v_separate):.3e}")
    print(f"  ||cs(A).p_a - aecs(A).p|| = {np.linalg.norm(p_a - p_a_separate):.3e}")


if __name__ == "__main__":
    main()
