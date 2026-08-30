"""Compute VCS and AECS from an adjacency matrix and visualize the scores."""

from __future__ import annotations

import argparse
import os
import tempfile
from math import pi

import numpy as np

from controllability_scoring import cs
from controllability_scoring.options import WOptions


def print_ranking(title: str, weights: np.ndarray) -> None:
    order = np.argsort(-weights)
    print(title)
    print("Node  Weight")
    for idx in order:
        print(f"{idx + 1:4d}  {weights[idx]:.6f}")


def _load_pyplot(*, save: bool):
    try:
        cache_root = os.path.join(tempfile.gettempdir(), "controllability_scoring_matplotlib")
        os.makedirs(cache_root, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "config"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_root, "cache"))

        import matplotlib

        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
    except ModuleNotFoundError:
        return None, None, None
    return plt, Circle, FancyArrowPatch


def plot_scores(adjacency: np.ndarray, p_v: np.ndarray, p_a: np.ndarray, *, save: str | None) -> bool:
    plt, Circle, FancyArrowPatch = _load_pyplot(save=save is not None)
    if plt is None or Circle is None or FancyArrowPatch is None:
        print()
        print("Visualization skipped: matplotlib is not installed.")
        print('Install it with: python -m pip install -e ".[examples]"')
        return False

    n = adjacency.shape[0]
    theta = np.linspace(pi / 2, pi / 2 + 2 * pi, n, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta)])

    fig, (ax_graph, ax_scores) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Controllability Scores", fontsize=14)

    ax_graph.set_title("Directed adjacency graph")
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")
    ax_graph.set_xlim(-1.35, 1.35)
    ax_graph.set_ylim(-1.35, 1.35)

    node_radius = 0.11
    for src in range(n):
        for dst in range(n):
            weight = adjacency[src, dst]
            if weight == 0:
                continue
            start = positions[src]
            end = positions[dst]
            delta = end - start
            length = np.linalg.norm(delta)
            if length == 0:
                continue
            direction = delta / length
            arrow_start = start + node_radius * direction
            arrow_end = end - node_radius * direction
            arrow = FancyArrowPatch(
                arrow_start,
                arrow_end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#386cb0",
                connectionstyle="arc3,rad=0.08",
            )
            ax_graph.add_patch(arrow)
            label_position = 0.5 * (arrow_start + arrow_end)
            ax_graph.text(
                label_position[0],
                label_position[1],
                f"{weight:.2g}",
                ha="center",
                va="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
            )

    for idx, (x, y) in enumerate(positions, start=1):
        circle = Circle((x, y), node_radius, facecolor="white", edgecolor="black", linewidth=1.5, zorder=3)
        ax_graph.add_patch(circle)
        ax_graph.text(x, y, str(idx), ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

    ax_scores.set_title("Node weights")
    nodes = np.arange(1, n + 1)
    width = 0.36
    ax_scores.bar(nodes - width / 2, p_v, width, label="VCS", color="#4daf4a")
    ax_scores.bar(nodes + width / 2, p_a, width, label="AECS", color="#984ea3")
    ax_scores.set_xlabel("node")
    ax_scores.set_ylabel("weight")
    ax_scores.set_xticks(nodes)
    ax_scores.set_ylim(0.0, max(float(np.max(p_v)), float(np.max(p_a))) * 1.2)
    ax_scores.legend()
    ax_scores.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=160, bbox_inches="tight")
        print()
        print(f"Saved visualization to: {save}")
    else:
        plt.show()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save the visualization to an image file instead of opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = 5
    self_loop_weight = -1.0
    T = 2.0

    adjacency = np.array(
        [
            [0.0, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.7],
            [0.4, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    A = adjacency + self_loop_weight * np.eye(n)
    w_options = WOptions.from_system(A, T, use_scaling=False)

    p_v, p_a, _, _, _, _ = cs(A, T, w_options=w_options)

    print("Adjacency matrix:")
    print(adjacency)

    print("System matrix A = adjacency + self_loop_weight * eye(N):")
    print(A)

    print("Scores:")
    print("Node  VCS       AECS")
    for idx, (v_score, a_score) in enumerate(zip(p_v, p_a), start=1):
        print(f"{idx:4d}  {v_score:.6f}  {a_score:.6f}")

    print()
    print_ranking("VCS ranking:", p_v)
    print()
    print_ranking("AECS ranking:", p_a)
    plot_scores(adjacency, p_v, p_a, save=args.save)


if __name__ == "__main__":
    main()
