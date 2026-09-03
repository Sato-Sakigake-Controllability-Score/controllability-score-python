# Usage Examples

<table>
    <thead>
        <tr>
            <th style="text-align:center">English</th>
            <th style="text-align:center"><a href="README.md">日本語</a></th>
        </tr>
    </thead>
</table>

This README is a guide for checking, step by step, how to use controllability score with Python code.

The files from `ex01_*.py` to `ex05_*.py` are provided as reference scripts
that run the basic workflows of this package in an organized form.

## Obtaining the Repository

First, move to any working directory in the terminal and clone the repository from GitHub.

```sh
git clone https://github.com/Sato-Sakigake-Controllability-Score/controllability-score-python.git
cd controllability-score-python
```

Below, this `controllability-score-python` directory is referred to as the root directory of the repository.

## Prerequisites

- Python 3.10 or later
- NumPy
- SciPy
- pytest, when running tests or developing

For source installation, run the following from the root directory.

```sh
python -m pip install -e .
```

For development dependencies, run:

```sh
python -m pip install -e ".[dev]"
```

Visualization dependencies are included in the development dependencies.

Run sample scripts from the root directory as follows.

```sh
python examples/ex01_minimal_cs.py
```

## 1. Prepare a System Matrix

The basic API takes a system matrix `A` and returns weights for each node.

```python
import numpy as np

A = np.diag([-1.0, -2.0, 0.5, 1.2])
```

In general, it is sufficient for `A` to be a square matrix whose rows and columns correspond to nodes, and the following variations are possible.

- Use the adjacency matrix `A` of a network directly as the system matrix
- For stabilization, add diagonal components and use a form such as `A - cI`
- Compute the Laplacian `L = D - A` and use `-L` as the system matrix

```python
adjacency = np.array(
    [
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.4],
        [0.2, 0.0, 0.0],
    ],
    dtype=float,
)

self_loop_weight = 2.0
A1 = adjacency - self_loop_weight * np.eye(adjacency.shape[0])

degree = adjacency.sum(axis=1)
L = np.diag(degree) - adjacency
A2 = -L
```

## 2. Treat an Adjacency Matrix as a Directed Graph

The minimum dependencies of the Python package are NumPy and SciPy, so the examples do not require a graph drawing library.
If you want to inspect the network structure, print the adjacency matrix and pass it to `cs`.

```python
adjacency = np.array(
    [
        [0.0, 0.8, 0.3, 0.0],
        [0.0, 0.3, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.6],
        [0.3, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)

print(adjacency)
```

This minimal example is collected in [`ex05_minimal_graph_cs.py`][ex05].

## 3. Compute VCS and AECS Simultaneously

When using both VCS and AECS, the basic approach is to use `cs`.

```python
from controllability_scoring import cs

p_v, p_a, info_v, info_a, wlist, transform = cs(A)
```

`p_v` and `p_a` are weight vectors for each node.

```python
print("VCS weights:")
print(p_v)

print("AECS weights:")
print(p_a)
```

If only VCS or only AECS is needed, the individual functions can also be used.

```python
from controllability_scoring import aecs, vcs

p_v, info_v, wlist_v, transform = vcs(A)
p_a, info_a, wlist_a, transform = aecs(A)
```

However, `cs(A)` computes `W_1, ..., W_n` all at once at the beginning of the algorithm and then computes VCS and AECS based on them. Therefore, when using both, calling `cs(A)` is more efficient than calling `vcs(A)` and `aecs(A)` sequentially.

## 4. Check Solver Information

If you want to check not only the computed weights but also the state of optimization, inspect `info_v` and `info_a`.

```python
p_v, p_a, info_v, info_a, wlist, transform = cs(A)
```

For example, `info_v` is a `CSResult` object and has attributes such as the following.

```python
print(info_v.objective_value)
print(info_v.iterations)
print(info_v.converged)
print(info_v.exit_flag)
print(info_v.exit_message)
```

The same kind of `info` can also be received from individual functions.

```python
p_v, info_v, wlist_v, transform = vcs(A)
p_a, info_a, wlist_a, transform = aecs(A)
```

## 5. Specify a Finite Time Horizon

By default, `T = inf` is used, but if you want to compute finite-time scores, specify the terminal time as the second argument or with `T=...`.

```python
T = 2.0
p_v, p_a, info_v, info_a, wlist, transform = cs(A, T)
```

The same specification can also be made with a keyword argument.

```python
p_v, p_a, info_v, info_a, wlist, transform = cs(A, T=2.0)
```

## 6. Specify Other Options

In Python, Gramian options are specified with `WOptions`, and solver options are specified with `PGSolverOptions`.

- Main specifications on the Gramian side: `method`, `steps`, `use_scaling`
- Main specifications on the solver side: `max_iter`, `tol`, `verbose`, `store_trace`

For example, the following shows an example in which options are specified together.

```python
from controllability_scoring import cs
from controllability_scoring.options import PGSolverOptions, WOptions

T = 2.0
w_options = WOptions.from_system(
    A,
    T,
    method="integral",
    steps=80,
    use_scaling=False,
)
solver_options = PGSolverOptions(
    max_iter=2000,
    tol=1e-9,
    verbose=True,
    store_trace=True,
)

p_v, p_a, info_v, info_a, wlist, transform = cs(
    A,
    T,
    w_options=w_options,
    solver_options=solver_options,
)
```

For details of the options, see [README_eng.md][repo-readme] in the repository root.

## 7. Run Sample Scripts

If you want to run organized examples, use the following scripts. Each section of the README can also be used as a guide when reading these scripts.

| File | Contents |
| --- | --- |
| [`ex01_minimal_cs.py`][ex01] | Basics of `vcs`, `aecs`, and `cs`, and checking solver information |
| [`ex02_finite_horizon.py`][ex02] | Score comparison using multiple finite time horizons `T` |
| [`ex03_edge_weight_sweep.py`][ex03] | Score changes when changing the weight of a single edge |
| [`ex04_visualize_scores.py`][ex04] | Compute VCS and AECS from an adjacency matrix and visualize the graph and scores |
| [`ex05_minimal_graph_cs.py`][ex05] | Display an adjacency matrix and compute VCS and AECS from the same matrix |

`ex04_visualize_scores.py` can also save the visualization to an image file.

```sh
python examples/ex04_visualize_scores.py --save examples/ex04_scores.png
```


[repo-readme]: ../README_eng.md
[ex01]: ./ex01_minimal_cs.py
[ex02]: ./ex02_finite_horizon.py
[ex03]: ./ex03_edge_weight_sweep.py
[ex04]: ./ex04_visualize_scores.py
[ex05]: ./ex05_minimal_graph_cs.py
