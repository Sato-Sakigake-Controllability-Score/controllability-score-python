# Controllability Score Computation Program for Python

<table>
  <thead>
    <tr>
      <th style="text-align:center">English</th>
      <th style="text-align:center"><a href="README.md">日本語</a></th>
    </tr>
  </thead>
</table>

## 1. Introduction

This project is a Python package for computing controllability scores. It mainly
takes a system matrix `A` and a terminal time `T` as inputs and outputs
controllability scores (VCS, AECS).

VCS and AECS can be computed separately. When both scores are needed, `cs`
computes them from one shared problem setting.

### Interpretation of Controllability and Observability Scores

The controllability score introduces a virtual framework in which state nodes
and input nodes correspond one-to-one, and quantitatively evaluates which nodes
should be intervened in, and with what weights, in order to control the whole
system effectively.

As a dual concept, the observability score indicates which state nodes should be
observed in the system

```math
\frac{dx}{dt}=Ax
```

to make it easier to understand the whole state. In Python, it can be computed
by passing `A.T` instead of `A`.

```python
p_obs_v, info_v, wlist_v, transform = vcs(A.T)
p_obs_a, info_a, wlist_a, transform = aecs(A.T)
p_obs_v, p_obs_a, info_v, info_a, wlist, transform = cs(A.T)
```

### Operating Environment

- Python: 3.10 or later
- Main dependencies: NumPy, SciPy
- Test dependency: pytest

---

## 2. Installation

For source installation:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

In this workspace, the shared virtual environment is located one directory above
this repository:

```bash
cd ..
source .venv/bin/activate
cd controllability-score-python
python -m pytest
```

The distribution name is `controllability-scoring`, and the import name is
`controllability_scoring`.

---

## 3. Overview of the Overall Structure

```text
project/
├─ src/controllability_scoring/
│  ├─ api.py                    # Top-level API
│  ├─ problem.py                # Problem setting
│  ├─ results.py                # Results
│  ├─ options.py                # Options
│  ├─ gramian/                  # Gramian computation
│  ├─ pg_solvers/               # Projected gradient solver
│  ├─ projections/              # Simplex projection
│  ├─ solvers/                  # VCS/AECS solvers
│  └─ utils/                    # Common utilities
├─ tests/                       # Tests
├─ examples/                    # Usage examples
├─ docs_archive/                # Existing documentation
├─ README.md                    # Documentation in Japanese
├─ README_eng.md                # This document
├─ pyproject.toml               # Package configuration
└─ LICENSE                      # License
```

The roles of the main components are as follows.

- `vcs`: top-level function that computes VCS
- `aecs`: top-level function that computes AECS
- `cs`: top-level function that computes both VCS and AECS from one `CSProblem`
- `CSProblem`: class representing the problem setting
- `CSResult`: class storing optimization results
- `WOptions`: options for Gramian computation
- `PGSolverOptions`: options for optimization

---

## 4. Usage Example

```python
import numpy as np
from controllability_scoring import vcs, aecs, cs

A = np.diag([-1.0, -2.0, 0.5, 1.2])

p_v, info_v, wlist_v, transform = vcs(A)
p_a, info_a, wlist_a, transform = aecs(A)
p_v, p_a, info_v, info_a, wlist, transform = cs(A)
```

More examples are available in `examples/`.

```bash
python examples/ex01_minimal_cs.py
python examples/ex02_finite_horizon.py
python examples/ex03_edge_weight_sweep.py
python examples/ex04_visualize_scores.py
python examples/ex05_minimal_graph_cs.py
```

---

## 5. Output

`vcs`, `aecs`, and `cs` return weight vectors on the probability simplex:

```math
p_i \ge 0,\quad \sum_i p_i = 1.
```

Larger weights indicate stronger contribution to the corresponding
controllability objective.

### `vcs(A, ...)`

Returns:

- `p_v`: optimal simplex weights for VCS
- `info_v`: solver diagnostics
- `wlist_v`: node-wise Gramian matrices
- `transform`: transform matrix

### `aecs(A, ...)`

Returns:

- `p_a`: optimal simplex weights for AECS
- `info_a`: solver diagnostics
- `wlist_a`: node-wise Gramian matrices
- `transform`: transform matrix

### `cs(A, ...)`

Returns:

- `p_v`: optimal simplex weights for VCS
- `p_a`: optimal simplex weights for AECS
- `info_v`: VCS solver diagnostics
- `info_a`: AECS solver diagnostics
- `wlist`: node-wise Gramian matrices
- `transform`: transform matrix

---

## 6. Current Scope

- Full-state VCS/AECS are implemented.
- Finite-horizon and infinite-horizon formulations are implemented.
- Lyapunov and integral methods are implemented.
- Exported Gramian coordinates can be selected by `w_output="orig"` or
  `w_output="trans"`.
- The Python API returns fixed-length tuples.

### Unsupported Features

Target controllability scores for a selected subset of state nodes are not
implemented yet. The current API focuses on full-state VCS/AECS.

---

## 7. Detailed Documentation

Additional theoretical background and implementation notes are available in
`docs_archive/`.

- [docs_archive/docs/index.md](docs_archive/docs/index.md): entry point for archived documentation
- [docs_archive/docs/theory.md](docs_archive/docs/theory.md): theoretical background of VCS/AECS
- [docs_archive/docs/gramian.md](docs_archive/docs/gramian.md): overview of Gramian computation
- [docs_archive/docs/api.md](docs_archive/docs/api.md): API and option details

---

## Citation

If you use this source code in a paper, please cite the following references.

```bibtex
@article{sato2024controllability,
  title={Controllability scores for selecting control nodes of large-scale network systems},
  author={Sato, Kazuhiro and Terasaki, Shun},
  journal={IEEE Transactions on Automatic Control},
  volume={69},
  number={7},
  pages={4673--4680},
  year={2024},
  publisher={IEEE}
}

@article{sato2025uniqueness,
  title={Uniqueness analysis of controllability scores and their application to brain networks},
  author={Sato, Kazuhiro and Kawamura, Ryohei},
  journal={IEEE Transactions on Control of Network Systems},
  volume={12},
  number={4},
  pages={2568--2580},
  year={2025},
  publisher={IEEE}
}

@article{sato2025target,
  title={Target Controllability Scores for Actuation-Constrained Network Intervention},
  author={Kazuhiro Sato},
  journal={arXiv preprint arXiv:2510.13354},
  year={2025}
}

@article{umezu2026infinite,
  title={Infinite-horizon controllability scores for linear time-invariant systems},
  author={Umezu, Kota and Sato, Kazuhiro},
  journal={arXiv preprint arXiv:2601.10260},
  year={2026}
}
```

## License

This software is released under the MIT License. See `LICENSE` for details.

## Contributors

This software has been developed by members of the Sato Group.

Project lead:

- Kazuhiro Sato

Main contributors:

- Riku Yaegashi
- Michiya Iwata

## Contact

For questions or bug reports, please open an issue on GitHub or contact:

Kazuhiro Sato

kazuhiro[at]mist.i.u-tokyo.ac.jp
