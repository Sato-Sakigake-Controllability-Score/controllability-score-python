# Controllability Scoring

Controllability-based node scoring for linear networked dynamical systems.

This package computes node importance weights by solving convex optimization
problems over the probability simplex using controllability Gramian contributions
derived from a system matrix A.

The resulting solution provides a principled and interpretable ranking of
candidate control nodes in large-scale systems.

---

## ✨ Features

- Volume-based controllability score (VCS, log-determinant objective)
- Energy-based controllability score (AECS, trace-inverse objective)
- Simplex-constrained convex optimization
- Projected gradient solver
- Finite- and infinite-horizon formulations
- Numerically stable matrix operations

---

## 📦 Installation

```bash
pip install controllability-scoring
```

---

## 🚀 Quickstart

After installation, you can compute controllability scores as follows:

```python
import numpy as np
from controllability_scoring import vcs, aecs, cs

# Example: n-node system
n = 10
rng = np.random.default_rng(0)

# Stable example system matrix
A = -0.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))

# Compute VCS
pV, infoV, WlistV, P = vcs(A, T=10.0)
print("Top-3 nodes (VCS):", np.argsort(-pV)[:3])

# Compute AECS
pA, infoA, WlistA, P = aecs(A, T=10.0)
print("Top-3 nodes (AECS):", np.argsort(-pA)[:3])

# Compute both at once
pV, pA, infoV, infoA, Wlist, P = cs(A, T=10.0)
```

The returned vectors lie on the probability simplex and represent
soft importance weights for control node ranking.

---

## 📊 Output

All scoring functions return weight vectors on the probability simplex:

$$
p_i \ge 0, \quad \sum_i p_i = 1.
$$

Larger weights indicate stronger controllability contribution.

### `vcs(A, ...)`

Returns:

- `pV` — optimal simplex weights (volume-based score)
- `infoV` — solver diagnostics
- `WlistV` — node-wise matrices
- `P` — optional auxiliary matrix

### `aecs(A, ...)`

Returns:

- `pA` — optimal simplex weights (energy-based score)
- `infoA`
- `WlistA`
- `P`

### `cs(A, ...)`

Computes both scores simultaneously.


---

## 📐 Mathematical Formulation

Let \( W_i(A, T) \) denote node-wise controllability contributions
constructed from the system matrix \( A \) and time horizon \( T \).

We optimize over the probability simplex

$$
\Delta = \{ p \in \mathbb{R}^{n} \;|\; p_{i} \ge 0,\ \sum_{i} p_{i} = 1 \}
$$

### Volume-based score (VCS)

$$
\max_{p \in \Delta}
\log\det\left( \sum_i p_i W_i \right)
$$

### Energy-based score (AECS)

$$
\min_{p \in \Delta}
\mathrm{tr}\left( \left( \sum_i p_i W_i \right)^{-1} \right)
$$

These convex formulations provide soft rankings of control nodes.

---

## Origin

This package is a Python implementation of the controllability scoring framework originally proposed in:

Kazuhiro Sato, "Controllability scores for selecting control nodes of large-scale network systems", IEEE Transactions on Automatic Control, 2024.

An earlier implementation was developed in MATLAB by Kouta Umeze.

## 📝 License

MIT License
