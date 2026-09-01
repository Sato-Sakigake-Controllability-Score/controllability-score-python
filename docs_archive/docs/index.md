# Controllability Scoring Documentation

Controllability Scoring provides convex-optimization-based node ranking
for linear networked dynamical systems.

Given a system matrix $A$ and a time horizon $T$,
the package computes simplex weights that rank candidate control nodes
according to controllability-based objectives.

---

## Implemented Scores

Two convex formulations are implemented:

### Volume-based score (VCS)

$$
\max_{p \in \Delta}
\log\det\bigl( \sum_{i} p_{i} W_{i} \bigr)
$$

### Energy-based score (AECS)

$$
\min_{p \in \Delta}
\mathrm{tr}\bigl( (\sum_{i} p_{i} W_{i})^{-1} \bigr)
$$

where $W_{i}$ are node-wise controllability contributions.

---

For theoretical details, see [theory.md](theory.md).
For function specifications, see [api.md](api.md).
