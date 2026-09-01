# Theoretical Background

## Linear System Model

We consider a linear time-invariant (LTI) system

$$
\dot{x}(t) = A x(t) + B u(t),
$$

where $A \in \mathbb{R}^{n \times n}$.

Node-wise controllability contributions are constructed internally
by associating candidate control inputs with network nodes.

---

## Finite-Horizon Controllability Gramian

For a time horizon $T$,

$$
W(T) = \int_{0}^{T} e^{At} B B^\top e^{A^\top t} dt.
$$

The package internally constructs node-wise matrices $W_{i}(A, T)$.

---

## Optimization over the Simplex

We optimize over the probability simplex

$$
\Delta = \{ p \in \mathbb{R}^{n} \mid p_{i} \ge 0,\ \sum_{i} p_{i} = 1 \}
$$

The combined matrix is

$$
S(p) = \sum_{i} p_{i} W_{i}.
$$

---

## Volume-Based Score (VCS)

$$
\max_{p \in \Delta} \log\det(S(p)).
$$

This objective measures the volume of the reachable ellipsoid.

Gradient:

$$
\frac{\partial}{\partial p_{i}} \log\det(S(p)) = \mathrm{tr}\bigl(S(p)^{-1} W_{i}\bigr).
$$


---

## Energy-Based Score (AECS)

$$
\min_{p \in \Delta}
\mathrm{tr}\bigl(S(p)^{-1}\bigr).
$$

Gradient:

$$
\frac{\partial}{\partial p_{i}}\mathrm{tr}\bigl(S(p)^{-1}\bigr) = - \mathrm{tr}\bigl(S(p)^{-1} W_{i} S(p)^{-1}\bigr).
$$

---

## Convexity

- $-\log\det(\cdot)$ is convex on the positive definite cone.
- $\mathrm{tr}(X^{-1})$ is convex on $X \succ 0$.
- Since $S(p)$ is affine in $p$, both problems are convex
  over the simplex domain.

---

## Numerical Considerations

To ensure positive definiteness:

- Symmetrization is applied
- Diagonal regularization may be used
- Stable matrix factorization is employed
