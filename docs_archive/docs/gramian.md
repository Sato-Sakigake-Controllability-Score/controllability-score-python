# Gramian Construction

This package internally constructs **node-wise controllability contributions**

$$
W_{i}(A,T)=\int_{0}^{T} e^{At} e_{i} e_{i}^\top e^{A^\top t} dt,
$$

which serve as the building blocks for the convex objectives.

Volume-based score (VCS):

$$
\max_{p \in \Delta}\log\det\bigl(\sum_{i} p_{i} W_{i}\bigr)
$$

Energy-based score (AECS):

$$
\min_{p \in \Delta} \mathrm{tr}\bigl((\sum_{i} p_{i} W_{i})^{-1}\bigr)
$$

Depending on

- finite vs infinite time horizon,
- whether spectral scaling is enabled,
- and which objective (VCS or AECS) is evaluated,

the Gramian construction differs.

---

# Spectral Decomposition and Scaling

When `use_scaling=True`, the system matrix $A$ is decomposed into spectral blocks:

$$
J = Q^{-1} A Q =\begin{bmatrix}
A_{S} & 0 & 0 \\
0 & A_{I} & 0 \\
0 & 0 & A_{U}
\end{bmatrix},
$$

where:

- $A_{S}$: stable block, with $\mathrm{Re}\lambda < 0$
- $A_{I}$: imaginary-axis block
- $A_{U}$: unstable block, with $\mathrm{Re}\lambda > 0$

Node inputs are transformed as

$$
\tilde b_{i} = Q^{-1} e_{i}.
$$

All Gramian computations are performed in this block coordinate system and stored in block form inside `WList`.

This separation allows:

- Numerically stable Lyapunov solves
- Proper handling of unstable and marginal modes
- Objective-dependent block selection

---

# Finite Horizon (T < ∞, Scaling Enabled)

For finite time,

$$
W_{i}(T)=\int_{0}^{T} e^{At} e_{i} e_{i}^\top e^{A^\top t}\,dt.
$$

After block decomposition, the transformed Gramian

$$
\tilde W_{i}(T) = \int_{0}^{T} e^{Jt} \tilde b_{i} \tilde b_{i}^\top e^{J^\top t} dt
$$

inherits the same block structure as $J$.

## Stable Block

For $A_{S}$, we use the identity

$$
A_{S} W + W A_{S}^\top = e^{A_{S} T} X e^{A_{S}^\top T} - X,
$$

where $X = \tilde b_{i,S} \tilde b_{i,S}^\top$.

This converts the finite-time integral into a continuous Lyapunov equation:

$$
A_{S} W + W A_{S}^\top + (X - e^{A_{S} T} X e^{A_{S}^\top T}) = 0,
$$

which is solved using `solve_continuous_lyapunov`.

This avoids time discretization and provides high numerical accuracy.

---

## Unstable Block

For $A_{U}$, direct computation leads to exponential growth.

Instead, we solve a Lyapunov equation for $-A_{U}$, effectively applying a
**time-reversal transformation**. This replaces growing exponentials with
decaying ones and stabilizes the computation.

---

## Imaginary-Axis Block

For $A_{I}$, the Lyapunov operator may be singular or ill-conditioned.

The implementation constructs an augmented matrix:

$$
\begin{bmatrix}
    - A_{I} & X \\
    0 & A_{I}^\top
\end{bmatrix},
$$

and extracts the integral from the off-diagonal block of its matrix exponential. This provides a stable evaluation of

$$
\int_{0}^{T} e^{A_{I} t} X e^{A_{I}^\top t} dt.
$$

---

## Block Scaling

Finite-time scaling introduces

$$
D^{-1} = \mathrm{diag}\bigl(
I_{S},\;
\frac{1}{\sqrt{T}} I_{I},\;
e^{-T A_{U}}
\bigr).
$$

This normalization ensures:

- Comparable magnitudes across spectral blocks
- Improved conditioning for optimization
- Stable evaluation of log-determinant and inverse-trace objectives

---

# Infinite Horizon (T = ∞, Scaling Enabled)

The classical infinite-horizon Gramian

$$
\int_{0}^{\infty} e^{At} X e^{A^\top t} dt
$$

exists only if $A$ is strictly stable.

If unstable or imaginary eigenvalues are present, the integral diverges.

Therefore, the implementation does not attempt to compute the classical Gramian directly when unstable modes exist. Instead, it follows the theoretical results established in the accompanying paper.

---

# Objective-Dependent Treatment

The handling of spectral blocks differs between VCS and AECS.

---

## VCS (logdet Objective)

For

$$
\max_{p \in \Delta}
\log\det\bigl(\sum_{i} p_{i} W_{i}\bigr),
$$

the paper shows that a spectrally consistent basis can be constructed as follows:

The stable block is handled by the standard Lyapunov solve.

The imaginary block is handled by a small negative shift:

$$
A_{I} \to A_{I} - \varepsilon I
$$

The unstable block is handled by time reversal:

$$
A_{U} \to -A_{U}
$$

This produces a well-defined basis suitable for evaluating the log-determinant objective.

All spectral blocks are used for VCS.

In `WList`, this is encoded as:

```python
vcs_blocks = [all existing blocks]
```

---

## AECS (trace-inverse Objective)

For

$$
\min_{p \in \Delta}
\mathrm{tr}\bigl(S(p)^{-1}\bigr),
$$

the paper proves:

> Only the stable invariant subspace contributes to the optimal solution.

Therefore, in the infinite-horizon scaled construction:

- Only the stable block $A_{S}$ is required.
- Imaginary and unstable blocks do not affect the minimizer.

In the current Python implementation, `aecs_blocks` selects the first stored
block:

```python
aecs_blocks = [0]
```

When a stable block is present, this corresponds to the stable block. Cases with
no stable block should be treated carefully and remain a numerical/theoretical
review point.

---

# Role of `vcs_blocks` and `aecs_blocks`

`WList` stores Gramians in block form.

The fields

- `vcs_blocks`
- `aecs_blocks`

specify which invariant subspaces are used when forming

$$
S(p) = \sum_{i} p_{i} W_{i}.
$$

This enables:

- Objective-dependent evaluation
- Spectral consistency
- Efficient computation
- Clear separation of theoretical roles

---

# Summary

Finite horizon:

- Stable block → Lyapunov via endpoint identity
- Unstable block → time-reversed Lyapunov
- Imaginary block → augmented exponential construction
- Explicit block scaling applied

Infinite horizon:

- Classical Gramian exists only for stable systems
- With scaling:
  - VCS uses stable + shifted imaginary + time-reversed unstable blocks
  - AECS uses stable block only (theoretical result)
- Block usage is controlled by `vcs_blocks` and `aecs_blocks`

This design ensures:

- Numerical robustness
- Spectral consistency
- Theoretical correctness
- Clear separation between objectives
