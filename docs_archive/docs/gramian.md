# Gramian Construction

This package internally constructs **node-wise controllability contributions**

$$
W_i(A,T)=\int_0^T e^{At} e_i e_i^\top e^{A^\top t} dt,
$$

which serve as the building blocks for the convex objectives.

Volume-based score (VCS):

$$
\max_{p \in \Delta}\log\det\!\left(\sum_i p_i W_i\right)
$$

Energy-based score (AECS):

$$
\min_{p \in \Delta} \operatorname{tr}\!\left(\left(\sum_i p_i W_i\right)^{-1}\right)
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
A_S & 0 & 0 \\
0 & A_I & 0 \\
0 & 0 & A_U
\end{bmatrix},
$$

where:

- $A_S$: stable block, with $\operatorname{Re}\lambda < 0$
- $A_I$: imaginary-axis block
- $A_U$: unstable block, with $\operatorname{Re}\lambda > 0$

Node inputs are transformed as

$$
\tilde b_i = Q^{-1} e_i.
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
W_i(T)=\int_0^T e^{At} e_i e_i^\top e^{A^\top t}\,dt.
$$

After block decomposition, the transformed Gramian

$$
\tilde W_i(T) = \int_0^T e^{Jt} \tilde b_i \tilde b_i^\top e^{J^\top t} dt
$$

inherits the same block structure as $J$.

## Stable Block

For $A_S$, we use the identity

$$
A_S W + W A_S^\top = e^{A_S T} X e^{A_S^\top T} - X,
$$

where $X = \tilde b_{i,S} \tilde b_{i,S}^\top$.

This converts the finite-time integral into a continuous Lyapunov equation:

$$
A_S W + W A_S^\top + (X - e^{A_S T} X e^{A_S^\top T}) = 0,
$$

which is solved using `solve_continuous_lyapunov`.

This avoids time discretization and provides high numerical accuracy.

---

## Unstable Block

For $A_U$, direct computation leads to exponential growth.

Instead, we solve a Lyapunov equation for $-A_U$, effectively applying a
**time-reversal transformation**. This replaces growing exponentials with
decaying ones and stabilizes the computation.

---

## Imaginary-Axis Block

For $A_I$, the Lyapunov operator may be singular or ill-conditioned.

The implementation constructs an augmented matrix:

$$
\begin{bmatrix}
    - A_I & X \\
    0 & A_I^\top
\end{bmatrix},
$$

and extracts the integral from the off-diagonal block of its matrix exponential. This provides a stable evaluation of

$$
\int_0^T e^{A_I t} X e^{A_I^\top t} dt.
$$

---

## Block Scaling

Finite-time scaling introduces

$$
D^{-1} = \mathrm{diag}\!\left(
I_S,\;
\frac{1}{\sqrt{T}} I_I,\;
e^{-T A_U}
\right).
$$

This normalization ensures:

- Comparable magnitudes across spectral blocks
- Improved conditioning for optimization
- Stable evaluation of log-determinant and inverse-trace objectives

---

# Infinite Horizon (T = ∞, Scaling Enabled)

The classical infinite-horizon Gramian

$$
\int_0^\infty e^{At} X e^{A^\top t} dt
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
\log\det\!\left(\sum_i p_i W_i\right),
$$

the paper shows that a spectrally consistent basis can be constructed as follows:

The stable block is handled by the standard Lyapunov solve.

The imaginary block is handled by a small negative shift:

$$
A_I \rightarrow A_I - \varepsilon I
$$

The unstable block is handled by time reversal:

$$
A_U \rightarrow -A_U
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
\operatorname{tr}\!\left(S(p)^{-1}\right),
$$

the paper proves:

> Only the stable invariant subspace contributes to the optimal solution.

Therefore, in the infinite-horizon scaled construction:

- Only the stable block $A_S$ is required.
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
S(p) = \sum_i p_i W_i.
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
