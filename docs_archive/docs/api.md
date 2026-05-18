# API Reference

This page documents the public API of `controllability_scoring`.

---

# Core Functions

## `vcs`

### Signature

```python
vcs(
    A,
    T=math.inf,
    *,
    w_options=None,
    steps=None,
    solver_options=None,
    initial_guess=None,
    w_output="trans",
)
```

### Parameters

- **A**  
  Array-like of shape `(n, n)`.  
  System / network matrix.

- **T**  
  Float. Time horizon.

- **w_options**  
  Instance of `WOptions`. Controls construction of node-wise matrices.

- **steps**  
  Optional discretization parameter (used when constructing `WOptions` internally).

- **solver_options**  
  Instance of `PGSolverOptions`. Controls projected gradient solver behavior.

- **initial_guess**  
  Optional initial vector on the probability simplex.

- **w_output**  
  Controls auxiliary outputs.

### Returns

- **pV**  
  Optimal simplex weights (volume-based score).

- **infoV**  
  Solver diagnostics.

- **WlistV**  
  Node-wise matrices.

- **P**  
  Optional auxiliary matrix.

---

## `aecs`

### Signature

```python
aecs(
    A,
    T=math.inf,
    *,
    w_options=None,
    steps=None,
    solver_options=None,
    initial_guess=None,
    w_output="trans",
)
```

### Returns

- **pA**  
  Optimal simplex weights (energy-based score).

- **infoA**  
  Solver diagnostics.

- **WlistA**  
  Node-wise matrices.

- **P**  
  Optional auxiliary matrix.

---

## `cs`

Computes both VCS and AECS.

### Signature

```python
cs(
    A,
    T=math.inf,
    *,
    w_options=None,
    steps=None,
    solver_options=None,
    initial_guess=None,
    w_output="trans",
)
```

### Returns

- **pV**  
  VCS weights.

- **pA**  
  AECS weights.

- **infoV**  
  VCS diagnostics.

- **infoA**  
  AECS diagnostics.

- **Wlist**  
  Node-wise matrices.

- **P**  
  Optional auxiliary matrix.

---

# Options

Both `WOptions` and `PGSolverOptions` are immutable dataclasses.
Invalid configurations are rejected at construction time.
Use `with_...()` methods to create modified copies.

---

## `WOptions`

Controls how node-wise controllability contributions are constructed.

### Fields

- **method**  
  One of `"lyap"`, `"integral"`, `"trapezoidal"`, `"simpson"`.

- **steps**  
  Integer discretization parameter for non-`"lyap"` methods.

- **use_scaling**  
  Boolean flag enabling numerical scaling.

### Method–Steps Rules

- If `method == "lyap"`, then `steps` is forced to `0`.
- If `method != "lyap"`, then `steps >= 1` is required.
- Switching from `"lyap"` to non-`"lyap"` defaults `steps` to `50`
  unless explicitly overridden.

### Recommended Usage

- Infinite horizon (`T = inf`) → `method="lyap"`.
- Finite horizon → `method="integral"` with `steps=50`.

---

## `PGSolverOptions`

Configures the projected gradient solver with Armijo backtracking.

### Line Search Parameters

- **step_size**  
  Initial step length.

- **step_size_inf**  
  Minimum allowable step length.

### Termination Parameters

- **max_iter**  
  Maximum number of iterations.

- **tol**  
  Stopping tolerance on update norm.

### Armijo Parameters

- **rho**  
  Backtracking shrink factor in `(0, 1)`.

- **sigma**  
  Sufficient decrease parameter in `(0, 1)`.

### Diagnostics

- **verbose**  
  Print iteration diagnostics.

- **store_trace**  
  Store iteration history.

### Validation Rules

- `step_size`, `step_size_inf`, `tol` must be positive.
- `max_iter` must be a positive integer.
- `rho`, `sigma` must satisfy `0 < rho < 1`.
