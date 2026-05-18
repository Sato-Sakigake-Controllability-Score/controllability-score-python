from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Literal
import numpy as np
import numpy.typing as npt
from typing import cast


from .utils import validate as v

Method = Literal["lyap", "integral", "trapezoidal", "simpson"]
_ALLOWED_METHODS = ("lyap", "integral", "trapezoidal", "simpson")


def _normalize_method(m: str) -> Method:
    m2 = str(m).strip().lower()
    if m2 not in _ALLOWED_METHODS:
        raise ValueError(f'Method must be one of {_ALLOWED_METHODS}, got "{m}".')
    return cast(Method, m2)

def _as_float(x, name: str) -> float:
    """
    Convert input to a finite float.

    Accepts Python numeric scalars and numpy scalars.
    Rejects bool explicitly (because bool is a subclass of int in Python).
    """
    if isinstance(x, bool):
        raise TypeError(f"{name} must be a real scalar")
    try:
        v = float(x)
    except Exception as e:
        raise TypeError(f"{name} must be a real scalar") from e
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _as_pos_float(x, name: str) -> float:
    """Finite float strictly greater than 0."""
    v = _as_float(x, name)
    if v <= 0:
        raise ValueError(f"{name} must be positive")
    return v


def _as_open01(x, name: str) -> float:
    """Finite float strictly inside the open interval (0, 1)."""
    v = _as_float(x, name)
    if not (0.0 < v < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    return v


def _as_pos_int(x, name: str) -> int:
    """
    Convert input to a positive int.

    Notes:
    - Reject bool (bool is an int subclass).
    - Accept numpy integer scalars.
    - Int-like strings (e.g., "1000") are rejected on purpose to avoid silent coercions.
    """
    if isinstance(x, bool):
        raise TypeError(f"{name} must be a positive integer")
    if isinstance(x, (int, np.integer)):
        v = int(x)
    else:
        raise TypeError(f"{name} must be a positive integer")
    if v <= 0:
        raise ValueError(f"{name} must be positive")
    return v

@dataclass(frozen=True, slots=True)
class WOptions:
    """
    Pythonic WOptions (immutable).

    Design:
    - Immutable options object (safe to share; no hidden side effects on assignment).
    - Validated at construction time; invalid states cannot exist.
    - To "modify" options, use with_method()/with_steps()/with_use_scaling()
      which return a new instance.

    Rules (same intent as MATLAB):
    - method == "lyap"  => steps is always 0
    - method != "lyap"  => steps >= 1
    - switching lyap -> non-lyap defaults steps to 50 unless explicitly overridden
    """
    method: Method = "lyap"
    steps: int = 0
    use_scaling: bool = True
    eigtol: float = 1e-12

    def __post_init__(self) -> None:
        m = _normalize_method(self.method)
        s = self.steps

        # keep normalized method (in case user passed "Lyap")
        object.__setattr__(self, "method", m)  # type: ignore[arg-type]

        if m == "lyap":
            # lyap forces steps=0 regardless of input
            object.__setattr__(self, "steps", 0)
            return

        # non-lyap: steps must be integer >= 1
        if not isinstance(s, (int, np.integer)):
            raise TypeError("steps must be an integer")
        if int(s) < 1:
            raise ValueError(f'steps must be >= 1 when method is "{m}"')
        object.__setattr__(self, "steps", int(s))

    @classmethod
    def from_system(
        cls,
        A: Optional[npt.NDArray[np.float64]] = None,
        T: float = np.inf,
        *,
        method: Optional[Method] = None,
        steps: Optional[int] = None,
        use_scaling: Optional[bool] = None,
    ) -> "WOptions":
        """
        Factory that mirrors the MATLAB constructor intent, but in a Pythonic way.

        - Validates A (if provided) and T.
        - Chooses defaults from T:
            T=inf    => method="lyap", steps=0
            finite T => method="integral", steps=50
        - If method is explicitly given and is non-lyap, and steps is omitted,
          steps defaults to 50 (MATLAB's lyap->non-lyap behavior).
        - If steps is explicitly given, it wins (and will be validated).
        """
        if A is not None:
            A = np.asarray(A, dtype=float)
            v.validate_A(A)

        T = float(T)
        v.validate_T(T)

        default_method: Method = "lyap" if np.isinf(T) else "integral"
        default_use_scaling = True

        m = default_method if method is None else method
        m_norm = _normalize_method(m)  # runtime validation

        # Decide steps with MATLAB-compatible intent:
        if m_norm == "lyap":
            s = 0
        else:
            # For any non-lyap, default to 50 unless user specified otherwise
            s = 50 if steps is None else int(steps)

        usc = default_use_scaling if use_scaling is None else bool(use_scaling)
        return cls(method=m_norm, steps=s, use_scaling=usc)  # validated in __post_init__

    # ---- Pythonic "updates" (return new instances) ----
    def with_method(self, method: Method) -> "WOptions":
        m_norm = _normalize_method(method)
        if m_norm == "lyap":
            return replace(self, method="lyap", steps=0)

        # MATLAB intent: lyap -> non-lyap sets steps to 50 automatically
        if self.method == "lyap":
            return replace(self, method=m_norm, steps=50)

        # keep current steps when already non-lyap
        return replace(self, method=m_norm)

    def with_steps(self, steps: int) -> "WOptions":
        # If method is lyap, __post_init__ will force steps=0 anyway.
        return replace(self, steps=int(steps))

    def with_use_scaling(self, use_scaling: bool) -> "WOptions":
        return replace(self, use_scaling=bool(use_scaling))

# =========================
# PGSolverOptions
# =========================
@dataclass(frozen=True, slots=True)
class PGSolverOptions:
    """
    Options for the projected gradient solver (Armijo backtracking).

    This class defines how the projected gradient solver behaves, independently
    of the problem being solved.  It only contains algorithmic hyperparameters
    and does not depend on problem data such as A, T, or discretization steps h.

    Change only the parameters you need; all others keep stable, well-tested
    defaults.

    This is a Python equivalent of MATLAB's PGSolverOptions with the same default
    values and parameter constraints, adapted to a more explicit and immutable
    (pytonic) design.

    Design principles:
    - Immutable options object (frozen dataclass):
        * no accidental mutation during runs
        * safe to share between experiments and tests
    - Validated at construction time:
        * invalid option states cannot exist
    - Modification is explicit:
        * use the constructor or with_xxx() methods to create modified copies

    Parameters:
    - step_size:
        Initial trial step length (alpha) for the Armijo backtracking line search.
    - step_size_inf:
        Minimum allowable step length during backtracking. If the trial step size
        falls below this value, the solver terminates because no further progress
        can be made.
    - max_iter:
        Maximum number of solver iterations.
    - tol:
        Stopping tolerance on the update norm:
            ||p_{k+1} - p_k|| <= tol.
    - rho:
        Backtracking shrink factor in the open interval (0, 1). During line search,
        the step size is updated as alpha <- rho * alpha.
    - sigma:
        Sufficient decrease parameter in the Armijo condition, in the open
        interval (0, 1).
    - verbose:
        If True, print iteration diagnostics during the solve.
    - store_trace:
        If True, store iteration history (e.g., objective values, step sizes)
        for later inspection.

    Constraints (mirrors MATLAB validateattributes intent):
    - step_size, step_size_inf, tol are positive finite scalars
    - max_iter is a positive integer
    - rho and sigma are finite scalars strictly between 0 and 1

    Examples:
        # Use default solver options
        opts = PGSolverOptions()

        # Change only what you need
        opts = PGSolverOptions(max_iter=2000, tol=1e-10)

        # Immutable update: create a modified copy
        opts2 = opts.with_tol(1e-10).with_max_iter(2000)
    """


    # ---- Line search parameters ----
    step_size: float = 0.1         # initial alpha
    step_size_inf: float = 1e-12   # minimum alpha

    # ---- Termination / iteration ----
    max_iter: int = 1000           # iteration cap
    tol: float = 1e-8              # ||p_{k+1} - p_k|| tolerance

    # ---- Armijo rule parameters ----
    rho: float = 0.5               # backtracking factor
    sigma: float = 1e-4            # sufficient decrease

    # ---- Diagnostics ----
    verbose: bool = False
    store_trace: bool = False

    def __post_init__(self) -> None:
        """
        Normalize and validate all fields.

        After this method returns, every attribute is guaranteed to satisfy
        the documented constraints (or the constructor raises).
        """
        object.__setattr__(self, "step_size", _as_pos_float(self.step_size, "step_size"))
        object.__setattr__(self, "step_size_inf", _as_pos_float(self.step_size_inf, "step_size_inf"))
        object.__setattr__(self, "max_iter", _as_pos_int(self.max_iter, "max_iter"))
        object.__setattr__(self, "tol", _as_pos_float(self.tol, "tol"))
        object.__setattr__(self, "rho", _as_open01(self.rho, "rho"))
        object.__setattr__(self, "sigma", _as_open01(self.sigma, "sigma"))

        # Ensure bool-ness (accepts numpy.bool_ etc.)
        object.__setattr__(self, "verbose", bool(self.verbose))
        object.__setattr__(self, "store_trace", bool(self.store_trace))

    # -------------------------
    # Immutable "updates"
    # -------------------------
    def with_step_size(self, step_size: float) -> "PGSolverOptions":
        """Return a new options object with step_size replaced."""
        return replace(self, step_size=step_size)

    def with_step_size_inf(self, step_size_inf: float) -> "PGSolverOptions":
        """Return a new options object with step_size_inf replaced."""
        return replace(self, step_size_inf=step_size_inf)

    def with_max_iter(self, max_iter: int) -> "PGSolverOptions":
        """Return a new options object with max_iter replaced."""
        return replace(self, max_iter=max_iter)

    def with_tol(self, tol: float) -> "PGSolverOptions":
        """Return a new options object with tol replaced."""
        return replace(self, tol=tol)

    def with_rho(self, rho: float) -> "PGSolverOptions":
        """Return a new options object with rho replaced."""
        return replace(self, rho=rho)

    def with_sigma(self, sigma: float) -> "PGSolverOptions":
        """Return a new options object with sigma replaced."""
        return replace(self, sigma=sigma)

    def with_verbose(self, verbose: bool) -> "PGSolverOptions":
        """Return a new options object with verbose replaced."""
        return replace(self, verbose=verbose)

    def with_store_trace(self, store_trace: bool) -> "PGSolverOptions":
        """Return a new options object with store_trace replaced."""
        return replace(self, store_trace=store_trace)

