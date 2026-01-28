#!/usr/bin/env python3
"""
Fixed-branch discrete-time optimal control via discrete-time PMP / KKT
====================================================================

This script solves several *fixed-branch* optimal control problems for a 2D state
(vx, vy) and scalar control s, using a discrete-time Pontryagin Maximum Principle (PMP)
structure:

- state recursion: v_{t+1} = f(v_t, s_t)
- costate recursion: λ_t = ∂ℓ/∂v(v_t,s_t,t) + (∂f/∂v(v_t,s_t))ᵀ λ_{t+1}
- per-step control maximization: s_t ∈ argmax_{s∈S_case} [ℓ(v_t,s_t,t) + λ_{t+1}ᵀ f(v_t,s_t)]

Key design goal: avoid optimizing jointly over (s_0,...,s_{N-1}). Instead, per-step
maximization is 1D, and boundary conditions are handled by low-dimensional root finding
over terminal multipliers (bridge constraints).

Dependencies:
- numpy
- scipy (minimize_scalar, root, least_squares)  (see SciPy docs)
- Optional: jax + jaxlib for autodiff (grad/jacrev)  (see JAX docs)

If JAX is unavailable (common on minimal environments), the solver falls back to robust
finite-difference derivatives for Jacobians/gradients.

The code is organized into modules-in-one-file:
- dynamics (original recurrence + fixed-branch variants)
- objective (default vy objective + hook)
- PMP solver (FBSM, single/multiple shooting, root solve)
- runner / CLI
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize_scalar, least_squares, root


# ---------------------------------------------------------------------------
# Optional JAX import (autodiff acceleration)
# ---------------------------------------------------------------------------
HAS_JAX = False
JAX_IMPORT_ERROR: Optional[Exception] = None
try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    HAS_JAX = True
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        # If config isn't available (older JAX), ignore.
        pass
except Exception as e:  # pragma: no cover
    HAS_JAX = False
    JAX_IMPORT_ERROR = e
    jax = None  # type: ignore
    jnp = None  # type: ignore


ArrayLike = Union[np.ndarray, Sequence[float]]


###############################################################################
# Dynamics module
###############################################################################


# --- Original recurrence (must be included verbatim) -------------------------
def next_v(v, s):
    assert -1 < s < 1
    vx, vy = v
    assert vx > 0
    c2 = 1 - s * s
    prev_vx = vx
    vy += 0.06 * c2 - 0.08
    if vy < 0:
        ty = 0.1 * vy * c2
        vx -= ty
        vy -= ty
    if s > 0:
        tx = 0.04 * prev_vx * s
        vx -= tx
        vy += 3.2 * tx
    vx += 0.1 * (prev_vx - vx)
    return 0.99 * vx, 0.98 * vy


# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    name: str
    y_on: bool  # force vy<0 correction ON
    s_on: bool  # force s>0 correction ON
    vy_nonneg: bool  # True => enforce vy>=0; False => enforce vy<0
    s_bounds: Tuple[float, float]  # numeric bounds used for optimization (lo, hi)

    def __post_init__(self):
        if self.case_id not in ("A", "B", "C", "D"):
            raise ValueError("case_id must be one of A/B/C/D")


CASE_SPECS: Dict[str, CaseSpec] = {
    # Case A: Y-off, S-off: vy>=0, s<=0
    "A": CaseSpec(
        "A",
        "Y-off, S-off",
        y_on=False,
        s_on=False,
        vy_nonneg=True,
        s_bounds=(-1.0 + 1e-9, 0.0),
    ),
    # Case B: Y-off, S-on: vy>=0, s>0
    "B": CaseSpec(
        "B",
        "Y-off, S-on",
        y_on=False,
        s_on=True,
        vy_nonneg=True,
        s_bounds=(+1e-9, 1.0 - 1e-9),
    ),
    # Case C: Y-on, S-off: vy<0, s<=0
    "C": CaseSpec(
        "C",
        "Y-on, S-off",
        y_on=True,
        s_on=False,
        vy_nonneg=False,
        s_bounds=(-1.0 + 1e-9, 0.0),
    ),
    # Case D: Y-on, S-on: vy<0, s>0
    "D": CaseSpec(
        "D",
        "Y-on, S-on",
        y_on=True,
        s_on=True,
        vy_nonneg=False,
        s_bounds=(+1e-9, 1.0 - 1e-9),
    ),
}


def make_f_case_np(case: CaseSpec) -> Callable[[np.ndarray, float], np.ndarray]:
    """NumPy dynamics with fixed branches (smooth within the fixed-branch region)."""
    y_on = case.y_on
    s_on = case.s_on

    def f(v: np.ndarray, s: float) -> np.ndarray:
        vx = float(v[0])
        vy = float(v[1])
        c2 = 1.0 - s * s
        prev_vx = vx

        vy = vy + 0.06 * c2 - 0.08
        vx_mod = vx

        if y_on:
            ty = 0.1 * vy * c2
            vx_mod = vx_mod - ty
            vy = vy - ty

        if s_on:
            tx = 0.04 * prev_vx * s
            vx_mod = vx_mod - tx
            vy = vy + 3.2 * tx

        vx_mod = vx_mod + 0.1 * (prev_vx - vx_mod)
        return np.array([0.99 * vx_mod, 0.98 * vy], dtype=np.float64)

    return f


def make_f_case_jax(case: CaseSpec) -> Optional[Callable]:
    """JAX dynamics with fixed branches (only if JAX is available)."""
    if not HAS_JAX:
        return None

    y_on = case.y_on
    s_on = case.s_on

    def f(v, s):
        vx, vy = v[0], v[1]
        c2 = 1.0 - s * s
        prev_vx = vx

        vy2 = vy + 0.06 * c2 - 0.08
        vx2 = vx

        if y_on:
            ty = 0.1 * vy2 * c2
            vx2 = vx2 - ty
            vy2 = vy2 - ty

        if s_on:
            tx = 0.04 * prev_vx * s
            vx2 = vx2 - tx
            vy2 = vy2 + 3.2 * tx

        vx2 = vx2 + 0.1 * (prev_vx - vx2)
        return jnp.array([0.99 * vx2, 0.98 * vy2])

    return jax.jit(f)


###############################################################################
# Objective module
###############################################################################

StageCostFn = Callable[..., float]  # user hook: signature (v,s,t) or (v,s,t,v_next)


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Base stage cost to MAXIMIZE.
    - default: vy_t (or vy_{t+1})
    - optional user hook: callable ℓ(v_t, s_t, t) or ℓ(v_t, s_t, t, v_{t+1})

    For PMP derivatives, the solver works with an "augmented" stage cost:
        ℓ_aug = ℓ_base - penalty_weight * penalty(v_t)
    """

    name: str
    use_vy_next: bool = False
    stage_cost_user: Optional[StageCostFn] = None  # if provided, overrides defaults


def _call_user_stage_cost(
    fn: StageCostFn, v: np.ndarray, s: float, t: int, v_next: np.ndarray
) -> float:
    """Call a user stage-cost hook; supports either (v,s,t) or (v,s,t,v_next)."""
    try:
        return float(fn(v, s, t, v_next))  # type: ignore[misc]
    except TypeError:
        return float(fn(v, s, t))  # type: ignore[misc]


###############################################################################
# Feasibility module (vy sign constraint)
###############################################################################


@dataclass(frozen=True)
class FeasibilitySpec:
    mode: str = "hard"  # "hard" or "soft"
    vy_tol: float = 0.0
    # for soft penalties
    penalty_weight: float = 1.0
    penalty_growth: float = 10.0
    penalty_rounds: int = 1  # >=1 (homotopy schedule)


def vy_violation_magnitudes(
    v_traj: np.ndarray, case: CaseSpec, vy_tol: float
) -> np.ndarray:
    vy = v_traj[:, 1]
    if case.vy_nonneg:
        # want vy >= 0
        return np.maximum(0.0, vy_tol - vy)
    # want vy < 0
    return np.maximum(0.0, vy + vy_tol)


def count_vy_violations(
    v_traj: np.ndarray, case: CaseSpec, vy_tol: float
) -> Tuple[int, float]:
    viol = vy_violation_magnitudes(v_traj, case, vy_tol)
    n = int(np.sum(viol > 0.0))
    max_viol = float(np.max(viol)) if viol.size else 0.0
    return n, max_viol


###############################################################################
# Terminal constraint setups
###############################################################################


@dataclass(frozen=True)
class TerminalConstraint:
    """
    Terminal constraint g(v_N)=0 represented as:
      - target vT (2,) if m>0
      - mask (2,) booleans specifying constrained components

    Example:
      full bridge: mask=[True, True], vT=[vxT, vyT]
      constrain vy only: mask=[False, True], vT=[*, vyT] (vxT ignored)
    """

    vT: Optional[np.ndarray] = None
    mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([False, False], dtype=bool)
    )

    def m(self) -> int:
        return int(np.sum(self.mask))

    def projector(self) -> np.ndarray:
        idx = np.where(self.mask)[0].tolist()
        P = np.zeros((len(idx), 2), dtype=np.float64)
        for k, i in enumerate(idx):
            P[k, i] = 1.0
        return P

    def residual(self, vN: np.ndarray) -> np.ndarray:
        m = self.m()
        if m == 0:
            return np.zeros((0,), dtype=np.float64)
        if self.vT is None:
            raise ValueError(
                "TerminalConstraint: vT must be provided when mask has constraints."
            )
        P = self.projector()
        return P @ (vN - self.vT)

    def lambdaN_from_mu(self, mu: np.ndarray) -> np.ndarray:
        """
        Map terminal multiplier mu (m,) to λ_N (2,) via λ_N = Pᵀ mu
        (no terminal cost by default).
        """
        P = self.projector()
        return P.T @ mu


###############################################################################
# Differentiation backend (JAX or finite difference)
###############################################################################


@dataclass(frozen=True)
class DifferentiationSpec:
    method: str = "auto"  # "auto", "jax", "finite"
    fd_eps: float = 1e-6


class Differentiator:
    """
    Provides:
      - jac_f_v(v,s) ≈ ∂f/∂v (2x2)
      - grad_ell_v(v,s,t) ≈ ∂ℓ_aug/∂v (2,)

    Uses JAX if available & requested, otherwise finite differences.
    """

    def __init__(
        self,
        f_np: Callable[[np.ndarray, float], np.ndarray],
        ell_np: Callable[[np.ndarray, float, int], float],
        spec: DifferentiationSpec,
        f_jax: Optional[Callable] = None,
        ell_jax: Optional[Callable] = None,
    ):
        self.f_np = f_np
        self.ell_np = ell_np
        self.spec = spec

        want_jax = (spec.method == "jax") or (spec.method == "auto")
        self.use_jax = bool(
            want_jax and HAS_JAX and (f_jax is not None) and (ell_jax is not None)
        )

        self.f_jax = f_jax
        self.ell_jax = ell_jax

        if spec.method == "jax" and not self.use_jax:
            # Hard request but unavailable
            raise RuntimeError(
                "DifferentiationSpec.method='jax' requested, but JAX is unavailable. "
                f"Import error: {JAX_IMPORT_ERROR}"
            )

        if self.use_jax:
            assert HAS_JAX and self.f_jax is not None and self.ell_jax is not None
            self._jac_f_v = jax.jit(jax.jacrev(self.f_jax, argnums=0))
            self._grad_ell_v = jax.jit(jax.grad(self.ell_jax, argnums=0))

    def jac_f_v(self, v: np.ndarray, s: float) -> np.ndarray:
        if self.use_jax:
            vv = jnp.asarray(v, dtype=jnp.float64)
            ss = jnp.asarray(s, dtype=jnp.float64)
            J = np.asarray(self._jac_f_v(vv, ss))
            return J.astype(np.float64)

        # finite difference
        eps = float(self.spec.fd_eps)
        v = np.asarray(v, dtype=np.float64)
        base = self.f_np(v, s)
        J = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            dv = np.zeros((2,), dtype=np.float64)
            dv[i] = eps
            fp = self.f_np(v + dv, s)
            fm = self.f_np(v - dv, s)
            J[:, i] = (fp - fm) / (2.0 * eps)
        # sanity: use base to silence unused variable warnings
        _ = base
        return J

    def grad_ell_v(self, v: np.ndarray, s: float, t: int) -> np.ndarray:
        if self.use_jax:
            vv = jnp.asarray(v, dtype=jnp.float64)
            ss = jnp.asarray(s, dtype=jnp.float64)
            tt = jnp.asarray(t, dtype=jnp.int32)
            g = np.asarray(self._grad_ell_v(vv, ss, tt))
            return g.astype(np.float64)

        # finite difference
        eps = float(self.spec.fd_eps)
        v = np.asarray(v, dtype=np.float64)
        g = np.zeros((2,), dtype=np.float64)
        for i in range(2):
            dv = np.zeros((2,), dtype=np.float64)
            dv[i] = eps
            fp = self.ell_np(v + dv, s, t)
            fm = self.ell_np(v - dv, s, t)
            g[i] = (fp - fm) / (2.0 * eps)
        return g


###############################################################################
# PMP solver module
###############################################################################


@dataclass(frozen=True)
class ShootingSpec:
    mode: str = "single"  # "single" or "multiple"
    segments: int = 1  # for multiple shooting: number of segments K>=1


@dataclass(frozen=True)
class ControlMaximizationSpec:
    method: str = "grid_brent"  # "grid_brent" or "brent"
    grid_points: int = 25
    refine_top_k: int = 3
    maxiter: int = 60


@dataclass(frozen=True)
class FBSMSpec:
    max_sweeps: int = 60
    tol: float = 1e-6
    relaxation: float = 1.0  # damping; 1.0 = full update


@dataclass(frozen=True)
class RootSolveSpec:
    method: str = "least_squares"  # "least_squares" or "root"
    max_nfev: int = 120
    xtol: float = 1e-10
    ftol: float = 1e-10
    gtol: float = 1e-10
    multistart: int = 5
    seed: int = 0


@dataclass(frozen=True)
class SolverConfig:
    N: int
    v0: np.ndarray
    case_id: str = "A"

    objective: ObjectiveSpec = dataclasses.field(
        default_factory=lambda: ObjectiveSpec(name="sum(vy_t)", use_vy_next=False)
    )
    terminal_constraint: TerminalConstraint = dataclasses.field(
        default_factory=TerminalConstraint
    )

    feasibility: FeasibilitySpec = dataclasses.field(default_factory=FeasibilitySpec)
    differentiation: DifferentiationSpec = dataclasses.field(
        default_factory=DifferentiationSpec
    )

    shooting: ShootingSpec = dataclasses.field(default_factory=ShootingSpec)
    control_opt: ControlMaximizationSpec = dataclasses.field(
        default_factory=ControlMaximizationSpec
    )
    fbs: FBSMSpec = dataclasses.field(default_factory=FBSMSpec)
    root: RootSolveSpec = dataclasses.field(default_factory=RootSolveSpec)

    init_control: str = "mid"  # "mid", "zeros", "random"
    verbose: bool = True


@dataclass
class SolveResult:
    success: bool
    case_id: str
    case_name: str
    N: int
    v0: np.ndarray
    terminal_constraint: TerminalConstraint
    objective_name: str
    objective_value: float  # base objective (no penalties)
    objective_aug_value: float  # augmented objective used (includes penalties if soft)
    vx: np.ndarray
    vy: np.ndarray
    s: np.ndarray
    lambda_x: np.ndarray
    lambda_y: np.ndarray
    diagnostics: Dict[str, Any]
    message: str = ""


class PMPSolver:
    def __init__(self, cfg: SolverConfig):
        if cfg.N < 1:
            raise ValueError("N must be >= 1")
        self.cfg = cfg
        self.case = CASE_SPECS[cfg.case_id]

        self.f_np = make_f_case_np(self.case)
        self.f_jax = make_f_case_jax(self.case)

    # ---------------------------------------------------------------------
    # Objective helpers
    # ---------------------------------------------------------------------
    def _base_stage_cost_np(self, v: np.ndarray, s: float, t: int) -> float:
        """Base stage cost ℓ(v,s,t) to maximize (no penalties)."""
        obj = self.cfg.objective
        if obj.stage_cost_user is not None:
            v_next = self.f_np(v, s)
            return _call_user_stage_cost(obj.stage_cost_user, v, s, t, v_next)

        # default vy_t or vy_{t+1}
        if not obj.use_vy_next:
            return float(v[1])
        v_next = self.f_np(v, s)
        return float(v_next[1])

    def _penalty_np(self, v: np.ndarray) -> float:
        """Squared-hinge penalty magnitude (>=0)."""
        vy = float(v[1])
        tol = float(self.cfg.feasibility.vy_tol)
        if self.case.vy_nonneg:
            viol = max(0.0, tol - vy)
        else:
            viol = max(0.0, vy + tol)
        return float(viol * viol)

    def _make_ell_aug_np(
        self, penalty_weight: float
    ) -> Callable[[np.ndarray, float, int], float]:
        """Augmented stage cost ℓ_aug = ℓ_base - w * penalty(v)."""

        def ell(v: np.ndarray, s: float, t: int) -> float:
            base = self._base_stage_cost_np(v, s, t)
            if self.cfg.feasibility.mode == "soft" and penalty_weight > 0.0:
                base -= float(penalty_weight) * self._penalty_np(v)
            return float(base)

        return ell

    def _make_ell_aug_jax(self, penalty_weight: float) -> Optional[Callable]:
        """JAX version of ℓ_aug(v,s,t) (only for built-in objectives)."""
        if not HAS_JAX or self.f_jax is None:
            return None

        obj = self.cfg.objective
        if obj.stage_cost_user is not None:
            # user hook is generally not JAX-traceable; fallback to finite differences
            return None

        w = float(penalty_weight)
        tol = float(self.cfg.feasibility.vy_tol)
        vy_nonneg = bool(self.case.vy_nonneg)
        use_vy_next = bool(obj.use_vy_next)
        f_jax = self.f_jax

        def ell(v, s, t):
            # base
            if not use_vy_next:
                base = v[1]
            else:
                v_next = f_jax(v, s)
                base = v_next[1]

            if self.cfg.feasibility.mode == "soft" and w > 0.0:
                vy = v[1]
                if vy_nonneg:
                    viol = jnp.maximum(0.0, tol - vy)
                else:
                    viol = jnp.maximum(0.0, vy + tol)
                base = base - w * (viol * viol)
            return base

        return jax.jit(ell)

    def _objective_base(
        self, v_traj: np.ndarray, s_seq: np.ndarray, t_offset: int = 0
    ) -> float:
        total = 0.0
        for k in range(len(s_seq)):
            total += self._base_stage_cost_np(
                v_traj[k], float(s_seq[k]), int(t_offset + k)
            )
        return float(total)

    def _objective_aug(
        self,
        v_traj: np.ndarray,
        s_seq: np.ndarray,
        penalty_weight: float,
        t_offset: int = 0,
    ) -> float:
        ell_aug = self._make_ell_aug_np(penalty_weight)
        total = 0.0
        for k in range(len(s_seq)):
            total += ell_aug(v_traj[k], float(s_seq[k]), int(t_offset + k))
        return float(total)

    # ---------------------------------------------------------------------
    # Simulation helpers
    # ---------------------------------------------------------------------
    def _simulate_forward(self, v0: np.ndarray, s_seq: np.ndarray) -> np.ndarray:
        N = len(s_seq)
        v = np.zeros((N + 1, 2), dtype=np.float64)
        v[0] = np.asarray(v0, dtype=np.float64)
        for t in range(N):
            v[t + 1] = self.f_np(v[t], float(s_seq[t]))
        return v

    def _initial_control_guess(self, N: int) -> np.ndarray:
        lo, hi = self.case.s_bounds
        if self.cfg.init_control == "zeros":
            return np.zeros((N,), dtype=np.float64)
        if self.cfg.init_control == "random":
            rng = np.random.default_rng(self.cfg.root.seed)
            return rng.uniform(lo, hi, size=(N,)).astype(np.float64)
        # "mid"
        return np.full((N,), 0.5 * (lo + hi), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Per-step control maximization
    # ---------------------------------------------------------------------
    def _maximize_control(
        self,
        v_t: np.ndarray,
        lam_next: np.ndarray,
        t_global: int,
        ell_aug: Callable[[np.ndarray, float, int], float],
    ) -> float:
        """
        Compute:
           s* = argmax_{s∈[lo,hi]} [ ell_aug(v_t,s,t) + lam_nextᵀ f(v_t,s) ]
        using scalar bounded optimization.

        SciPy's bounded minimize_scalar uses a Brent-type method for local minimization on an interval.
        We add a coarse grid to get a more global-ish max in 1D.
        """
        lo, hi = self.case.s_bounds
        v_t = np.asarray(v_t, dtype=np.float64)
        lam_next = np.asarray(lam_next, dtype=np.float64)

        def H(s: float) -> float:
            v_next = self.f_np(v_t, s)
            return float(ell_aug(v_t, s, t_global) + float(lam_next @ v_next))

        method = self.cfg.control_opt.method
        if method == "brent":
            res = minimize_scalar(
                lambda ss: -H(ss),
                bounds=(lo, hi),
                method="bounded",
                options=dict(maxiter=int(self.cfg.control_opt.maxiter)),
            )
            if res.success:
                return float(res.x)
            return float(np.clip(0.5 * (lo + hi), lo, hi))

        # grid_brent
        grid_n = int(max(5, self.cfg.control_opt.grid_points))
        s_grid = np.linspace(lo, hi, grid_n)
        vals = np.array([H(float(sg)) for sg in s_grid], dtype=np.float64)

        k = int(max(1, min(self.cfg.control_opt.refine_top_k, grid_n)))
        top_idx = np.argsort(vals)[-k:][::-1]

        best_s = float(s_grid[top_idx[0]])
        best_val = float(vals[top_idx[0]])

        span = hi - lo
        step = span / max(1, grid_n - 1)

        for idx in top_idx:
            s0 = float(s_grid[idx])
            a = max(lo, s0 - 2.0 * step)
            b = min(hi, s0 + 2.0 * step)
            if b <= a + 1e-14:
                continue
            res = minimize_scalar(
                lambda ss: -H(ss),
                bounds=(a, b),
                method="bounded",
                options=dict(maxiter=int(self.cfg.control_opt.maxiter)),
            )
            if res.success:
                s_opt = float(res.x)
                v_opt = float(H(s_opt))
                if v_opt > best_val:
                    best_val, best_s = v_opt, s_opt

        return float(np.clip(best_s, lo, hi))

    # ---------------------------------------------------------------------
    # Forward-backward sweep (given terminal costate)
    # ---------------------------------------------------------------------
    def solve_given_lambdaN(
        self,
        lambdaN: np.ndarray,
        v0: np.ndarray,
        N: int,
        t_offset: int,
        penalty_weight: float,
        init_s: Optional[np.ndarray] = None,
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Forward-backward sweep method (FBSM) with fixed terminal costate λ_N.

        Returns:
          converged, v_traj, s_seq, lambda_traj, info
        """
        N = int(N)
        if init_s is None:
            s = self._initial_control_guess(N)
        else:
            s = np.asarray(init_s, dtype=np.float64).copy()
            if s.shape != (N,):
                raise ValueError(f"init_s shape {s.shape} does not match N={N}")

        ell_aug_np = self._make_ell_aug_np(penalty_weight)
        ell_aug_jax = self._make_ell_aug_jax(penalty_weight)

        diff = Differentiator(
            f_np=self.f_np,
            ell_np=ell_aug_np,
            spec=self.cfg.differentiation,
            f_jax=self.f_jax,
            ell_jax=ell_aug_jax,
        )

        lamN = np.asarray(lambdaN, dtype=np.float64)
        relax = float(self.cfg.fbs.relaxation)
        tol = float(self.cfg.fbs.tol)

        converged = False
        last_max_delta = np.inf
        sweeps = 0

        for it in range(int(self.cfg.fbs.max_sweeps)):
            sweeps = it + 1
            s_prev = s.copy()

            # forward
            v = self._simulate_forward(v0, s)

            # backward costates
            lam = np.zeros((N + 1, 2), dtype=np.float64)
            lam[N] = lamN
            for k in range(N - 1, -1, -1):
                t_global = int(t_offset + k)
                grad_l = diff.grad_ell_v(v[k], float(s[k]), t_global)
                jac_f = diff.jac_f_v(v[k], float(s[k]))
                lam[k] = grad_l + jac_f.T @ lam[k + 1]

            # update controls
            for k in range(N):
                t_global = int(t_offset + k)
                s_star = self._maximize_control(v[k], lam[k + 1], t_global, ell_aug_np)
                s[k] = relax * s_star + (1.0 - relax) * s[k]

            max_delta = float(np.max(np.abs(s - s_prev)))
            last_max_delta = max_delta
            if max_delta < tol:
                converged = True
                break

        # final pass for output
        v = self._simulate_forward(v0, s)
        lam = np.zeros((N + 1, 2), dtype=np.float64)
        lam[N] = lamN
        for k in range(N - 1, -1, -1):
            t_global = int(t_offset + k)
            grad_l = diff.grad_ell_v(v[k], float(s[k]), t_global)
            jac_f = diff.jac_f_v(v[k], float(s[k]))
            lam[k] = grad_l + jac_f.T @ lam[k + 1]

        info = dict(
            fbs_sweeps=sweeps,
            fbs_last_max_delta=last_max_delta,
            diff_method=("jax" if diff.use_jax else "finite"),
        )
        return converged, v, s, lam, info

    # ---------------------------------------------------------------------
    # Single shooting solve (m=0 or low-dim mu root solve)
    # ---------------------------------------------------------------------
    def solve_single_shooting(self) -> SolveResult:
        tc = self.cfg.terminal_constraint
        m = tc.m()
        v0 = self.cfg.v0
        N = self.cfg.N

        # penalty schedule (soft only)
        if self.cfg.feasibility.mode == "soft":
            weights = [
                self.cfg.feasibility.penalty_weight
                * (self.cfg.feasibility.penalty_growth**k)
                for k in range(int(max(1, self.cfg.feasibility.penalty_rounds)))
            ]
        else:
            weights = [0.0]

        def finalize(
            converged: bool,
            v: np.ndarray,
            s: np.ndarray,
            lam: np.ndarray,
            root_diag: Dict[str, Any],
            msg: str,
            penalty_weight_used: float,
        ) -> SolveResult:
            obj_base = self._objective_base(v, s, t_offset=0)
            obj_aug = self._objective_aug(v, s, penalty_weight_used, t_offset=0)

            term_res = tc.residual(v[-1])
            term_err = float(np.linalg.norm(term_res)) if term_res.size else 0.0
            n_viol, max_viol = count_vy_violations(
                v, self.case, self.cfg.feasibility.vy_tol
            )

            diag = dict(
                **root_diag,
                terminal_residual=term_res,
                terminal_error_norm=term_err,
                vy_min=float(np.min(v[:, 1])),
                vy_max=float(np.max(v[:, 1])),
                s_min=float(np.min(s)),
                s_max=float(np.max(s)),
                vy_violations=n_viol,
                vy_max_violation=max_viol,
                converged_fbs=bool(converged),
            )
            # Hamiltonian along trajectory (optional diagnostics)
            H_base = np.zeros((len(s),), dtype=np.float64)
            H_aug = np.zeros((len(s),), dtype=np.float64)
            ell_aug_np = self._make_ell_aug_np(penalty_weight_used)
            for tt in range(len(s)):
                v_next = v[tt + 1]
                H_base[tt] = self._base_stage_cost_np(
                    v[tt], float(s[tt]), int(tt)
                ) + float(lam[tt + 1] @ v_next)
                H_aug[tt] = float(
                    ell_aug_np(v[tt], float(s[tt]), int(tt))
                    + float(lam[tt + 1] @ v_next)
                )
            diag["hamiltonian_base"] = H_base
            diag["hamiltonian_aug"] = H_aug

            success = bool(converged)
            if m > 0:
                success = success and (term_err < 1e-6)
            if self.cfg.feasibility.mode == "hard":
                success = success and (n_viol == 0)

            return SolveResult(
                success=success,
                case_id=self.case.case_id,
                case_name=self.case.name,
                N=int(N),
                v0=v0.copy(),
                terminal_constraint=tc,
                objective_name=self.cfg.objective.name,
                objective_value=float(obj_base),
                objective_aug_value=float(obj_aug),
                vx=v[:, 0].copy(),
                vy=v[:, 1].copy(),
                s=s.copy(),
                lambda_x=lam[:, 0].copy(),
                lambda_y=lam[:, 1].copy(),
                diagnostics=diag,
                message=msg,
            )

        # m = 0: free terminal, λ_N = 0
        if m == 0:
            lamN = np.zeros((2,), dtype=np.float64)
            init_s = self._initial_control_guess(N)
            t0 = time.perf_counter()
            conv, v, s, lam, info = self.solve_given_lambdaN(
                lamN, v0=v0, N=N, t_offset=0, penalty_weight=weights[-1], init_s=init_s
            )
            info["wall_time_sec"] = float(time.perf_counter() - t0)
            last_info = info  # capture diagnostics from this sweep
            return finalize(
                conv,
                v,
                s,
                lam,
                root_diag=info,
                msg="single_shooting (m=0)",
                penalty_weight_used=weights[-1],
            )

        # m > 0: solve for terminal multiplier mu in R^m
        rng = np.random.default_rng(self.cfg.root.seed)
        x0_list: List[np.ndarray] = [np.zeros((m,), dtype=np.float64)]
        for _ in range(max(0, int(self.cfg.root.multistart) - 1)):
            x0_list.append(rng.normal(scale=0.1, size=(m,)).astype(np.float64))

        best: Optional[SolveResult] = None

        for x0 in x0_list:
            mu = x0.copy()
            init_s = self._initial_control_guess(N)

            conv = False
            v = np.zeros((N + 1, 2), dtype=np.float64)
            s = init_s.copy()
            lam = np.zeros((N + 1, 2), dtype=np.float64)
            root_diag: Dict[str, Any] = {}
            last_info: Dict[str, Any] = {}  # last inner FBSM diagnostics

            t_start = time.perf_counter()

            for w in weights:
                # residual function for root/least_squares
                def residual(mu_vec: np.ndarray) -> np.ndarray:
                    nonlocal conv, v, s, lam, init_s, last_info
                    lamN = tc.lambdaN_from_mu(mu_vec)
                    conv, v, s, lam, info = self.solve_given_lambdaN(
                        lamN, v0=v0, N=N, t_offset=0, penalty_weight=w, init_s=init_s
                    )
                    last_info = info  # capture diagnostics from this sweep
                    init_s = s.copy()  # warm-start next residual eval

                    r_term = tc.residual(v[-1])

                    if self.cfg.feasibility.mode == "hard":
                        n_viol, max_viol = count_vy_violations(
                            v, self.case, self.cfg.feasibility.vy_tol
                        )
                        if n_viol > 0:
                            return r_term + (1e3 + max_viol) * np.ones_like(r_term)

                    if self.cfg.feasibility.mode == "soft":
                        # augment residual with scaled violations to help least-squares steer
                        viol = vy_violation_magnitudes(
                            v, self.case, self.cfg.feasibility.vy_tol
                        )
                        return np.concatenate([r_term, np.sqrt(max(w, 0.0)) * viol])

                    return r_term

                if self.cfg.root.method == "root":
                    # root expects square system; only safe in hard mode (no extra residuals)
                    if self.cfg.feasibility.mode == "soft":
                        raise ValueError(
                            "root method is incompatible with soft penalty residual augmentation. Use least_squares."
                        )
                    sol = root(
                        residual,
                        mu,
                        method="hybr",
                        tol=float(self.cfg.root.xtol),
                        options=dict(maxfev=int(self.cfg.root.max_nfev)),
                    )
                    mu = np.asarray(sol.x, dtype=np.float64)
                    root_diag.update(
                        dict(
                            root_success=bool(sol.success),
                            root_status=int(sol.status),
                            root_message=str(sol.message),
                            root_nfev=int(sol.nfev),
                            penalty_weight=float(w),
                        )
                    )
                else:
                    ls = least_squares(
                        residual,
                        mu,
                        method="trf",
                        max_nfev=int(self.cfg.root.max_nfev),
                        xtol=float(self.cfg.root.xtol),
                        ftol=float(self.cfg.root.ftol),
                        gtol=float(self.cfg.root.gtol),
                    )
                    mu = np.asarray(ls.x, dtype=np.float64)
                    root_diag.update(
                        dict(
                            ls_success=bool(ls.success),
                            ls_status=int(ls.status),
                            ls_message=str(ls.message),
                            ls_nfev=int(ls.nfev),
                            ls_cost=float(ls.cost),
                            penalty_weight=float(w),
                        )
                    )

                # warm-start controls for next penalty round
                init_s = s.copy()

            root_diag.update(last_info)
            root_diag["wall_time_sec"] = float(time.perf_counter() - t_start)
            # merge last inner info if available
            try:
                root_diag.update(info)
            except Exception:
                pass

            res = finalize(
                conv,
                v,
                s,
                lam,
                root_diag=root_diag,
                msg="single_shooting",
                penalty_weight_used=weights[-1],
            )

            if best is None:
                best = res
            else:
                # choose feasible first, then higher base objective
                if (res.success and not best.success) or (
                    res.success == best.success
                    and res.objective_value > best.objective_value
                ):
                    best = res

        assert best is not None
        return best

    # ---------------------------------------------------------------------
    # Multiple shooting solve
    # ---------------------------------------------------------------------
    def solve_multiple_shooting(self) -> SolveResult:
        tc = self.cfg.terminal_constraint
        m = tc.m()
        v0 = self.cfg.v0
        N = int(self.cfg.N)
        K_req = int(max(1, self.cfg.shooting.segments))

        if m == 0 or K_req <= 1:
            return self.solve_single_shooting()

        # Build segment cuts
        cuts = np.linspace(0, N, K_req + 1)
        cuts = np.round(cuts).astype(int)
        cuts[0] = 0
        cuts[-1] = N
        cuts = np.unique(cuts)
        if len(cuts) < 2:
            return self.solve_single_shooting()
        seg_starts = cuts[:-1]
        seg_ends = cuts[1:]
        seg_lens = seg_ends - seg_starts
        K = len(seg_lens)

        if np.any(seg_lens <= 0):
            return self.solve_single_shooting()

        # Unknown vector z:
        #   boundary states x_{t_k} for k=1..K-1   => 2*(K-1)
        #   boundary costates λ_{t_k} for k=1..K-1 => 2*(K-1)
        #   terminal multiplier mu (m,)
        n_x = 2 * (K - 1)
        n_l = 2 * (K - 1)
        n_mu = m
        n = n_x + n_l + n_mu

        # nominal forward sim for x guess
        s_nom = self._initial_control_guess(N)
        v_nom = self._simulate_forward(v0, s_nom)

        z0 = np.zeros((n,), dtype=np.float64)
        # x boundary guesses
        for k in range(1, K):
            z0[2 * (k - 1) : 2 * (k - 1) + 2] = v_nom[seg_starts[k]]
        # λ boundaries start at zero; mu zero

        def unpack(z: np.ndarray):
            z = np.asarray(z, dtype=np.float64)
            xb = z[:n_x].reshape((K - 1, 2)) if K > 1 else np.zeros((0, 2))
            lb = z[n_x : n_x + n_l].reshape((K - 1, 2)) if K > 1 else np.zeros((0, 2))
            mu = z[n_x + n_l :]
            return xb, lb, mu

        # penalty schedule (soft only)
        if self.cfg.feasibility.mode == "soft":
            weights = [
                self.cfg.feasibility.penalty_weight
                * (self.cfg.feasibility.penalty_growth**k)
                for k in range(int(max(1, self.cfg.feasibility.penalty_rounds)))
            ]
        else:
            weights = [0.0]

        rng = np.random.default_rng(self.cfg.root.seed)
        z0_list: List[np.ndarray] = [z0]
        for _ in range(max(0, int(self.cfg.root.multistart) - 1)):
            z0_list.append(z0 + rng.normal(scale=0.01, size=z0.shape))

        best: Optional[SolveResult] = None

        for z_init in z0_list:
            z = z_init.copy()
            root_diag: Dict[str, Any] = {}
            last_info: Dict[str, Any] = {}  # last inner FBSM diagnostics
            v_full = None
            s_full = None
            lam_full = None
            conv_all = False

            t_start = time.perf_counter()

            for w in weights:
                # per-segment control warm starts
                init_s_segs = [self._initial_control_guess(int(L)) for L in seg_lens]

                def residual(z_vec: np.ndarray) -> np.ndarray:
                    nonlocal v_full, s_full, lam_full, conv_all
                    xb, lb, mu = unpack(z_vec)
                    lamN = tc.lambdaN_from_mu(mu)

                    # boundary states list (x0 fixed)
                    x_bound = [v0.copy()] + [xb[k - 1].copy() for k in range(1, K)]
                    # boundary costates list (for k=1..K-1)
                    lam_bound = [lb[k - 1].copy() for k in range(1, K)]

                    res_parts: List[np.ndarray] = []
                    conv_all = True

                    v_accum: List[np.ndarray] = [x_bound[0].copy()]
                    s_accum: List[float] = []
                    lam_accum: List[np.ndarray] = []

                    for k in range(K):
                        v_start = x_bound[k]
                        lam_end = lam_bound[k] if k < K - 1 else lamN
                        t_off = int(seg_starts[k])
                        L = int(seg_lens[k])

                        conv_k, v_seg, s_seg, lam_seg, info = self.solve_given_lambdaN(
                            lam_end,
                            v0=v_start,
                            N=L,
                            t_offset=t_off,
                            penalty_weight=w,
                            init_s=init_s_segs[k],
                        )
                        conv_all = conv_all and conv_k

                        v_end = v_seg[-1]
                        lam_start = lam_seg[0]

                        # state continuity residual
                        if k < K - 1:
                            res_parts.append(v_end - x_bound[k + 1])
                        else:
                            res_parts.append(tc.residual(v_end))

                        # costate continuity residual (λ at boundary t_k), skip k=0 (λ_0 free)
                        if k > 0:
                            res_parts.append(lam_start - lam_bound[k - 1])

                        # hard feasibility rejection
                        if self.cfg.feasibility.mode == "hard":
                            n_viol, max_viol = count_vy_violations(
                                v_seg, self.case, self.cfg.feasibility.vy_tol
                            )
                            if n_viol > 0:
                                res_parts = [
                                    rp + (1e3 + max_viol) * np.ones_like(rp)
                                    for rp in res_parts
                                ]
                                break

                        if self.cfg.feasibility.mode == "soft":
                            viol = vy_violation_magnitudes(
                                v_seg, self.case, self.cfg.feasibility.vy_tol
                            )
                            res_parts.append(np.sqrt(max(w, 0.0)) * viol)

                        # accumulate trajectory pieces
                        v_accum.extend(v_seg[1:])
                        s_accum.extend(list(s_seg))
                        lam_accum.extend(list(lam_seg[:-1]))  # omit segment end costate

                        init_s_segs[k] = s_seg.copy()  # warm start next eval

                    if len(v_accum) == N + 1 and len(s_accum) == N:
                        v_full = np.asarray(v_accum, dtype=np.float64)
                        s_full = np.asarray(s_accum, dtype=np.float64)
                        lam_full = np.vstack(
                            [
                                np.asarray(lam_accum, dtype=np.float64),
                                np.asarray(lamN, dtype=np.float64),
                            ]
                        )

                    return np.concatenate([rp.ravel() for rp in res_parts])

                # solve square system
                if self.cfg.root.method == "root":
                    if self.cfg.feasibility.mode == "soft":
                        raise ValueError(
                            "root method incompatible with soft penalty augmentation; use least_squares."
                        )
                    sol = root(
                        residual,
                        z,
                        method="hybr",
                        tol=float(self.cfg.root.xtol),
                        options=dict(maxfev=int(self.cfg.root.max_nfev)),
                    )
                    z = np.asarray(sol.x, dtype=np.float64)
                    root_diag.update(
                        dict(
                            root_success=bool(sol.success),
                            root_status=int(sol.status),
                            root_message=str(sol.message),
                            root_nfev=int(sol.nfev),
                            penalty_weight=float(w),
                            segments=int(K),
                        )
                    )
                else:
                    ls = least_squares(
                        residual,
                        z,
                        method="trf",
                        max_nfev=int(self.cfg.root.max_nfev),
                        xtol=float(self.cfg.root.xtol),
                        ftol=float(self.cfg.root.ftol),
                        gtol=float(self.cfg.root.gtol),
                    )
                    z = np.asarray(ls.x, dtype=np.float64)
                    root_diag.update(
                        dict(
                            ls_success=bool(ls.success),
                            ls_status=int(ls.status),
                            ls_message=str(ls.message),
                            ls_nfev=int(ls.nfev),
                            ls_cost=float(ls.cost),
                            penalty_weight=float(w),
                            segments=int(K),
                        )
                    )

            root_diag.update(last_info)
            root_diag["wall_time_sec"] = float(time.perf_counter() - t_start)

            if v_full is None or s_full is None or lam_full is None:
                # fall back
                res = self.solve_single_shooting()
                res.message = "multiple_shooting failed to reconstruct -> fallback single_shooting"
                return res

            # final diagnostics
            obj_base = self._objective_base(v_full, s_full, t_offset=0)
            obj_aug = self._objective_aug(v_full, s_full, weights[-1], t_offset=0)
            term_res = tc.residual(v_full[-1])
            term_err = float(np.linalg.norm(term_res)) if term_res.size else 0.0
            n_viol, max_viol = count_vy_violations(
                v_full, self.case, self.cfg.feasibility.vy_tol
            )

            success = bool(conv_all) and (term_err < 1e-6)
            if self.cfg.feasibility.mode == "hard":
                success = success and (n_viol == 0)

            diag = dict(
                **root_diag,
                terminal_residual=term_res,
                terminal_error_norm=term_err,
                vy_min=float(np.min(v_full[:, 1])),
                vy_max=float(np.max(v_full[:, 1])),
                s_min=float(np.min(s_full)),
                s_max=float(np.max(s_full)),
                vy_violations=n_viol,
                vy_max_violation=max_viol,
                converged_fbs=bool(conv_all),
            )
            # Hamiltonian along trajectory (optional diagnostics)
            H_base = np.zeros((len(s_full),), dtype=np.float64)
            H_aug = np.zeros((len(s_full),), dtype=np.float64)
            ell_aug_np = self._make_ell_aug_np(weights[-1])
            for tt in range(len(s_full)):
                v_next = v_full[tt + 1]
                H_base[tt] = self._base_stage_cost_np(
                    v_full[tt], float(s_full[tt]), int(tt)
                ) + float(lam_full[tt + 1] @ v_next)
                H_aug[tt] = float(
                    ell_aug_np(v_full[tt], float(s_full[tt]), int(tt))
                    + float(lam_full[tt + 1] @ v_next)
                )
            diag["hamiltonian_base"] = H_base
            diag["hamiltonian_aug"] = H_aug

            res = SolveResult(
                success=success,
                case_id=self.case.case_id,
                case_name=self.case.name,
                N=int(N),
                v0=v0.copy(),
                terminal_constraint=tc,
                objective_name=self.cfg.objective.name,
                objective_value=float(obj_base),
                objective_aug_value=float(obj_aug),
                vx=v_full[:, 0].copy(),
                vy=v_full[:, 1].copy(),
                s=s_full.copy(),
                lambda_x=lam_full[:, 0].copy(),
                lambda_y=lam_full[:, 1].copy(),
                diagnostics=diag,
                message="multiple_shooting",
            )

            if best is None:
                best = res
            else:
                if (res.success and not best.success) or (
                    res.success == best.success
                    and res.objective_value > best.objective_value
                ):
                    best = res

        assert best is not None
        return best

    def solve(self) -> SolveResult:
        if self.cfg.shooting.mode == "multiple":
            return self.solve_multiple_shooting()
        return self.solve_single_shooting()


###############################################################################
# Runner / CLI
###############################################################################


def print_result_summary(results: List[SolveResult]) -> pd.DataFrame:
    """Print result summary as a formatted DataFrame."""
    data = []
    for r in results:
        d = r.diagnostics
        term_err = float(d.get("terminal_error_norm", 0.0))
        time_sec = float(d.get("wall_time_sec", 0.0))
        viol = int(d.get("vy_violations", 0))
        diff = str(d.get("diff_method", "?"))
        data.append({
            "case": r.case_id,
            "success": "yes" if r.success else "no",
            "obj": r.objective_value,
            "obj_aug": r.objective_aug_value,
            "term_err": term_err,
            "vy_min": float(d.get("vy_min", np.nan)),
            "vy_max": float(d.get("vy_max", np.nan)),
            "s_min": float(d.get("s_min", np.nan)),
            "s_max": float(d.get("s_max", np.nan)),
            "viol": viol,
            "time_sec": time_sec,
            "diff": diff[:6],
        })
    
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\n" + df.to_string(index=False))
    return df


def plot_optimum_all_cases(
    results: List[SolveResult], save_path: Optional[str] = None, show: bool = False
) -> None:
    """Plot vx, vy, s curves in time for all cases (4 subplots) at the optimum."""
    # Sort results by case_id to ensure consistent order (A, B, C, D)
    results_sorted = sorted(results, key=lambda r: r.case_id)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    
    for idx, res in enumerate(results_sorted):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Time arrays
        t_state = np.arange(res.N + 1)
        t_ctrl = np.arange(res.N)
        
        # Get case info for interpretable labels
        case = CASE_SPECS[res.case_id]
        vy_label = "vy>=0" if case.vy_nonneg else "vy<0"
        s_label = "s<=0" if case.s_bounds[1] <= 0 else "s>0"
        title = f"Case {res.case_id}: {vy_label}, {s_label}\nobj={res.objective_value:.6g} | success={res.success}"
        
        # Plot vx and vy on left y-axis
        ax.plot(t_state, res.vx, label="vx", color="tab:orange", linewidth=1.5)
        ax.plot(t_state, res.vy, label="vy", color="tab:blue", linewidth=1.5)
        ax.set_xlabel("t", fontsize=10)
        ax.set_ylabel("vx, vy", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        
        # Plot s on right y-axis
        ax2 = ax.twinx()
        ax2.plot(t_ctrl, res.s, label="s", color="tab:green", linewidth=1.5, linestyle="--")
        ax2.set_ylabel("s", fontsize=10, color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def maybe_plot(
    res: SolveResult, show: bool = True, save_prefix: Optional[str] = None
) -> None:
    t_state = np.arange(res.N + 1)
    t_ctrl = np.arange(res.N)

    fig, ax = plt.subplots()
    ax.plot(t_state, res.vx, label="vx")
    ax.plot(t_state, res.vy, label="vy")
    ax.set_xlabel("t")
    ax.set_title(
        f"Case {res.case_id}: {res.case_name}\nobj={res.objective_value:.6g} | success={res.success}"
    )
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(t_ctrl, res.s, label="s")
    ax2.set_xlabel("t")
    ax2.set_title("control")
    ax2.legend()
    fig2.tight_layout()

    # Optional Hamiltonian plot (if available in diagnostics)
    H_base = res.diagnostics.get("hamiltonian_base", None)
    H_aug = res.diagnostics.get("hamiltonian_aug", None)
    fig3 = ax3 = None
    if H_base is not None:
        fig3, ax3 = plt.subplots()
        ax3.plot(t_ctrl, H_base, label="H_base")
        if H_aug is not None:
            ax3.plot(t_ctrl, H_aug, label="H_aug")
        ax3.set_xlabel("t")
        ax3.set_title("Hamiltonian")
        ax3.legend()
        fig3.tight_layout()

    if save_prefix is not None:
        fig.savefig(save_prefix + "_state.png", dpi=150)
        fig2.savefig(save_prefix + "_control.png", dpi=150)
        if fig3 is not None:
            fig3.savefig(save_prefix + "_hamiltonian.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-branch PMP solver (2D state, 1D control)"
    )

    p.add_argument("--N", type=int, default=20, help="horizon length N>=1")
    p.add_argument(
        "--v0", type=float, nargs=2, default=[1.0, 0.1], metavar=("vx0", "vy0")
    )

    p.add_argument(
        "--setup",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Boundary setup: 1 free end; 2 bridge full; 3 partial end; 4 go anywhere",
    )

    p.add_argument(
        "--vT",
        type=float,
        nargs=2,
        default=None,
        metavar=("vxT", "vyT"),
        help="terminal target vT (needed for setups 2/3)",
    )
    p.add_argument(
        "--terminal_component",
        type=str,
        default="vy",
        choices=["vx", "vy"],
        help="For setup 3: which component to constrain at time N",
    )

    p.add_argument(
        "--cases",
        type=str,
        default="A,B,C,D",
        help="Comma-separated cases to run (subset of A,B,C,D)",
    )

    # objective
    p.add_argument(
        "--use_vy_next",
        action="store_true",
        help="maximize sum vy_{t+1} instead of vy_t",
    )

    # feasibility
    p.add_argument("--feas_mode", type=str, default="hard", choices=["hard", "soft"])
    p.add_argument("--vy_tol", type=float, default=0.0)
    p.add_argument("--penalty_weight", type=float, default=1.0)
    p.add_argument("--penalty_growth", type=float, default=10.0)
    p.add_argument("--penalty_rounds", type=int, default=1)

    # differentiation
    p.add_argument(
        "--diff", type=str, default="auto", choices=["auto", "jax", "finite"]
    )
    p.add_argument("--fd_eps", type=float, default=1e-6)

    # shooting
    p.add_argument(
        "--shooting", type=str, default="single", choices=["single", "multiple"]
    )
    p.add_argument("--segments", type=int, default=4)

    # FBSM
    p.add_argument("--fbs_max_sweeps", type=int, default=60)
    p.add_argument("--fbs_tol", type=float, default=1e-6)
    p.add_argument("--relax", type=float, default=1.0)

    # root
    p.add_argument(
        "--root_method",
        type=str,
        default="least_squares",
        choices=["least_squares", "root"],
    )
    p.add_argument("--root_max_nfev", type=int, default=120)
    p.add_argument("--multistart", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # control maximization
    p.add_argument(
        "--ctrl_method", type=str, default="grid_brent", choices=["grid_brent", "brent"]
    )
    p.add_argument("--grid_points", type=int, default=25)
    p.add_argument("--refine_top_k", type=int, default=3)
    p.add_argument("--ctrl_maxiter", type=int, default=60)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def build_terminal_constraint(args: argparse.Namespace) -> TerminalConstraint:
    setup = int(args.setup)
    if setup in (1, 4):
        return TerminalConstraint(vT=None, mask=np.array([False, False], dtype=bool))
    if setup == 2:
        if args.vT is None:
            raise ValueError("setup 2 requires --vT vxT vyT")
        return TerminalConstraint(
            vT=np.array(args.vT, dtype=np.float64),
            mask=np.array([True, True], dtype=bool),
        )
    if setup == 3:
        if args.vT is None:
            raise ValueError(
                "setup 3 requires --vT (vxT vyT) even if one component is unused"
            )
        mask = np.array([False, False], dtype=bool)
        if args.terminal_component == "vx":
            mask[0] = True
        else:
            mask[1] = True
        return TerminalConstraint(vT=np.array(args.vT, dtype=np.float64), mask=mask)
    raise ValueError("unknown setup")


def main() -> None:
    args = parse_args()

    v0 = np.array(args.v0, dtype=np.float64)
    tc = build_terminal_constraint(args)

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in CASE_SPECS:
            raise ValueError(f"Unknown case '{c}'. Must be among A,B,C,D.")

    obj_name = "sum(vy_{t+1})" if bool(args.use_vy_next) else "sum(vy_t)"
    objective = ObjectiveSpec(
        name=obj_name, use_vy_next=bool(args.use_vy_next), stage_cost_user=None
    )

    results: List[SolveResult] = []

    for case_id in cases:
        cfg = SolverConfig(
            N=int(args.N),
            v0=v0.copy(),
            case_id=case_id,
            objective=objective,
            terminal_constraint=tc,
            feasibility=FeasibilitySpec(
                mode=str(args.feas_mode),
                vy_tol=float(args.vy_tol),
                penalty_weight=float(args.penalty_weight),
                penalty_growth=float(args.penalty_growth),
                penalty_rounds=int(args.penalty_rounds),
            ),
            differentiation=DifferentiationSpec(
                method=str(args.diff), fd_eps=float(args.fd_eps)
            ),
            shooting=ShootingSpec(mode=str(args.shooting), segments=int(args.segments)),
            control_opt=ControlMaximizationSpec(
                method=str(args.ctrl_method),
                grid_points=int(args.grid_points),
                refine_top_k=int(args.refine_top_k),
                maxiter=int(args.ctrl_maxiter),
            ),
            fbs=FBSMSpec(
                max_sweeps=int(args.fbs_max_sweeps),
                tol=float(args.fbs_tol),
                relaxation=float(args.relax),
            ),
            root=RootSolveSpec(
                method=str(args.root_method),
                max_nfev=int(args.root_max_nfev),
                multistart=int(args.multistart),
                seed=int(args.seed),
            ),
            verbose=bool(args.verbose),
        )

        solver = PMPSolver(cfg)
        res = solver.solve()
        results.append(res)

        if cfg.verbose:
            print(f"\n=== Case {res.case_id} ({res.case_name}) ===")
            print(f"success: {res.success}")
            print(f"objective ({res.objective_name}): {res.objective_value:.6g}")
            print(f"objective_aug: {res.objective_aug_value:.6g}")
            print(
                f"terminal mask: {res.terminal_constraint.mask}, vT: {res.terminal_constraint.vT}"
            )
            print(
                f"terminal error norm: {res.diagnostics.get('terminal_error_norm', 0.0):.3e}"
            )
            print(
                f"vy range: [{res.diagnostics.get('vy_min', np.nan):.6g}, {res.diagnostics.get('vy_max', np.nan):.6g}]"
            )
            print(
                f"s  range: [{res.diagnostics.get('s_min', np.nan):.6g}, {res.diagnostics.get('s_max', np.nan):.6g}]"
            )
            print(
                f"vy violations: {res.diagnostics.get('vy_violations', 0)} (max {res.diagnostics.get('vy_max_violation', 0.0):.3e})"
            )
            print(f"diff backend: {res.diagnostics.get('diff_method', '?')}")
            if res.diagnostics.get("ls_nfev") is not None:
                print(
                    f"least_squares nfev: {res.diagnostics.get('ls_nfev')} | cost: {res.diagnostics.get('ls_cost'):.3e}"
                )

        if bool(args.plot):
            maybe_plot(res, show=True)

    # Print summary as DataFrame
    print_result_summary(results)
    
    # Create output directory and plot optimum results
    output_dir = Path("outputs") / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot all cases at optimum
    plot_optimum_all_cases(
        results,
        save_path=str(output_dir / "pmp_optimum.png"),
        show=False
    )


if __name__ == "__main__":
    main()
