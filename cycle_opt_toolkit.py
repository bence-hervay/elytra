#!/usr/bin/env python3
"""
Cycle optimal control (discrete-time) — CasADi + IPOPT (regime-fixing)

This is a *fixed* version of the previous toolkit script: it now uses the same
practical optimization workflow that produced good solutions before:
  - regime-fixing outer loop (no if_else needed)
  - multi-start initialization
  - a built-in "known good" warm-start control for N=255 (resampled to any N)
  - a final polishing solve from the best control found

You can add arbitrary extra constraints by providing an `extra_constraints` callback.
See the commented example in `main()`.

Outputs (when run as a script):
  cycle_opt_outputs/solution_N{N}.csv
  cycle_opt_outputs/trials_N{N}.csv
  cycle_opt_outputs/timeseries_N{N}.html
  cycle_opt_outputs/phase_N{N}.html

Dependencies: numpy, pandas, casadi, plotly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import casadi as ca
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# Built-in warm start (good unconstrained N=255 solution control)
# ----------------------------

KNOWN_GOOD_S_255 = np.array(
    [
        0.999999999967226,
        0.98489864786308,
        0.95778302625986,
        0.931874648308411,
        0.907093111509824,
        0.883365034153665,
        0.860623296705803,
        0.838806379832651,
        0.817857784977157,
        0.797725525642058,
        0.77836167944237,
        0.7597219925586,
        0.741765529517223,
        0.724454362297847,
        0.707753293658673,
        0.691629610316761,
        0.676052862243603,
        0.660994664861237,
        0.646428521366635,
        0.63232966278661,
        0.618674903683225,
        0.605442511700146,
        0.592612089371034,
        0.580164466808413,
        0.56808160406046,
        0.556346502068292,
        0.544943121281091,
        0.533856307093706,
        0.52307172136379,
        0.512575779344945,
        0.50235559144066,
        0.492398909242205,
        0.482694075363483,
        0.473229976627847,
        0.463996000196978,
        0.45498199226058,
        0.446178218928252,
        0.437575328981818,
        0.42916431815758,
        0.420936494633488,
        0.412883445395739,
        0.404997003152458,
        0.397269213448222,
        0.389692301611259,
        0.382258639134002,
        0.374960709045547,
        0.367791069779252,
        0.360742316967296,
        0.353807042502725,
        0.346977790093297,
        0.340247006383821,
        0.333606986536097,
        0.32704981291648,
        0.320567285235522,
        0.314150840091529,
        0.307791457362809,
        0.301479550234244,
        0.295204834781031,
        0.288956173894315,
        0.282721388819534,
        0.276487029546262,
        0.270238092534124,
        0.263957670486911,
        0.257626513661069,
        0.251222474866306,
        0.244719799902595,
        0.238088210167372,
        0.231291702190342,
        0.224286956127332,
        0.217021195596126,
        0.209429264358613,
        0.201429563587327,
        0.192918295681748,
        0.183761130217362,
        0.17378083794218,
        0.162738420506011,
        0.150303367031049,
        0.136004963204844,
        0.119148921606071,
        0.0986666852694724,
        0.0728242157106082,
        0.038609550680429,
        -5.03998989116156e-06,
        -5.70168640057317e-06,
        -6.74548102719018e-06,
        -8.76567635277974e-06,
        -1.60775955331854e-05,
        -0.999999999817077,
        -0.999999999756643,
        -0.962366140045191,
        -0.776238455032111,
        -0.627464075138947,
        -0.524336741542843,
        -0.462550990500806,
        -0.430579563519232,
        -0.416663955471645,
        -0.412431383455008,
        -0.41300744828153,
        -0.415855467686233,
        -0.419731740646297,
        -0.424045531016463,
        -0.428521450297059,
        -0.43303297760128,
        -0.437522979485049,
        -0.441966389596747,
        -0.446352851021094,
        -0.450678695281353,
        -0.454943255525778,
        -0.459147180854604,
        -0.463291670382659,
        -0.467378128170102,
        -0.47140800985378,
        -0.47538275613266,
        -0.479303765337814,
        -0.483172383418617,
        -0.486989901565568,
        -0.490757557082398,
        -0.494476535555324,
        -0.498147973461416,
        -0.501772960847161,
        -0.50535254392483,
        -0.508887727529596,
        -0.512379477421497,
        -0.51582872243317,
        -0.51923635647129,
        -0.522603240381707,
        -0.525930203688634,
        -0.529218046217717,
        -0.532467539612243,
        -0.535679428750952,
        -0.538854433074906,
        -0.541993247830865,
        -0.545096545236995,
        -0.548164975577253,
        -0.551199168229352,
        -0.554199732631418,
        -0.557167259191637,
        -0.560102320144956,
        -0.563005470360682,
        -0.565877248104227,
        -0.568718175756265,
        -0.571528760492099,
        -0.574309494924036,
        -0.577060857708976,
        -0.579783314123779,
        -0.58247731661025,
        -0.585143305291839,
        -0.587781708463665,
        -0.59039294305777,
        -0.59297741508492,
        -0.595535520054432,
        -0.598067643373399,
        -0.600574160726577,
        -0.603055438437884,
        -0.605511833814806,
        -0.607943695476634,
        -0.610351363667194,
        -0.612735170553565,
        -0.615095440510752,
        -0.617432490393864,
        -0.619746629797875,
        -0.622038161306096,
        -0.624307380727612,
        -0.626554577324527,
        -0.628780034029455,
        -0.630984027653675,
        -0.63316682908655,
        -0.63532870348678,
        -0.637469910465466,
        -0.639590704261941,
        -0.641691333912291,
        -0.643772043411238,
        -0.645833071867471,
        -0.64787465365282,
        -0.649897018545764,
        -0.651900391869168,
        -0.653884994622913,
        -0.655851043611106,
        -0.657798751565157,
        -0.659728327262214,
        -0.66163997542571,
        -0.663533906046861,
        -0.665410126270643,
        -0.667271544821827,
        -0.669090206083235,
        -0.67109061204213,
        -0.671851814423714,
        -0.678696154177997,
        -0.660630258587478,
        -0.727386306394004,
        -0.5589306942905,
        -0.929470997237408,
        -5.95344681486583e-05,
        -0.979441373290573,
        -4.02679128054397e-05,
        -0.978888765621664,
        -3.33066341496843e-05,
        -0.996683719980696,
        -2.89264353459015e-05,
        -0.999999995639669,
        -2.85242641486147e-05,
        -0.999999998534366,
        -3.09882655729388e-05,
        -0.999999999210779,
        -3.67769449721367e-05,
        -0.999999999468728,
        -4.81790359260325e-05,
        -0.999999999587495,
        -6.64619744925331e-05,
        -0.999999999640893,
        -0.192410309091768,
        -0.999999999695108,
        -0.0001334575149404,
        -0.999999999710502,
        -5.5442637985935e-05,
        -0.999999999677263,
        -0.400110930542217,
        -0.999999999747586,
        -4.98848853542749e-05,
        -0.999999999777536,
        -5.14240880205469e-05,
        -0.999999999758722,
        -2.74805812410192e-05,
        -0.999999999637793,
        -1.71427110436707e-05,
        -0.983201022958269,
        -0.999999999232248,
        -1.44736171895711e-05,
        -0.99999999972965,
        -1.85029018066067e-05,
        -0.999999999783832,
        -1.93999667372342e-05,
        -0.999999999746339,
        -1.55457591783725e-05,
        -0.889558421283715,
        -0.999999999144342,
        -1.15274134894476e-05,
        -1.07321897022881e-05,
        -8.25197930007457e-06,
        -6.23400661536779e-06,
        -4.87396298858162e-06,
        -3.94868157718681e-06,
        -3.29159453264123e-06,
        -2.80470597599841e-06,
        -2.43064730547657e-06,
        -2.134584004157e-06,
        -1.89444940893869e-06,
        -1.69570008407223e-06,
        -1.52840893214942e-06,
        -1.38558967936835e-06,
        0.195805912695142,
        0.487166879209771,
        0.777170963760288,
        0.999999999907176,
        0.999999999855312,
    ],
    dtype=float,
)


# ============================
# Float dynamics (for initial guesses / regime updates)
# ============================


def step_full_float(vx: float, vy: float, s: float) -> tuple[float, float, float]:
    """One step. Returns (vx_next, vy_next, vy_drift_prebranch)."""
    s = float(np.clip(s, -1.0, 1.0))
    prev_vx = float(vx)
    c2 = 1.0 - s * s

    vy_drift = vy + 0.06 * c2 - 0.08  # pre-branch drift value
    vy_drift_prebranch = float(vy_drift)

    if vy_drift < 0.0:
        ty = 0.1 * vy_drift * c2  # negative
        vx = vx - ty
        vy_drift = vy_drift - ty

    if s > 0.0:
        tx = 0.04 * prev_vx * s
        vx = vx - tx
        vy_drift = vy_drift + 3.2 * tx

    vx = vx + 0.1 * (prev_vx - vx)
    return 0.99 * vx, 0.98 * vy_drift, vy_drift_prebranch


def simulate_cycle_float(
    S: np.ndarray, v0: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (VX[0..N], VY[0..N], vy_drift_prebranch[0..N-1])."""
    S = np.asarray(S, float)
    N = len(S)
    vx, vy = map(float, v0)
    VX = [vx]
    VY = [vy]
    VD: list[float] = []
    for k in range(N):
        vx, vy, vy_drift = step_full_float(vx, vy, float(S[k]))
        VX.append(vx)
        VY.append(vy)
        VD.append(vy_drift)
    return np.asarray(VX), np.asarray(VY), np.asarray(VD)


def fixed_point_for_cycle_float(
    S: np.ndarray, v_init: tuple[float, float] = (2.0, -0.6), n_iter: int = 160
) -> tuple[float, float]:
    """Numerical fixed point v0 ≈ F_S(v0) by iterating the cycle map."""
    vx, vy = map(float, v_init)
    for _ in range(n_iter):
        VX, VY, _ = simulate_cycle_float(S, (vx, vy))
        vx, vy = float(VX[-1]), float(VY[-1])
    return vx, vy


# ============================
# NLP builder (regime-fixed)
# ============================

ExtraConstraints = Callable[
    [ca.SX, ca.SX, ca.SX], Tuple[Sequence[ca.SX], Sequence[float], Sequence[float]]
]


@dataclass(frozen=True)
class SolverBundle:
    N: int
    solver: ca.Function
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray


def build_cycle_solver(
    N: int,
    *,
    anchor_max_s0: bool = True,
    extra_constraints: Optional[ExtraConstraints] = None,
    ipopt_options: Optional[dict] = None,
) -> SolverBundle:
    """Build the smooth NLP with regime parameters p_y, p_s supplied externally."""
    if ipopt_options is None:
        ipopt_options = {}

    VX = ca.SX.sym("VX", N + 1)
    VY = ca.SX.sym("VY", N + 1)
    S = ca.SX.sym("S", N)

    # Parameters: p_y (vy_drift branch), p_s (s>0 branch)
    p = ca.SX.sym("p", 2 * N)
    p_y = p[:N]
    p_s = p[N:]

    x = ca.vertcat(VX, VY, S)

    g: List[ca.SX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for k in range(N):
        vx = VX[k]
        vy = VY[k]
        s = S[k]
        prev_vx = vx
        c2 = 1 - s * s
        vy_drift = vy + 0.06 * c2 - 0.08

        # Regime consistency:
        # p_y=1 -> vy_drift <= 0 ; p_y=0 -> vy_drift >= 0
        # p_s=1 -> s >= 0        ; p_s=0 -> s <= 0
        g += [(2 * p_y[k] - 1) * vy_drift, -(2 * p_s[k] - 1) * s]
        lbg += [-ca.inf, -ca.inf]
        ubg += [0.0, 0.0]

        # Apply branches as smooth multipliers
        ty = 0.1 * vy_drift * c2 * p_y[k]
        vx1 = vx - ty
        vy1 = vy_drift - ty

        tx = 0.04 * prev_vx * s * p_s[k]
        vx2 = vx1 - tx
        vy2 = vy1 + 3.2 * tx

        vx3 = vx2 + 0.1 * (prev_vx - vx2)
        vx_next = 0.99 * vx3
        vy_next = 0.98 * vy2

        g += [VX[k + 1] - vx_next, VY[k + 1] - vy_next]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

    # Periodicity
    g += [VX[N] - VX[0], VY[N] - VY[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]

    # Anchor against cyclic shift: force step 0 to be a maximizer of S
    if anchor_max_s0:
        for k in range(1, N):
            g.append(S[k] - S[0])
            lbg.append(-ca.inf)
            ubg.append(0.0)

    # Extra constraints (user-supplied)
    if extra_constraints is not None:
        gg, ll, uu = extra_constraints(VX, VY, S)
        g += list(gg)
        lbg += list(ll)
        ubg += list(uu)

    g_all = ca.vertcat(*g) if g else ca.SX.zeros(0)

    # Objective: maximize mean(vy) (nodes 0..N-1)
    f = -(1.0 / N) * ca.sum1(VY[:N])

    nlp = {"x": x, "p": p, "f": f, "g": g_all}

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 12000,
        "ipopt.tol": 1e-12,
        "ipopt.acceptable_tol": 1e-11,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.compl_inf_tol": 1e-8,
        "ipopt.bound_relax_factor": 0.0,
        "ipopt.mu_strategy": "adaptive",
    }
    opts.update(ipopt_options)

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Variable bounds: VX >= 0, VY free, S in [-1,1]
    lbx = np.array([0.0] * (N + 1) + [-ca.inf] * (N + 1) + [-1.0] * N, float)
    ubx = np.array([ca.inf] * (N + 1) + [ca.inf] * (N + 1) + [1.0] * N, float)

    return SolverBundle(
        N=N,
        solver=solver,
        lbx=lbx,
        ubx=ubx,
        lbg=np.asarray(lbg, float),
        ubg=np.asarray(ubg, float),
    )


# ============================
# Solve (outer-loop regime fixing)
# ============================


@dataclass(frozen=True)
class CycleSolution:
    mean_vy: float
    max_viol: float
    status: str
    VX: np.ndarray
    VY: np.ndarray
    S: np.ndarray


def _unpack_x(x: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    VX = x[: N + 1]
    VY = x[N + 1 : 2 * (N + 1)]
    S = x[2 * (N + 1) :]
    return VX, VY, S


def _max_violation(g: np.ndarray, lbg: np.ndarray, ubg: np.ndarray) -> float:
    viol = np.maximum(g - ubg, 0.0) + np.maximum(lbg - g, 0.0)
    return float(np.max(viol)) if viol.size else 0.0


def solve_cycle_regime_fixed(
    bundle: SolverBundle,
    S_init: np.ndarray,
    *,
    outer_max: int = 18,
    accept_viol: float = 5e-8,
    v_init: tuple[float, float] = (2.0, -0.6),
    fixed_point_iters: int = 160,
) -> CycleSolution:
    """Solve starting from an initial control S_init using the regime-fixing outer loop."""
    N = bundle.N
    S_init = np.clip(np.asarray(S_init, float), -1.0, 1.0)

    # Initial guess for VX,VY from a cycle fixed point under the full dynamics
    v0 = fixed_point_for_cycle_float(S_init, v_init=v_init, n_iter=fixed_point_iters)
    VX0, VY0, vy_drift0 = simulate_cycle_float(S_init, v0)

    # initial regimes
    p_s = (S_init > 0).astype(float)
    p_y = (vy_drift0 < 0).astype(float)

    x0 = np.concatenate([VX0, VY0, S_init])

    best_x = x0
    best_status = ""
    best_viol = float("inf")

    for _ in range(max(1, outer_max)):
        p = np.concatenate([p_y, p_s])
        sol = bundle.solver(
            x0=x0, lbx=bundle.lbx, ubx=bundle.ubx, lbg=bundle.lbg, ubg=bundle.ubg, p=p
        )

        x = np.array(sol["x"]).squeeze()
        g = np.array(sol["g"]).squeeze()
        status = bundle.solver.stats().get("return_status", "")
        viol = _max_violation(g, bundle.lbg, bundle.ubg)

        if viol < best_viol:
            best_viol = viol
            best_x = x
            best_status = status

        VX, VY, S = _unpack_x(x, N)

        # Recompute regimes implied by current iterate
        c2 = 1.0 - S * S
        vy_drift = VY[:-1] + 0.06 * c2 - 0.08
        p_y_new = (vy_drift < 0).astype(float)
        p_s_new = (S > 0).astype(float)

        x0 = x
        if np.array_equal(p_y_new, p_y) and np.array_equal(p_s_new, p_s):
            break
        p_y, p_s = p_y_new, p_s_new

    VX, VY, S = _unpack_x(best_x, N)
    mean_vy = float(np.mean(VY[:-1]))

    return CycleSolution(
        mean_vy=mean_vy,
        max_viol=float(best_viol),
        status=best_status,
        VX=VX,
        VY=VY,
        S=S,
    )


# ============================
# Multi-start initializations (this is what makes it work)
# ============================


def resample_control(S: np.ndarray, N: int) -> np.ndarray:
    """Linear time resample of a control sequence."""
    S = np.asarray(S, float)
    if len(S) == N:
        return S.copy()
    t_old = np.linspace(0.0, len(S) - 1.0, len(S))
    t_new = np.linspace(0.0, len(S) - 1.0, N)
    return np.interp(t_new, t_old, S)


def rotate_to_max_at_zero(S: np.ndarray) -> np.ndarray:
    """Rotate cyclic sequence so that argmax(S) becomes index 0."""
    S = np.asarray(S, float).copy()
    idx = int(np.argmax(S))
    return np.roll(S, -idx)


def generate_initial_controls(
    N: int, rng: np.random.Generator, n: int, include_known_good: bool = True
) -> list[np.ndarray]:
    """Practical seeds: known-good warm start + a few deterministic + randoms."""
    inits: list[np.ndarray] = []

    if include_known_good:
        base = rotate_to_max_at_zero(resample_control(KNOWN_GOOD_S_255, N))
        inits.append(np.clip(base, -1.0, 1.0))
        # small perturbations around the base (helps escape tiny basins)
        for sigma in [0.01, 0.03, 0.07]:
            for _ in range(2):
                inits.append(np.clip(base + sigma * rng.standard_normal(N), -1.0, 1.0))

    # deterministic
    inits.append(np.zeros(N))
    inits.append(-0.5 * np.ones(N))
    inits.append(0.5 * np.ones(N))
    inits.append(-0.8 * np.ones(N))
    inits.append(0.8 * np.ones(N))
    inits.append(rng.choice([-1.0, 1.0], size=N))

    # smooth random walk
    z = rng.standard_normal(N)
    s_rw = np.clip(np.cumsum(z) / 10.0, -1.0, 1.0)
    inits.append(s_rw)

    while len(inits) < n:
        inits.append(rng.uniform(-1.0, 1.0, size=N))

    return inits[:n]


def is_ipopt_success(status: str) -> bool:
    return ("Solve_Succeeded" in status) or ("Solved_To_Acceptable_Level" in status)


def multi_start_solve(
    N: int,
    *,
    n_trials: int = 24,
    seed: int = 0,
    outer_max: int = 18,
    accept_viol: float = 5e-8,
    ipopt_options: Optional[dict] = None,
    extra_constraints: Optional[ExtraConstraints] = None,
    include_known_good_init: bool = True,
) -> tuple[CycleSolution, pd.DataFrame]:
    """Run multiple starts; return best solution and a trials summary."""
    bundle = build_cycle_solver(
        N,
        anchor_max_s0=True,
        extra_constraints=extra_constraints,
        ipopt_options=ipopt_options,
    )

    rng = np.random.default_rng(seed)
    inits = generate_initial_controls(
        N, rng, n_trials, include_known_good=include_known_good_init
    )

    rows = []
    best: Optional[CycleSolution] = None

    for i, S0 in enumerate(inits):
        sol = solve_cycle_regime_fixed(
            bundle, S0, outer_max=outer_max, accept_viol=accept_viol
        )
        accepted = bool(sol.max_viol <= accept_viol and is_ipopt_success(sol.status))
        rows.append(
            dict(
                trial=i,
                accepted=accepted,
                mean_vy=sol.mean_vy,
                max_viol=sol.max_viol,
                status=sol.status,
            )
        )
        if accepted and (best is None or sol.mean_vy > best.mean_vy):
            best = sol

    df = pd.DataFrame(rows)

    if best is None:
        # fall back: pick the smallest constraint violation trial
        best_idx = int(df["max_viol"].values.argmin())
        best = solve_cycle_regime_fixed(
            bundle, inits[best_idx], outer_max=outer_max, accept_viol=accept_viol
        )

    # polish from best S with tighter IPOPT settings
    bundle_polish = build_cycle_solver(
        N,
        anchor_max_s0=True,
        extra_constraints=extra_constraints,
        ipopt_options={
            **(ipopt_options or {}),
            "ipopt.max_iter": 22000,
            "ipopt.tol": 1e-12,
            "ipopt.acceptable_tol": 1e-11,
        },
    )
    best_polished = solve_cycle_regime_fixed(
        bundle_polish, best.S, outer_max=max(outer_max, 22), accept_viol=accept_viol
    )

    return best_polished, df


# ============================
# Convenience: dataframe + plots
# ============================


def solution_to_dataframe(sol: CycleSolution) -> pd.DataFrame:
    N = len(sol.S)
    s = sol.S
    ds = np.empty(N, float)
    ds[0] = s[0] - s[-1]
    ds[1:] = s[1:] - s[:-1]
    return pd.DataFrame(
        {"t": np.arange(N), "vx": sol.VX[:N], "vy": sol.VY[:N], "s": s, "ds": ds}
    )


def make_timeseries_plot(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Velocity (vx, vy)", "Control s", "Slope Δs"),
        vertical_spacing=0.08,
    )
    t = df["t"]
    fig.add_trace(go.Scatter(x=t, y=df["vx"], mode="lines", name="vx"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["vy"], mode="lines", name="vy"), row=1, col=1)
    fig.add_hline(y=0.0, line_width=1, line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=df["s"], mode="lines", name="s"), row=2, col=1)
    fig.update_yaxes(range=[-1.05, 1.05], row=2, col=1)
    fig.add_hline(y=0.0, line_width=1, line_color="gray", row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=df["ds"], mode="lines", name="Δs"), row=3, col=1)
    fig.add_hline(y=0.0, line_width=1, line_color="gray", row=3, col=1)

    fig.update_layout(title=title, height=950, hovermode="x unified", dragmode="pan")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def make_phase_plot(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["vx"], y=df["vy"], mode="lines", name="trajectory"))
    fig.add_trace(
        go.Scatter(
            x=[df["vx"].iloc[0]], y=[df["vy"].iloc[0]], mode="markers", name="start"
        )
    )
    fig.add_hline(y=0.0, line_width=1, line_color="gray")
    fig.add_vline(x=0.0, line_width=1, line_color="gray")
    fig.update_layout(
        title=title, xaxis_title="vx", yaxis_title="vy", height=600, dragmode="pan"
    )
    return fig


def write_html(fig: go.Figure, path: Path) -> None:
    config = {
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    }
    fig.write_html(str(path), include_plotlyjs="cdn", config=config)


# ============================
# Main (example usage)
# ============================


def main() -> None:
    # Choose N / trials
    N = 255
    n_trials = 20
    seed = 0

    ipopt_options = {
        # You can tune these if needed:
        # "ipopt.max_iter": 15000,
        # "ipopt.tol": 1e-12,
        # "ipopt.acceptable_tol": 1e-11,
    }

    # --- Optional extra constraints (EXAMPLE ONLY) ---
    # Uncomment and customize, then pass extra_constraints=my_constraints.
    #
    # def my_constraints(VX: ca.SX, VY: ca.SX, S: ca.SX):
    #     g, lb, ub = [], [], []
    #     # Example: enforce VY >= 0 on all nodes 0..N-1
    #     # for k in range(VY.shape[0]-1):
    #     #     g.append(VY[k]); lb.append(0.0); ub.append(ca.inf)
    #     # Example: enforce S non-increasing on interval [a..b]
    #     # a, b = 50, 80
    #     # for k in range(a, b):
    #     #     g.append(S[k+1] - S[k]); lb.append(-ca.inf); ub.append(0.0)
    #     return g, lb, ub
    #
    # extra_constraints = my_constraints
    extra_constraints = None

    best, trials = multi_start_solve(
        N,
        n_trials=n_trials,
        seed=seed,
        outer_max=18,
        accept_viol=5e-8,
        ipopt_options=ipopt_options,
        extra_constraints=extra_constraints,
        include_known_good_init=True,
    )

    # Diagnostics: anchor check for cyclic shift invariance
    s0 = float(best.S[0])
    smax = float(np.max(best.S))
    argmax = int(np.argmax(best.S))
    print(f"Best mean(vy) = {best.mean_vy:.12f}")
    print(f"Max constraint violation = {best.max_viol:.3e}")
    print(f"IPOPT status = {best.status}")
    print(
        f"Anchor check: s[0]={s0:.6f}, max(s)={smax:.6f}, argmax={argmax} (should be 0)"
    )

    out_dir = Path("cycle_opt_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = solution_to_dataframe(best)
    df.to_csv(out_dir / f"solution_N{N}.csv", index=False)
    trials.to_csv(out_dir / f"trials_N{N}.csv", index=False)

    write_html(
        make_timeseries_plot(
            df, f"N={N} best cycle timeseries (mean vy={best.mean_vy:.6f})"
        ),
        out_dir / f"timeseries_N{N}.html",
    )
    write_html(
        make_phase_plot(df, f"N={N} phase portrait (vx vs vy)"),
        out_dir / f"phase_N{N}.html",
    )

    print("Wrote outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
