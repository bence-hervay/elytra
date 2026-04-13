#!/usr/bin/env python3
"""
Cycle-length sweep for the (vx, vy, s) system.

- coarse sweep: N=100..1000 step 10
- fine sweep: around coarse best ±50 step 1
- multi-start (zeros/random/warm) + regime-update outer loop
- polishing pass with higher IPOPT CPU limit to get tight feasibility

Dependencies:
  pip install casadi==3.6.7 numpy pandas plotly matplotlib
"""

import time, math, colorsys
from pathlib import Path
import numpy as np
import pandas as pd
import casadi as ca


# -------------------------
# Dynamics (float)
# -------------------------
def next_v_float(v, s):
    vx, vy = float(v[0]), float(v[1])
    c2 = 1.0 - s * s
    prev_vx = vx
    vy = vy + 0.06 * c2 - 0.08
    if vy < 0.0:
        ty = 0.1 * vy * c2
        vx -= ty
        vy -= ty
    if s > 0.0:
        tx = 0.04 * prev_vx * s
        vx -= tx
        vy += 3.2 * tx
    vx = vx + 0.1 * (prev_vx - vx)
    return np.array([0.99 * vx, 0.98 * vy], dtype=float)


def simulate_float(v0, S):
    V = np.zeros((2, len(S) + 1), dtype=float)
    V[:, 0] = v0
    v = np.array(v0, dtype=float)
    for k, s in enumerate(S):
        v = next_v_float(v, float(s))
        V[:, k + 1] = v
    return V


def fixed_point_for_control(S, v_init=np.array([2.0, 0.0]), n_iter=12):
    v = np.array(v_init, dtype=float)
    for _ in range(n_iter):
        v = simulate_float(v, S)[:, -1]
    return v


def rotate_to_max_at_zero(S):
    idx = int(np.argmax(S))
    if idx == 0:
        return S.copy()
    N = len(S)
    return np.array([S[(idx + k) % N] for k in range(N)], dtype=float)


def resample_control(S_old, N_new):
    N_old = len(S_old)
    t_old = np.arange(N_old) / N_old
    t_new = np.arange(N_new) / N_new
    t_ext = np.concatenate([t_old, t_old + 1.0])
    S_ext = np.concatenate([S_old, S_old])
    S_new = np.interp(t_new, t_ext, S_ext)
    S_new = np.clip(S_new, -0.999, 0.999)
    S_new = rotate_to_max_at_zero(S_new)
    return S_new


# -------------------------
# Frozen-regime multiple-shooting NLP builder
# -------------------------
def build_param_frozen_solver(
    N,
    vx_min=1e-6,
    s_eps=1e-9,
    add_symmetry_max_s0=True,
    tol=1e-8,
    max_iter=3000,
    max_cpu_time=0.3,
    print_level=0,
):
    X = ca.SX.sym("X", 2, N + 1)
    S = ca.SX.sym("S", N)
    w = ca.vertcat(ca.reshape(X, -1, 1), S)
    nX = 2 * (N + 1)

    p = ca.SX.sym("p", 2 * N)
    p_y = p[:N]  # active when vy1<0
    p_s = p[N:]  # active when s>0

    g_eq = []
    g_ineq = []

    for k in range(N):
        vx = X[0, k]
        vy = X[1, k]
        s = S[k]
        c2 = 1 - s * s
        prev_vx = vx

        vy1 = vy + 0.06 * c2 - 0.08

        ty = 0.1 * vy1 * c2
        vx1 = vx - p_y[k] * ty
        vy1_corr = vy1 - p_y[k] * ty

        tx = 0.04 * prev_vx * s
        vx2 = vx1 - p_s[k] * tx
        vy2 = vy1_corr + p_s[k] * 3.2 * tx

        vx3 = vx2 + 0.1 * (prev_vx - vx2)
        vx_next = 0.99 * vx3
        vy_next = 0.98 * vy2

        g_eq += [X[0, k + 1] - vx_next, X[1, k + 1] - vy_next]

        # sign-consistency for regimes
        g_ineq += [(2 * p_y[k] - 1) * vy1]  # <=0  -> vy1<0 if p_y=1, vy1>=0 if p_y=0
        g_ineq += [-(2 * p_s[k] - 1) * s]  # <=0  -> s>=0 if p_s=1, s<=0 if p_s=0

    g_eq += [X[0, N] - X[0, 0], X[1, N] - X[1, 0]]

    if add_symmetry_max_s0:
        for k in range(1, N):
            g_ineq += [S[k] - S[0]]  # s0 is a (global) maximum

    g = ca.vertcat(*(g_eq + g_ineq))
    f = -(1.0 / N) * ca.sum2(X[1, 0:N])

    lbw = -np.inf * np.ones(nX + N)
    ubw = np.inf * np.ones(nX + N)
    for k in range(N + 1):
        lbw[2 * k] = vx_min
    lbw[nX:] = -1.0 + s_eps
    ubw[nX:] = 1.0 - s_eps

    m = int(g.shape[0])
    lbg = -np.inf * np.ones(m)
    ubg = np.zeros(m)
    eq_count = len(g_eq)
    lbg[:eq_count] = 0.0
    ubg[:eq_count] = 0.0

    nlp = {"x": w, "p": p, "f": f, "g": g}
    opts = {
        "ipopt": {
            "print_level": int(print_level),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "constr_viol_tol": float(tol),
            "compl_inf_tol": float(tol),
            "acceptable_tol": max(1e-6, float(tol) * 100),
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
            "max_cpu_time": float(max_cpu_time),
        },
        "print_time": False,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return dict(solver=solver, lbw=lbw, ubw=ubw, lbg=lbg, ubg=ubg, N=N)


def unpack_w_ms(w_opt, N):
    nX = 2 * (N + 1)
    V = np.array(w_opt[:nX]).reshape((N + 1, 2)).T
    S = np.array(w_opt[nX:])
    return V, S


def derive_regimes_from_traj(Vsim, S, eps_s=1e-9):
    vy1 = Vsim[1, :-1] + 0.06 * (1 - S * S) - 0.08
    p_y = (vy1 < 0).astype(float)
    p_s = (S > eps_s).astype(float)
    return p_y, p_s, vy1


def solve_trial(prob, S0, fp_iter=10, v_init=np.array([2.0, 0.0])):
    N = prob["N"]
    v0 = fixed_point_for_control(S0, v_init=v_init, n_iter=fp_iter)
    V0 = simulate_float(v0, S0)
    p_y0, p_s0, _ = derive_regimes_from_traj(V0, S0)
    p0 = np.concatenate([p_y0, p_s0])
    w0 = np.concatenate([V0.T.reshape(-1), S0])
    sol = prob["solver"](
        x0=w0, lbx=prob["lbw"], ubx=prob["ubw"], lbg=prob["lbg"], ubg=prob["ubg"], p=p0
    )
    w_opt = np.array(sol["x"]).reshape(-1)
    Vopt, Sopt = unpack_w_ms(w_opt, N)
    Vsim = simulate_float(Vopt[:, 0], Sopt)
    mean_vy = float(Vsim[1, :-1].mean())
    per_err = float(np.linalg.norm(Vsim[:, -1] - Vsim[:, 0]))
    dyn_err = float(np.max(np.abs(Vsim - Vopt)))
    stats = prob["solver"].stats()
    return dict(
        mean_vy=mean_vy,
        per_err=per_err,
        dyn_err=dyn_err,
        status=stats["return_status"],
        iters=stats.get("iter_count"),
        V=Vopt,
        S=Sopt,
        Vsim=Vsim,
    )


def refine_from_guess(prob, V_guess, S_guess):
    N = prob["N"]
    vy1 = V_guess[1, :-1] + 0.06 * (1 - S_guess * S_guess) - 0.08
    p_y = (vy1 < 0).astype(float)
    p_s = (S_guess > 1e-9).astype(float)
    p0 = np.concatenate([p_y, p_s])
    w0 = np.concatenate([V_guess.T.reshape(-1), S_guess])
    sol = prob["solver"](
        x0=w0, lbx=prob["lbw"], ubx=prob["ubw"], lbg=prob["lbg"], ubg=prob["ubg"], p=p0
    )
    w_opt = np.array(sol["x"]).reshape(-1)
    Vopt, Sopt = unpack_w_ms(w_opt, N)
    Vsim = simulate_float(Vopt[:, 0], Sopt)
    mean_vy = float(Vsim[1, :-1].mean())
    per_err = float(np.linalg.norm(Vsim[:, -1] - Vsim[:, 0]))
    dyn_err = float(np.max(np.abs(Vsim - Vopt)))
    return dict(
        mean_vy=mean_vy, per_err=per_err, dyn_err=dyn_err, V=Vopt, S=Sopt, Vsim=Vsim
    )


def regime_update_solve(prob, V_guess, S_guess, max_outer=4, eps_s=1e-6, eps_y=1e-7):
    V = V_guess.copy()
    S = S_guess.copy()
    last_p = None
    for _ in range(max_outer):
        vy1 = V[1, :-1] + 0.06 * (1 - S * S) - 0.08
        p_s = (S > eps_s).astype(float)
        p_y = (vy1 < -eps_y).astype(float)
        p = np.concatenate([p_y, p_s])
        if last_p is not None and np.array_equal(p, last_p):
            break
        last_p = p.copy()
        w0 = np.concatenate([V.T.reshape(-1), S])
        sol = prob["solver"](
            x0=w0,
            lbx=prob["lbw"],
            ubx=prob["ubw"],
            lbg=prob["lbg"],
            ubg=prob["ubg"],
            p=p,
        )
        w_opt = np.array(sol["x"]).reshape(-1)
        V, S = unpack_w_ms(w_opt, prob["N"])
    Vsim = simulate_float(V[:, 0], S)
    mean_vy = float(Vsim[1, :-1].mean())
    per_err = float(np.linalg.norm(Vsim[:, -1] - Vsim[:, 0]))
    dyn_err = float(np.max(np.abs(Vsim - V)))
    return dict(
        mean_vy=mean_vy,
        per_err=per_err,
        dyn_err=dyn_err,
        status=prob["solver"].stats()["return_status"],
        V=V,
        S=S,
        Vsim=Vsim,
    )


# -------------------------
# Initial guesses
# -------------------------
def heuristic_control(N, rotate=True):
    Lp = max(1, int(round(0.34 * N)))
    Lz1 = int(round(0.03 * N))
    Ln = max(1, int(round(0.55 * N)))
    Lz2 = N - (Lp + Lz1 + Ln)
    if Lz2 < 0:
        Ln = max(1, Ln + Lz2)
        Lz2 = N - (Lp + Lz1 + Ln)
    s = np.zeros(N, dtype=float)
    s[:Lp] = 0.9
    s[Lp : Lp + Lz1] = 0.0
    s[Lp + Lz1 : Lp + Lz1 + Ln] = -0.8
    s[Lp + Lz1 + Ln :] = 0.0
    rng = np.random.default_rng(0)
    s += rng.normal(0, 0.01, size=N)
    s = np.clip(s, -0.999, 0.999)
    if rotate:
        s = rotate_to_max_at_zero(s)
    return s


def initial_guess_control(N, rng):
    mode = rng.choice(
        ["piecewise4", "sin", "smooth_rw", "bangbang", "uniform"],
        p=[0.45, 0.15, 0.22, 0.10, 0.08],
    )
    if mode == "uniform":
        s = rng.uniform(-1, 1, N)
    elif mode == "sin":
        A = float(rng.uniform(0.3, 1.0))
        phi = float(rng.uniform(0, 2 * np.pi))
        offset = float(rng.uniform(-0.3, 0.3))
        k = np.arange(N)
        s = A * np.sin(2 * np.pi * k / N + phi) + offset
        s = np.clip(s, -0.999, 0.999)
        s = rotate_to_max_at_zero(s)
    elif mode == "smooth_rw":
        s = np.zeros(N)
        s[0] = float(rng.uniform(-0.5, 0.5))
        drift = float(rng.uniform(-0.02, 0.02))
        for k in range(1, N):
            s[k] = s[k - 1] + drift + float(rng.normal(0, 0.15))
        s = np.tanh(s)
        s = rotate_to_max_at_zero(s)
    elif mode == "bangbang":
        p_pos = float(rng.uniform(0.2, 0.6))
        p_zero = float(rng.uniform(0.0, 0.3))
        p_neg = max(0.0, 1.0 - p_pos - p_zero)
        vals = rng.choice([1.0, 0.0, -1.0], size=N, p=[p_pos, p_zero, p_neg])
        s = vals.astype(float)
        s += rng.normal(0, 0.05, size=N)
        s = np.clip(s, -0.999, 0.999)
        s = rotate_to_max_at_zero(s)
    else:
        # piecewise4
        Lz1 = int(rng.integers(0, max(4, N // 8) + 1))
        Lz2 = int(rng.integers(0, max(4, N // 8) + 1))
        remaining = N - (Lz1 + Lz2)
        if remaining < 2:
            Lz1 = Lz2 = 0
            remaining = N
        Lp = int(rng.integers(1, remaining))
        Ln = remaining - Lp
        a = float(rng.uniform(0.2, 1.0))
        b = float(rng.uniform(0.2, 1.0))
        s = np.zeros(N, dtype=float)
        s[:Lp] = a
        s[Lp : Lp + Lz1] = 0.0
        s[Lp + Lz1 : Lp + Lz1 + Ln] = -b
        s[Lp + Lz1 + Ln :] = 0.0
        s += rng.normal(0, 0.02, size=N)
        s = np.clip(s, -0.999, 0.999)
        s = rotate_to_max_at_zero(s)
    return s, mode


# -------------------------
# Helpers for warm starts
# -------------------------
def tile_control(S_base, N):
    L = len(S_base)
    assert N % L == 0
    S = np.tile(S_base, N // L)
    S = rotate_to_max_at_zero(np.clip(S, -0.999, 0.999))
    return S


def best_divisor_tile_warm(N, known_solutions):
    best = None
    best_d = None
    for d, sol in known_solutions.items():
        if d > 0 and (N % d == 0):
            if best is None or sol["mean_vy"] > best:
                best = sol["mean_vy"]
                best_d = d
    if best_d is None:
        return None, None
    return tile_control(known_solutions[best_d]["S"], N), best_d


def best_global_resample_warm(N, known_solutions):
    if not known_solutions:
        return None, None
    best_d = max(known_solutions.keys(), key=lambda d: known_solutions[d]["mean_vy"])
    S_best = known_solutions[best_d]["S"]
    if len(S_best) == N:
        return S_best.copy(), best_d
    return resample_control(S_best, N), best_d


# -------------------------
# Main sweep (you can adapt)
# -------------------------
def solve_one_length_rigorous(
    N: int,
    rng: np.random.Generator,
    known_solutions: dict,
    prev_best_S=None,
    n_random: int = 2,
    screen_cpu=0.10,
    refine_cpu=0.50,
):
    screen = build_param_frozen_solver(
        N, tol=1e-8, max_iter=4000, max_cpu_time=screen_cpu
    )
    refine = build_param_frozen_solver(
        N, tol=1e-12, max_iter=20000, max_cpu_time=refine_cpu
    )

    trials = []
    trials.append(("zeros", solve_trial(screen, np.zeros(N), fp_iter=8)))
    trials.append(
        (
            "heuristic",
            solve_trial(screen, heuristic_control(N, rotate=True), fp_iter=10),
        )
    )
    for _ in range(n_random):
        Srand, mode = initial_guess_control(N, rng)
        trials.append((mode, solve_trial(screen, Srand, fp_iter=10)))
    if prev_best_S is not None:
        Sw_prev = (
            resample_control(prev_best_S, N)
            if len(prev_best_S) != N
            else prev_best_S.copy()
        )
        trials.append(("warm_prev", solve_trial(screen, Sw_prev, fp_iter=10)))
    Sw_tile, d_tile = best_divisor_tile_warm(N, known_solutions)
    if Sw_tile is not None:
        trials.append((f"warm_tile_{d_tile}", solve_trial(screen, Sw_tile, fp_iter=10)))
    Sw_res, d_res = best_global_resample_warm(N, known_solutions)
    if Sw_res is not None:
        trials.append(
            (f"warm_resample_{d_res}", solve_trial(screen, Sw_res, fp_iter=10))
        )

    trial_means = np.array([r["mean_vy"] for _, r in trials])
    best_mode, best_screen = max(trials, key=lambda x: x[1]["mean_vy"])

    ref = refine_from_guess(refine, best_screen["V"], best_screen["S"])
    upd = regime_update_solve(
        refine, ref["V"], ref["S"], max_outer=6, eps_s=1e-7, eps_y=1e-9
    )
    return upd, dict(
        N=N,
        best_mean_vy=float(upd["mean_vy"]),
        per_err=float(upd["per_err"]),
        dyn_err=float(upd["dyn_err"]),
        status=str(upd.get("status", "")),
        best_mode=best_mode,
        mean_trials=float(trial_means.mean()),
        std_trials=float(trial_means.std()),
        n_trials=len(trials),
    )


if __name__ == "__main__":
    out = Path("cycle_length_sweep_100_1000_outputs")
    out.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(20260131)

    # Coarse sweep
    best = {}
    prev = None
    rows = []
    for N in range(200, 300, 10):
        upd, summ = solve_one_length_rigorous(
            N,
            rng,
            known_solutions=best,
            prev_best_S=prev,
            n_random=2 if N < 400 else 3,
            screen_cpu=0.1,
            refine_cpu=0.2,
        )
        best[N] = dict(S=upd["S"], Vsim=upd["Vsim"], mean_vy=float(upd["mean_vy"]))
        prev = upd["S"]
        rows.append(summ)
        print("coarse", N, summ["best_mean_vy"], summ["status"])
    pd.DataFrame(rows).to_csv(out / "coarse_summary.csv", index=False)

    # You can add: fine sweep around argmax(rows), polishing, and plot generation
    print("Done.")
