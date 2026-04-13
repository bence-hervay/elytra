#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import argparse
import logging
import numpy as np
import casadi as ca


Array = np.ndarray


@dataclass(frozen=True)
class Config:
    N: int
    trials: int
    seed: int
    fp_iters: int
    v_init: Tuple[float, float]
    ipopt_max_iter: int
    ipopt_tol: float
    ipopt_cpu: float
    verbose: bool


@dataclass(frozen=True)
class Trial:
    S: Array
    V: Array
    mean_vy: float


@dataclass(frozen=True)
class SolverBundle:
    N: int
    solver: ca.Function
    lbx: Array
    ubx: Array
    lbg: Array
    ubg: Array


@dataclass(frozen=True)
class Solution:
    mean_vy: float
    status: str
    V: Array
    S: Array


def step_float(v: Array, s: float) -> Array:
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


def simulate_float(v0: Array, S: Array) -> Array:
    V = np.zeros((2, len(S) + 1), dtype=float)
    V[:, 0] = v0
    v = np.array(v0, dtype=float)
    for k, s in enumerate(S):
        v = step_float(v, float(s))
        V[:, k + 1] = v
    return V


def fixed_point_for_control(
    S: Array, v_init: Tuple[float, float], n_iter: int
) -> Array:
    v = np.array(v_init, dtype=float)
    for _ in range(n_iter):
        v = simulate_float(v, S)[:, -1]
    return v


def rotate_to_max_at_zero(S: Array) -> Array:
    idx = int(np.argmax(S))
    if idx == 0:
        return S.copy()
    N = len(S)
    return np.array([S[(idx + k) % N] for k in range(N)], dtype=float)


def heuristic_control(N: int, rotate: bool = True) -> Array:
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


def initial_guess_control(N: int, rng: np.random.Generator) -> tuple[Array, str]:
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
    return s, str(mode)


def trial_from_control(S: Array, v_init: Tuple[float, float], fp_iters: int) -> Trial:
    v0 = np.array(v_init, dtype=float)
    if fp_iters > 0:
        v0 = fixed_point_for_control(S, v_init=v_init, n_iter=fp_iters)
    V = simulate_float(v0, S)
    mean_vy = float(V[1, :-1].mean())
    return Trial(S=S, V=V, mean_vy=mean_vy)


def build_frozen_solver(
    N: int, tol: float, max_iter: int, max_cpu: float
) -> SolverBundle:
    X = ca.SX.sym("X", 2, N + 1)
    S = ca.SX.sym("S", N)
    w = ca.vertcat(ca.reshape(X, -1, 1), S)
    nX = 2 * (N + 1)

    p = ca.SX.sym("p", 2 * N)
    p_y = p[:N]
    p_s = p[N:]

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
        g_ineq += [(2 * p_y[k] - 1) * vy1]
        g_ineq += [-(2 * p_s[k] - 1) * s]

    g_eq += [X[0, N] - X[0, 0], X[1, N] - X[1, 0]]

    for k in range(1, N):
        g_ineq += [S[k] - S[0]]

    g = ca.vertcat(*(g_eq + g_ineq))
    f = -(1.0 / N) * ca.sum2(X[1, 0:N])

    lbx = -np.inf * np.ones(nX + N)
    ubx = np.inf * np.ones(nX + N)
    for k in range(N + 1):
        lbx[2 * k] = 1e-6
    lbx[nX:] = -1.0 + 1e-9
    ubx[nX:] = 1.0 - 1e-9

    m = int(g.shape[0])
    lbg = -np.inf * np.ones(m)
    ubg = np.zeros(m)
    eq_count = len(g_eq)
    lbg[:eq_count] = 0.0
    ubg[:eq_count] = 0.0

    nlp = {"x": w, "p": p, "f": f, "g": g}
    opts = {
        "ipopt": {
            "print_level": 0,
            "max_iter": int(max_iter),
            "tol": float(tol),
            "constr_viol_tol": float(tol),
            "compl_inf_tol": float(tol),
            "acceptable_tol": max(1e-6, float(tol) * 100),
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
            "max_cpu_time": float(max_cpu),
        },
        "print_time": False,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return SolverBundle(N=N, solver=solver, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)


def unpack_w(w_opt: Array, N: int) -> tuple[Array, Array]:
    nX = 2 * (N + 1)
    V = np.array(w_opt[:nX]).reshape((N + 1, 2)).T
    S = np.array(w_opt[nX:])
    return V, S


def derive_regimes(V: Array, S: Array) -> Array:
    vy1 = V[1, :-1] + 0.06 * (1 - S * S) - 0.08
    p_y = (vy1 < 0).astype(float)
    p_s = (S > 1e-9).astype(float)
    return np.concatenate([p_y, p_s])


def optimize_from_best(cfg: Config) -> Solution:
    rng = np.random.default_rng(cfg.seed)
    best: Trial | None = None
    best_mode = ""

    if cfg.verbose:
        logging.info("trials=%d fp_iters=%d", cfg.trials, cfg.fp_iters)

    trials: list[tuple[str, Array]] = []
    if cfg.trials >= 1:
        trials.append(("zeros", np.zeros(cfg.N, dtype=float)))
    if cfg.trials >= 2:
        trials.append(("heuristic", heuristic_control(cfg.N, rotate=True)))
    for _ in range(max(0, cfg.trials - len(trials))):
        S, mode = initial_guess_control(cfg.N, rng)
        trials.append((mode, S))

    for mode, S in trials:
        trial = trial_from_control(S, cfg.v_init, cfg.fp_iters)
        if best is None or trial.mean_vy > best.mean_vy:
            best = trial
            best_mode = mode
            if cfg.verbose:
                logging.info(
                    "new best mean_vy=%.12f mode=%s", float(trial.mean_vy), best_mode
                )

    assert best is not None

    bundle = build_frozen_solver(
        cfg.N, tol=cfg.ipopt_tol, max_iter=cfg.ipopt_max_iter, max_cpu=cfg.ipopt_cpu
    )
    p0 = derive_regimes(best.V, best.S)
    w0 = np.concatenate([best.V.T.reshape(-1), best.S])
    sol = bundle.solver(
        x0=w0, lbx=bundle.lbx, ubx=bundle.ubx, lbg=bundle.lbg, ubg=bundle.ubg, p=p0
    )
    w_opt = np.array(sol["x"]).reshape(-1)
    Vopt, Sopt = unpack_w(w_opt, cfg.N)
    Vsim = simulate_float(Vopt[:, 0], Sopt)
    mean_vy = float(Vsim[1, :-1].mean())
    stats = bundle.solver.stats()
    status = str(stats.get("return_status", ""))
    if cfg.verbose:
        logging.info(
            "ipopt status=%s iter_count=%s",
            status,
            str(stats.get("iter_count", "n/a")),
        )
    return Solution(mean_vy=mean_vy, status=status, V=Vopt, S=Sopt)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=255, help="Cycle length (number of steps).")
    p.add_argument("--trials", type=int, default=100, help="Number of random trials.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for trials.")
    p.add_argument(
        "--fp-iters",
        type=int,
        default=100,
        help="Fixed-point iterations from vx0 to refine v0 (0 disables).",
    )
    p.add_argument(
        "--ipopt-max-iter",
        type=int,
        default=12000,
        help="IPOPT maximum iterations for the final solve.",
    )
    p.add_argument(
        "--ipopt-tol",
        type=float,
        default=1e-10,
        help="IPOPT convergence tolerance.",
    )
    p.add_argument(
        "--ipopt-cpu",
        type=float,
        default=0.5,
        help="IPOPT max CPU time (seconds).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print iteration counts and trial progress.",
    )
    args = p.parse_args()
    return Config(
        N=args.N,
        trials=args.trials,
        seed=args.seed,
        fp_iters=args.fp_iters,
        v_init=(0.5, 0.0),
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_tol=args.ipopt_tol,
        ipopt_cpu=args.ipopt_cpu,
        verbose=bool(args.verbose),
    )


def main() -> None:
    cfg = parse_args()
    if cfg.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    sol = optimize_from_best(cfg)
    print(f"N={cfg.N} best_mean_vy={sol.mean_vy:.12f}")


if __name__ == "__main__":
    main()
