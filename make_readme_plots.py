#!/usr/bin/env python3
"""Generate the PNG figures embedded in README.md.

Reads the stored optimal cycle (cycle_opt_outputs/solution_N255.csv) and the
cycle-length sweep summary, re-simulates the control with the exact float
dynamics, and writes the figures to docs/img/.

Usage: uv run python make_readme_plots.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
IMG = REPO / "docs" / "img"

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

C_VY = "#2a78d6"  # blue
C_VX = "#1baf7a"  # aqua
C_ANGLE = "#4a3aa7"  # violet
C_PATH = "#eb6834"  # orange


def step(vx: float, vy: float, s: float) -> tuple[float, float]:
    """One tick of the elytra dynamics (see elytra_min.py), s = sin(-pitch)."""
    c2 = 1.0 - s * s
    vx0 = vx
    vy += 0.06 * c2 - 0.08
    if vy < 0.0:
        ty = 0.1 * vy * c2
        vx -= ty
        vy -= ty
    if s > 0.0:
        tx = 0.04 * vx0 * s
        vx -= tx
        vy += 3.2 * tx
    vx += 0.1 * (vx0 - vx)
    return 0.99 * vx, 0.98 * vy


def simulate(vx0: float, vy0: float, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = vx0, vy0
    VX, VY = np.empty(len(S)), np.empty(len(S))
    for k, s in enumerate(S):
        vx, vy = step(vx, vy, float(s))
        VX[k], VY[k] = vx, vy
    return VX, VY


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": SECONDARY,
            "font.family": "sans-serif",
        }
    )


def style_ax(ax: mpl.axes.Axes) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def fig_best_cycle(df: pd.DataFrame, alt: np.ndarray) -> None:
    t = df["t"].to_numpy()
    theta = np.degrees(np.arcsin(df["s"].to_numpy()))

    fig, (ax_a, ax_v, ax_h) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"hspace": 0.35}
    )

    ax_a.plot(t, theta, color=C_ANGLE, lw=1.6)
    ax_a.axhline(0, color=AXIS, lw=0.8)
    ax_a.set_title("Control: nose-up pitch angle θ(t)   (s = sin θ)")
    ax_a.set_ylabel("θ  (degrees)")
    ax_a.text(38, 38, "climb — bleed speed into height", color=SECONDARY, fontsize=9)
    ax_a.text(125, -52, "dive — regain speed", color=SECONDARY, fontsize=9)
    ax_a.text(214, 18, "chatter", color=SECONDARY, fontsize=9, ha="center")
    style_ax(ax_a)

    ax_v.plot(t, df["vx"], color=C_VX, lw=1.6, label="vx (horizontal)")
    ax_v.plot(t, df["vy"], color=C_VY, lw=1.6, label="vy (vertical)")
    ax_v.axhline(0, color=AXIS, lw=0.8)
    mean_vy = float(df["vy"].mean())
    ax_v.axhline(mean_vy, color=C_VY, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax_v.text(
        t[-1], mean_vy + 0.07, f"mean vy = +{mean_vy:.4f}", color=C_VY, fontsize=9, ha="right"
    )
    ax_v.set_title("Velocities")
    ax_v.set_ylabel("blocks / tick")
    ax_v.legend(loc="upper left", ncols=2)
    ax_v.text(t[-1] + 2, df["vx"].iloc[-1], "vx", color=C_VX, fontsize=9, va="center")
    ax_v.text(t[-1] + 2, df["vy"].iloc[-1], "vy", color=C_VY, fontsize=9, va="center")
    style_ax(ax_v)

    ax_h.plot(t, alt, color=C_PATH, lw=1.6)
    ax_h.axhline(0, color=AXIS, lw=0.8)
    ax_h.set_title("Altitude gained over one cycle")
    ax_h.set_ylabel("blocks")
    ax_h.set_xlabel("t  (game ticks, 20 per second)")
    ax_h.text(
        t[-1], alt[-1] - 3, f"+{alt[-1]:.1f} blocks", color=C_PATH, fontsize=9, ha="right", va="top"
    )
    style_ax(ax_h)

    fig.savefig(IMG / "best_cycle.png", bbox_inches="tight")
    plt.close(fig)


def fig_flight_path(vx0: float, vy0: float, S: np.ndarray, cycles: int = 3) -> None:
    VX, VY = simulate(vx0, vy0, np.tile(S, cycles))
    x = np.cumsum(VX)
    y = np.cumsum(VY)
    N = len(S)
    gain = y[N - 1] - 0.0

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, y, color=C_PATH, lw=1.6)
    ends = [k * N - 1 for k in range(1, cycles + 1)]
    ax.scatter(x[ends], y[ends], s=18, color=INK, zorder=5)
    for k, e in enumerate(ends):
        ax.annotate(
            f"cycle {k + 1}:  {y[e]:+.1f} blocks",
            (x[e], y[e]),
            textcoords="offset points",
            xytext=(6, -12),
            color=SECONDARY,
            fontsize=9,
        )
    ax.set_title(f"Flight path — the cycle climbs {gain:.1f} blocks every {N} ticks, forever")
    ax.set_xlabel("horizontal distance  (blocks)")
    ax.set_ylabel("altitude  (blocks)")
    style_ax(ax)
    fig.savefig(IMG / "flight_path.png", bbox_inches="tight")
    plt.close(fig)


def fig_phase_portrait(df: pd.DataFrame) -> None:
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.plot(vx, vy, color=C_VY, lw=1.6)
    ax.axhline(0, color=AXIS, lw=0.8)
    for k in range(0, len(vx), 32):
        ax.annotate(
            "",
            xy=(vx[(k + 2) % len(vx)], vy[(k + 2) % len(vx)]),
            xytext=(vx[k], vy[k]),
            arrowprops=dict(arrowstyle="-|>", color=C_VY, lw=1.2),
        )
    ax.scatter([vx[0]], [vy[0]], s=26, color=INK, zorder=5)
    ax.annotate(
        "t = 0", (vx[0], vy[0]), textcoords="offset points", xytext=(8, 4),
        color=SECONDARY, fontsize=9,
    )
    ax.set_title("Phase portrait of the optimal cycle")
    ax.set_xlabel("vx  (blocks / tick)")
    ax.set_ylabel("vy  (blocks / tick)")
    ax.set_aspect("equal")
    style_ax(ax)
    fig.savefig(IMG / "phase_portrait.png", bbox_inches="tight")
    plt.close(fig)


def steady_state_vy(s: float, iters: int = 4000) -> float:
    vx, vy = 1.0, 0.0
    for _ in range(iters):
        vx, vy = step(vx, vy, s)
    return vy


def fig_steady_state() -> tuple[float, float]:
    s_grid = np.linspace(-0.999, 0.999, 401)
    vy_ss = np.array([steady_state_vy(float(s)) for s in s_grid])
    theta = np.degrees(np.arcsin(s_grid))
    best = int(np.argmax(vy_ss))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(theta, vy_ss, color=C_VY, lw=1.6)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.scatter([theta[best]], [vy_ss[best]], s=22, color=INK, zorder=5)
    ax.annotate(
        f"best constant pitch: θ = {theta[best]:.0f}°, vy = {vy_ss[best]:.4f}",
        (theta[best], vy_ss[best]),
        textcoords="offset points",
        xytext=(10, 10),
        color=SECONDARY,
        fontsize=9,
    )
    ax.set_title("Steady-state sink rate of every constant pitch — no fixed angle climbs")
    ax.set_xlabel("nose-up pitch angle θ  (degrees)")
    ax.set_ylabel("steady-state vy  (blocks / tick)")
    style_ax(ax)
    fig.savefig(IMG / "steady_state.png", bbox_inches="tight")
    plt.close(fig)
    return float(theta[best]), float(vy_ss[best])


def fig_cycle_sweep() -> None:
    df = pd.read_csv(REPO / "cycle_length_sweep_100_1000_outputs" / "coarse_summary.csv")
    best = df.loc[df["best_mean_vy"].idxmax()]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["N"], df["best_mean_vy"], color=C_VY, lw=1.6, marker="o", ms=4)
    ax.scatter([best["N"]], [best["best_mean_vy"]], s=90, facecolors="none",
               edgecolors=INK, lw=1.2, zorder=5)
    ax.annotate(
        f"best: N = {int(best['N'])}, mean vy = +{best['best_mean_vy']:.5f}",
        (best["N"], best["best_mean_vy"]),
        textcoords="offset points",
        xytext=(-120, -30),
        color=SECONDARY,
        fontsize=9,
    )
    ax.set_title("Best mean vy vs. cycle length N (coarse sweep)")
    ax.set_xlabel("cycle length N  (ticks)")
    ax.set_ylabel("mean vy  (blocks / tick)")
    style_ax(ax)
    fig.savefig(IMG / "cycle_sweep.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    IMG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPO / "cycle_opt_outputs" / "solution_N255.csv")
    S = df["s"].to_numpy()
    vx0, vy0 = float(df["vx"].iloc[0]), float(df["vy"].iloc[0])
    N = len(S)

    # Validate the stored solution against the exact float dynamics
    VX, VY = simulate(vx0, vy0, S)
    drift = float(np.hypot(VX[-1] - vx0, VY[-1] - vy0))
    alt = np.cumsum(df["vy"].to_numpy())

    mean_vy = float(df["vy"].mean())
    mean_vx = float(df["vx"].mean())
    print(f"N = {N}")
    print(f"mean vy = {mean_vy:.8f} blocks/tick  ({20 * mean_vy:.3f} blocks/s)")
    print(f"mean vx = {mean_vx:.8f} blocks/tick  ({20 * mean_vx:.3f} blocks/s)")
    print(f"climb per cycle = {N * mean_vy:.3f} blocks in {N / 20:.2f} s")
    print(f"horizontal distance per cycle = {N * mean_vx:.1f} blocks")
    print(f"periodicity drift after 1 re-simulated cycle = {drift:.3e}")
    print(f"vx range: [{df['vx'].min():.3f}, {df['vx'].max():.3f}]")
    print(f"vy range: [{df['vy'].min():.3f}, {df['vy'].max():.3f}]")
    print(f"ticks with s>0: {(S > 0).sum()}, s<0: {(S < 0).sum()}")

    fig_best_cycle(df, alt)
    fig_flight_path(vx0, vy0, S)
    fig_phase_portrait(df)
    theta_b, vy_b = fig_steady_state()
    print(f"best constant pitch: theta = {theta_b:.2f} deg, steady vy = {vy_b:.5f}")
    fig_cycle_sweep()
    print(f"figures written to {IMG}")


if __name__ == "__main__":
    main()
