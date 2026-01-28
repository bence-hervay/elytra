import numpy as np
import torch
import matplotlib.pyplot as plt
import cma
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def step_v(
    vx: torch.Tensor,
    vy: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(a)
    c2 = c * c

    vx_prev = vx
    vy = vy + 0.06 * c2 - 0.08

    neg = vy < 0.0
    f = 0.1 * vy * c2
    vx = torch.where(neg, vx - f, vx)
    vy = torch.where(neg, vy - f, vy)

    pos = a > 0.0
    acc = 0.04 * vx_prev * torch.sin(a)
    vx = torch.where(pos, vx - acc, vx)
    vy = torch.where(pos, vy + 3.2 * acc, vy)

    vx = 0.9 * vx + 0.1 * vx_prev
    return 0.99 * vx, 0.98 * vy


def lens(T: int, pieces: int) -> torch.Tensor:
    n = T - 1
    base = n // pieces
    rem = n - base * pieces
    arr = [base + 1] * rem + [base] * (pieces - rem)
    return torch.tensor(arr, dtype=torch.long)


def da_from_seg(s: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    w = L.to(dtype=s.dtype)
    mean_s = (s * w).sum() / w.sum()
    s0 = s - mean_s
    return torch.repeat_interleave(s0, L)


def a_from_raw_da(raw_da: torch.Tensor, T: int, angle_0: torch.Tensor) -> torch.Tensor:
    raw_da0 = raw_da - raw_da.mean()
    raw_angle_0 = torch.tan(angle_0)
    raw_a = torch.empty(T)
    raw_a[0] = raw_angle_0
    raw_a[1:] = torch.cumsum(raw_da0, dim=0) + raw_angle_0
    return torch.atan(raw_a)


def sim(
    vx0: torch.Tensor, vy0: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vx = vx0
    vy = vy0
    vx_hist = torch.empty(len(a))
    vy_hist = torch.empty(len(a))
    for t in range(len(a)):
        vx, vy = step_v(vx, vy, a[t])
        vx_hist[t] = vx
        vy_hist[t] = vy
    return vx_hist, vy_hist, vx, vy


def loss_and_metrics(
    vx0: torch.Tensor, vy0: torch.Tensor, a: torch.Tensor, w_cyc: float
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    vx_hist, vy_hist, vxT, vyT = sim(vx0, vy0, a)
    mean_vy = vy_hist.mean()
    mean_vx = vx_hist.mean()
    cyc = (vxT - vx0) ** 2 + (vyT - vy0) ** 2
    loss = -mean_vy + w_cyc * cyc
    return loss, mean_vy, mean_vx, cyc, vx_hist, vy_hist


@dataclass
class Policy:
    vx0: float
    vy0: float
    a: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    mean_vy: float
    mean_vx: float
    cyc: float
    loss: float

    @classmethod
    def from_raw(
        cls,
        log_vx0: torch.Tensor,
        vy0: torch.Tensor,
        raw_da: torch.Tensor,
        T: int,
        w_cyc: float,
        angle_0: torch.Tensor,
    ) -> "Policy":
        with torch.no_grad():
            vx0 = torch.exp(log_vx0)
            a = a_from_raw_da(raw_da, T, angle_0)
            loss, mean_vy, mean_vx, cyc, vx_hist, vy_hist = loss_and_metrics(
                vx0, vy0, a, w_cyc
            )
            return cls(
                vx0=float(vx0.item()),
                vy0=float(vy0.item()),
                a=a.detach().cpu().numpy(),
                vx=vx_hist.detach().cpu().numpy(),
                vy=vy_hist.detach().cpu().numpy(),
                mean_vy=float(mean_vy.item()),
                mean_vx=float(mean_vx.item()),
                cyc=float(cyc.item()),
                loss=float(loss.item()),
            )

    @classmethod
    def from_cma_x(
        cls, x: np.ndarray, T: int, pieces: int, L: torch.Tensor, w_cyc: float
    ) -> "Policy":
        with torch.no_grad():
            # x[0]=log(vx0), x[1]=vy0, x[2]=angle_0, x[3:]=segment slopes -> raw_da[t] for t=0..T-2
            log_vx0 = torch.tensor(float(x[0]))
            vy0 = torch.tensor(float(x[1]))
            angle_0 = torch.tensor(float(x[2]))
            s = torch.tensor(x[3:], dtype=log_vx0.dtype)
            raw_da = da_from_seg(s, L)
            return cls.from_raw(log_vx0, vy0, raw_da, T, w_cyc, angle_0)

    def raw_da_for_gd(self) -> np.ndarray:
        raw_a = np.tan(self.a).astype(float)
        raw_da = (raw_a[1:] - raw_a[:-1]).astype(float)
        return raw_da


@dataclass
class GDHist:
    vx0: np.ndarray
    vy0: np.ndarray
    a0: np.ndarray
    mean_vy: np.ndarray
    cyc: np.ndarray


def plot_policy(p: Policy, title: str, filename: str) -> None:
    print(
        f"{title}: mean(vy)={p.mean_vy:.8f}  mean(vx)={p.mean_vx:.8f}  vx0={p.vx0:.8f}  vy0={p.vy0:.8f}  cyc={p.cyc:.3e}"
    )

    t = np.arange(len(p.a))
    fig = plt.figure(figsize=(10, 4), dpi=300)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    plt.plot(t, p.a, color="black", lw=1, label="a(t)")
    plt.plot(t, p.vy, color="tab:blue", lw=1, label="vy(t)")
    plt.axhline(
        p.mean_vy, color="tab:blue", linestyle="--", linewidth=1, label="mean(vy)"
    )
    plt.plot(t, p.vx, color="tab:orange", lw=1, label="vx(t)")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.title(title)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_gd_hist(h: GDHist, filename: str) -> None:
    t = np.arange(len(h.vx0))
    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=300)
    ax2 = ax1.twinx()

    ax1.plot(t, h.vx0, color="tab:orange", lw=1, label="vx0")
    ax1.plot(t, h.vy0, color="tab:blue", lw=1, label="vy0")
    ax1.plot(t, h.a0, color="black", lw=1, label="a0")

    ax2.plot(t, h.mean_vy, color="tab:blue", linestyle="--", lw=1, label="mean(vy)")

    ax1.set_xlabel("iteration")
    ax1.set_ylabel("vx0, vy0, a0")
    ax2.set_ylabel("mean(vy)")

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, ncol=4, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_vxvy_trajectory(p: Policy, filename: str) -> None:
    """Plot vx,vy trajectory as small dots."""
    vx = p.vx
    vy = p.vy

    fig = plt.figure(figsize=(8, 8), dpi=400)
    ax = fig.add_subplot(111)
    ax.scatter(vx, vy, s=1, c="blue")
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_title("vx,vy Trajectory")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.savefig(filename, bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_angle_possibilities(p: Policy, filename: str, T: int) -> None:
    """Plot 4x4 grid showing angle possibilities at every 10th step."""
    vx = p.vx
    vy = p.vy
    a = p.a

    # Select time steps (every 10th, up to 16 for 4x4 grid)
    step_interval = 10
    selected_steps = [i for i in range(0, min(T, 160), step_interval)][:16]

    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(12, 12), dpi=400)
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.5)

    # Sample angles for possibilities (from -pi/2 to pi/2)
    n_angles = 100
    angles_sample = np.linspace(-np.pi / 2, np.pi / 2, n_angles)

    # Process each selected time step
    for idx, t in enumerate(selected_steps):
        if t >= len(vx) - 1:
            continue

        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        # Current state
        vx_curr = vx[t]
        vy_curr = vy[t]
        a_curr = a[t]

        # Compute next vx,vy for each sampled angle
        vx_nexts_list: list[float] = []
        vy_nexts_list: list[float] = []

        vx_t = torch.tensor(float(vx_curr))
        vy_t = torch.tensor(float(vy_curr))

        for angle in angles_sample:
            vx_next, vy_next = step_v(vx_t, vy_t, torch.tensor(float(angle)))
            vx_nexts_list.append(float(vx_next.item()))
            vy_nexts_list.append(float(vy_next.item()))

        vx_nexts = np.array(vx_nexts_list)
        vy_nexts = np.array(vy_nexts_list)

        # Compute actual next state
        vx_actual_next_t, vy_actual_next_t = step_v(
            vx_t, vy_t, torch.tensor(float(a_curr))
        )
        vx_actual_next = float(vx_actual_next_t.item())
        vy_actual_next = float(vy_actual_next_t.item())

        # Normalize for visibility (include both current state and possibilities)
        vx_all = np.concatenate([[vx_curr], vx_nexts])
        vy_all = np.concatenate([[vy_curr], vy_nexts])

        vx_range = vx_all.max() - vx_all.min()
        vy_range = vy_all.max() - vy_all.min()
        margin = 0.1

        if vx_range > 0:
            vx_min = vx_all.min() - margin * vx_range
            vx_max = vx_all.max() + margin * vx_range
        else:
            vx_min = vx_curr - 0.1
            vx_max = vx_curr + 0.1

        if vy_range > 0:
            vy_min = vy_all.min() - margin * vy_range
            vy_max = vy_all.max() + margin * vy_range
        else:
            vy_min = vy_curr - 0.1
            vy_max = vy_curr + 0.1

        # Plot possibilities curve
        ax.plot(vx_nexts, vy_nexts, "b-", linewidth=0.3, alpha=0.7)

        # Mark original v
        ax.scatter([vx_curr], [vy_curr], s=15, c="green", marker="o", zorder=5)

        # Mark endpoints (min and max pitch)
        min_pitch_idx = np.argmin(angles_sample)
        max_pitch_idx = np.argmax(angles_sample)
        ax.scatter(
            [vx_nexts[min_pitch_idx]],
            [vy_nexts[min_pitch_idx]],
            s=25,
            c="red",
            marker="x",
            linewidths=1,
            zorder=6,
        )
        ax.scatter(
            [vx_nexts[max_pitch_idx]],
            [vy_nexts[max_pitch_idx]],
            s=25,
            c="purple",
            marker="x",
            linewidths=1,
            zorder=6,
        )

        # Mark optimal angle (actual chosen)
        ax.scatter(
            [vx_actual_next],
            [vy_actual_next],
            s=40,
            c="orange",
            marker="x",
            linewidths=1.5,
            zorder=7,
        )

        ax.set_xlim(vx_min, vx_max)
        ax.set_ylim(vy_min, vy_max)
        ax.set_aspect("equal")
        ax.set_xlabel("vx", fontsize=7)
        ax.set_ylabel("vy", fontsize=7)
        ax.set_title(f"t={t}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    fig.suptitle("Angle Possibilities at Selected Time Steps", fontsize=12, y=0.995)
    fig.savefig(filename, bbox_inches="tight", dpi=400)
    plt.close(fig)


def cma_stage(
    T: int, pieces: int, w_cyc: float, maxfevals: int, sigma: float
) -> Policy:
    L = lens(T, pieces)

    x0 = np.zeros(3 + pieces, dtype=float)
    # x0[0]=log(vx0), x0[1]=vy0, x0[2]=angle_0, x0[3:]=segment slopes
    opts = {"maxfevals": maxfevals, "seed": 1}
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    last_print = -1
    while not es.stop():
        xs = es.ask()
        ys = []
        for x in xs:
            with torch.no_grad():
                log_vx0 = torch.tensor(float(x[0]))
                vy0 = torch.tensor(float(x[1]))
                angle_0 = torch.tensor(float(x[2]))
                s = torch.tensor(x[3:], dtype=log_vx0.dtype)
                vx0 = torch.exp(log_vx0)
                raw_da = da_from_seg(s, L)
                a = a_from_raw_da(raw_da, T, angle_0)
                loss, _, _, _, _, _ = loss_and_metrics(vx0, vy0, a, w_cyc)
                ys.append(float(loss.item()))
        es.tell(xs, ys)

        if es.countiter != last_print and (es.countiter <= 5 or es.countiter % 10 == 0):
            last_print = es.countiter
            print(
                f"CMA iter {es.countiter:4d}  evals {es.countevals:6d}/{maxfevals}  best loss {es.best.f:.6f}"
            )

    print("CMA stop:", es.stop())
    print("CMA best loss:", es.result.fbest)

    return Policy.from_cma_x(es.result.xbest, T, pieces, L, w_cyc)


def gd_stage(
    T: int, iters: int, lr: float, w_cyc: float, p0: Policy
) -> tuple[Policy, GDHist]:
    raw_da_init = p0.raw_da_for_gd()
    log_vx0 = torch.nn.Parameter(torch.tensor(float(np.log(p0.vx0))))
    vy0 = torch.nn.Parameter(torch.tensor(float(p0.vy0)))
    angle_0 = torch.nn.Parameter(torch.tensor(float(p0.a[0])))
    raw_da = torch.nn.Parameter(torch.tensor(raw_da_init))

    optim = torch.optim.Adam([log_vx0, vy0, angle_0, raw_da], lr=lr)

    vx0_hist = np.empty(iters, dtype=float)
    vy0_hist = np.empty(iters, dtype=float)
    a0_hist = np.empty(iters, dtype=float)
    mean_vy_hist = np.empty(iters, dtype=float)
    cyc_hist = np.empty(iters, dtype=float)

    for it in range(iters):
        optim.zero_grad()

        vx0 = torch.exp(log_vx0)
        a = a_from_raw_da(raw_da, T, angle_0)
        loss, mean_vy, mean_vx, cyc, _, _ = loss_and_metrics(vx0, vy0, a, w_cyc)

        loss.backward()
        optim.step()

        vx0_hist[it] = float(vx0.detach().item())
        vy0_hist[it] = float(vy0.detach().item())
        a0_hist[it] = float(a[0].detach().item())
        mean_vy_hist[it] = float(mean_vy.detach().item())
        cyc_hist[it] = float(cyc.detach().item())

        if it <= 5 or (it + 1) % 100 == 0:
            print(
                f"GD iter {it + 1:5d}/{iters}  mean(vy) {mean_vy_hist[it]:.8f}  "
                f"vx0 {vx0_hist[it]:.8f}  vy0 {vy0_hist[it]:.8f}  cyc {cyc_hist[it]:.3e}"
            )

    p1 = Policy.from_raw(
        log_vx0.detach(), vy0.detach(), raw_da.detach(), T, w_cyc, angle_0.detach()
    )
    h = GDHist(
        vx0=vx0_hist, vy0=vy0_hist, a0=a0_hist, mean_vy=mean_vy_hist, cyc=cyc_hist
    )
    return p1, h


T = 160

pieces = 10
cma_maxfevals = 10000
cma_sigma = 0.5
w_cyc_cma = 10.0

gd_iters = 2000
gd_lr = 0.001
w_cyc_gd = 10.0

output_dir = Path("outputs") / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_dir.mkdir(parents=True, exist_ok=True)

p_cma = cma_stage(T, pieces, w_cyc_cma, cma_maxfevals, cma_sigma)
plot_policy(p_cma, "Stage 1 (CMA-ES) optimum", str(output_dir / "cma_optimum.png"))

p_gd, h_gd = gd_stage(T, gd_iters, gd_lr, w_cyc_gd, p_cma)
plot_policy(p_gd, "Stage 2 (GD) optimum", str(output_dir / "gd_optimum.png"))

plot_gd_hist(h_gd, str(output_dir / "gd_history.png"))
plot_vxvy_trajectory(p_gd, str(output_dir / "vxvy_trajectory.png"))
plot_angle_possibilities(p_gd, str(output_dir / "angle_possibilities.png"), T)
