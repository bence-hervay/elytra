import numpy as np
import torch
import matplotlib.pyplot as plt
import cma
from dataclasses import dataclass


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


def sim(vx0: torch.Tensor, vy0: torch.Tensor, a: torch.Tensor):
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
):
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
    ):
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
    ):
        with torch.no_grad():
            # x[0]=log(vx0), x[1]=vy0, x[2]=angle_0, x[3:]=segment slopes -> raw_da[t] for t=0..T-2
            log_vx0 = torch.tensor(float(x[0]))
            vy0 = torch.tensor(float(x[1]))
            angle_0 = torch.tensor(float(x[2]))
            s = torch.tensor(x[3:], dtype=log_vx0.dtype)
            raw_da = da_from_seg(s, L)
            return cls.from_raw(log_vx0, vy0, raw_da, T, w_cyc, angle_0)

    def raw_da_for_gd(self):
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


def plot_policy(p: Policy, title: str):
    print(
        f"{title}: mean(vy)={p.mean_vy:.8f}  mean(vx)={p.mean_vx:.8f}  vx0={p.vx0:.8f}  vy0={p.vy0:.8f}  cyc={p.cyc:.3e}"
    )

    t = np.arange(len(p.a))
    plt.figure(figsize=(10, 4), dpi=300)
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
    plt.show()


def plot_gd_hist(h: GDHist):
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
    fig.savefig("gd_history.png")
    plt.close(fig)


def cma_stage(T: int, pieces: int, w_cyc: float, maxfevals: int, sigma: float):
    L = lens(T, pieces)

    x0 = np.zeros(3 + pieces, dtype=float)
    # x0[0]=log(vx0), x0[1]=vy0, x0[2]=angle_0, x0[3:]=segment slopes
    opts = {"maxfevals": maxfevals}
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


def gd_stage(T: int, iters: int, lr: float, w_cyc: float, p0: Policy):
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

    p1 = Policy.from_raw(log_vx0.detach(), vy0.detach(), raw_da.detach(), T, w_cyc, angle_0.detach())
    h = GDHist(
        vx0=vx0_hist, vy0=vy0_hist, a0=a0_hist, mean_vy=mean_vy_hist, cyc=cyc_hist
    )
    return p1, h


T = 200

pieces = 10
cma_maxfevals = 10000
cma_sigma = 0.5
w_cyc_cma = 10.0

gd_iters = 3000
gd_lr = 0.001
w_cyc_gd = 10.0

p_cma = cma_stage(T, pieces, w_cyc_cma, cma_maxfevals, cma_sigma)
plot_policy(p_cma, "Stage 1 (CMA-ES) optimum")

p_gd, h_gd = gd_stage(T, gd_iters, gd_lr, w_cyc_gd, p_cma)
plot_policy(p_gd, "Stage 2 (GD) optimum")

plot_gd_hist(h_gd)
