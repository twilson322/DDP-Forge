#!/usr/bin/env python3
"""Figures + results/summary.md from the experiment CSVs.  python3 scripts/plot.py --results results"""
import argparse
import pathlib

import numpy as np
import pandas as pd

import plotstyle
from plotstyle import SERIES, MUTED, INK

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ALGO = {"ilqr": ("iLQR", SERIES[1]), "ddp": ("DDP", SERIES[0])}


def md(df, fmt=".4g"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def convergence(res):
    h = pd.read_csv(res / "cost_history.csv")
    cc = pd.read_csv(res / "crosscheck.csv")
    best = cc.groupby("problem")["IPOPT warm-started at C++ solution"].min()
    probs = [("cartpole", "Cart-pole swing-up"), ("acrobot_bounded", "Acrobot swing-up, |u| <= 12 Nm"),
             ("quad2d_obstacles", "Planar quadrotor, 3 obstacles")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for a, (p, title) in zip(ax, probs):
        for algo, (lab, col) in ALGO.items():
            x = h[(h.problem == p) & (h.algo == algo)]
            gap = np.maximum((x.cost - best[p]) / best[p], 1e-12)
            a.semilogy(x.iter, gap, color=col, label=lab)
        a.set_title(title, fontsize=10)
        a.set_xlabel("iteration")
        a.set_ylim(1e-10, 1e2)
    ax[0].set_ylabel("(J - J*) / J*,  J* from IPOPT")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(res / "convergence.png")
    plt.close(fig)
    c = pd.read_csv(res / "converge.csv")
    c = c.merge(cc[["problem", "solver", "IPOPT warm-started at C++ solution", "relative gap"]],
                left_on=["problem", "algo"], right_on=["problem", "solver"]).drop(columns=["solver"])
    c = c[["problem", "algo", "converged", "iterations", "cost", "IPOPT warm-started at C++ solution",
           "relative gap", "max_violation", "time_ms", "ms_per_iter"]]
    return md(c)


def snapshots(res):
    fig, ax = plt.subplots(3, 1, figsize=(8.0, 8.2), gridspec_kw={"height_ratios": [1, 1.25, 1.5]})
    # cart-pole
    d = pd.read_csv(res / "traj_cartpole_ddp.csv")
    l = 0.5
    idx = np.linspace(0, len(d) - 1, 14).astype(int)
    for j, i in enumerate(idx):
        r = d.iloc[i]
        a = 0.25 + 0.75 * j / (len(idx) - 1)
        px, th = r.x0, r.x1
        ax[0].add_patch(Rectangle((px - 0.12, -0.06), 0.24, 0.12, fill=False, ec=SERIES[0], alpha=a, lw=1))
        ax[0].plot([px, px + l * np.sin(th)], [0, -l * np.cos(th)], color=SERIES[1], alpha=a, lw=1.5)
    ax[0].plot(d.x0 + l * np.sin(d.x1), -l * np.cos(d.x1), color=MUTED, lw=0.6, ls=":")
    ax[0].set_aspect("equal")
    ax[0].set_title("Cart-pole swing-up (DDP), 4 s, frames fade in with time", fontsize=10)
    ax[0].set_ylim(-0.65, 0.65)
    # acrobot
    d = pd.read_csv(res / "traj_acrobot_bounded_ddp.csv")
    idx = np.linspace(0, len(d) - 1, 16).astype(int)
    for j, i in enumerate(idx):
        r = d.iloc[i]
        a = 0.2 + 0.8 * j / (len(idx) - 1)
        off = 2.6 * j
        x1, y1 = off + np.sin(r.x0), -np.cos(r.x0)
        x2, y2 = x1 + np.sin(r.x0 + r.x1), y1 - np.cos(r.x0 + r.x1)
        ax[1].plot([off, x1], [0, y1], color=SERIES[0], alpha=a, lw=2)
        ax[1].plot([x1, x2], [y1, y2], color=SERIES[2], alpha=a, lw=2)
        ax[1].plot(off, 0, "o", color=INK, ms=3)
        ax[1].annotate(f"{r.t:.1f}s", (off, -2.3), ha="center", fontsize=7, color=MUTED)
    ax[1].set_aspect("equal")
    ax[1].set_ylim(-2.5, 2.2)
    ax[1].set_title("Acrobot swing-up (DDP, elbow torque only, |u| <= 12 Nm)", fontsize=10)
    ax[1].set_xticks([])
    # quad
    d = pd.read_csv(res / "traj_quad2d_obstacles_ilqr.csv")
    for (cx, cy, r) in [(2.0, 1.2, 0.8), (3.6, 2.3, 0.7), (4.6, 0.8, 0.6)]:
        ax[2].add_patch(Circle((cx, cy), r, color="#c9c8c0"))
    ax[2].plot(d.x0, d.x1, color=SERIES[0], lw=1.5)
    for i in range(0, len(d), 10):
        r = d.iloc[i]
        dx, dy = 0.2 * np.cos(r.x2), 0.2 * np.sin(r.x2)
        ax[2].plot([r.x0 - dx, r.x0 + dx], [r.x1 - dy, r.x1 + dy], color=INK, lw=1.2)
    ax[2].plot(6, 3, "*", color=SERIES[2], ms=12)
    ax[2].set_aspect("equal")
    ax[2].set_title("Planar quadrotor through obstacles (augmented-Lagrangian iLQR), bars show attitude", fontsize=10)
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    fig.savefig(res / "trajectories.png")
    plt.close(fig)


def mpc(res):
    m = pd.read_csv(res / "mpc.csv")
    g = m.groupby(["push_Ns", "controller", "max_iters"]).agg(
        success_pct=("success", lambda v: 100 * v.mean()), final_err_deg=("final_angle_err_deg", "median"),
        mean_ms=("mean_solve_ms", "mean"), p99_ms=("p99_solve_ms", "mean"), max_ms=("max_solve_ms", "max"),
        cost=("cost", "median")).reset_index()
    g.columns = ["push [N s]", "controller", "iterations per step", "success [%]", "median final angle error [deg]",
                 "mean solve [ms]", "p99 solve [ms]", "worst solve [ms]", "median tracking cost"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for f, lab, col in [("mpc_tvlqr_1", "TVLQR + LQR hold", SERIES[7]), ("mpc_mpc_1", "MPC, 1 iteration (RTI)", SERIES[3]),
                        ("mpc_mpc_2", "MPC, 2 iterations", SERIES[0]), ("mpc_mpc_50", "MPC, up to 50 iterations", SERIES[2])]:
        d = pd.read_csv(res / f"{f}_push4.csv")
        ax.plot(d.t, np.degrees(d.theta), color=col, lw=1.2, label=lab)
    ax.axhline(180, color=MUTED, lw=0.8, ls=":")
    ax.axvline(5.0, color=MUTED, lw=0.8)
    ax.annotate("4 N s push", (5.05, 30), fontsize=8, color=MUTED)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pole angle [deg]")
    ax.set_title("Tracking the offline plan on a mismatched plant (pole +20 %, cart friction)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(res / "mpc.png")
    plt.close(fig)
    return md(g, ".3g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = pathlib.Path(ap.parse_args().results)
    parts = ["# Results (generated by scripts/plot.py)", "## Offline solves and IPOPT cross-check\n\n" + convergence(res)]
    snapshots(res)
    parts.append("## MPC\n\n" + mpc(res))
    (res / "summary.md").write_text("\n\n".join(parts) + "\n")
    print((res / "summary.md").read_text())


if __name__ == "__main__":
    main()
