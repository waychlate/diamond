import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, RED = "#2457A6", "#B34712", "#C62828"


def load(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "eval_ttc_predictions.csv"))


def solver_figure(euler_dir: str, heun_dir: str, out: str, heun_label: str) -> None:
    """Overlaid histograms of per-frame pixel MSE and per-frame TTC absolute error."""
    e, h = load(euler_dir), load(heun_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)

    bins = np.linspace(0, max(e.pixel_mse.max(), h.pixel_mse.max()) * 1.02, 50)
    for df, color, name in ((e, BLUE, "Euler"), (h, ORANGE, heun_label)):
        ax1.hist(df.pixel_mse, bins=bins, color=color, alpha=0.6,
                 label=f"{name} (mean {df.pixel_mse.mean():.3f})")
    ax1.set_xlabel("Pixel MSE (per generated frame)")
    ax1.set_ylabel("Count")
    ax1.set_title("Generated-frame error")

    bins = np.linspace(0, 15, 31)
    for df, color, name in ((e, BLUE, "Euler"), (h, ORANGE, heun_label)):
        ax2.hist(df.abs_error, bins=bins, color=color, alpha=0.6,
                 label=f"{name} (MAE {df.abs_error.mean():.2f} s)")
    ax2.set_xlabel("TTC absolute error (s, per point)")
    ax2.set_ylabel("Count")
    ax2.set_title("TTC error")

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    fig.suptitle(f"Euler vs. {heun_label}: {e.episode.nunique()} episodes x {e.lookahead_step.nunique()} steps = {len(e)} points each", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def control_figure(run_dir: str, out: str, dream_label: str) -> None:
    """TTC MAE vs. lookahead step for generated frames and the real-frame control."""
    d = load(run_dir)
    if "real_abs_error" not in d:
        raise SystemExit(f"{run_dir} has no real_abs_error column: rerun the eval with the control.")
    g = d.groupby("lookahead_step")
    n = g.episode.nunique()
    steps = n.index.values
    plt.figure(figsize=(8.5, 5), dpi=200)
    for col, color, label in (("abs_error", ORANGE, dream_label), ("real_abs_error", BLUE, "Real frames (control)")):
        plt.plot(steps, g[col].mean(), color=color, marker="o", markersize=3.5, linewidth=2, label=f"{label} (mean {d[col].mean():.2f} s)")
    plt.xlabel("Lookahead step")
    plt.ylabel("TTC MAE (s)")
    plt.title(f"TTC MAE vs. lookahead, {int(n.max())} episodes", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def points_figure(run_dir: str, out: str, dream_label: str) -> None:
    """Every (episode, step) TTC absolute error vs. lookahead step: generated frames red, real frames blue."""
    d = load(run_dir)
    if "real_abs_error" not in d:
        raise SystemExit(f"{run_dir} has no real_abs_error column: rerun the eval with the control.")
    rng = np.random.default_rng(0)
    jitter = lambda: rng.uniform(-0.2, 0.2, len(d))  # spread points sideways so they don't stack
    plt.figure(figsize=(10, 5.5), dpi=200)
    for col, color, label in (("abs_error", RED, dream_label), ("real_abs_error", BLUE, "Real frames (control)")):
        plt.scatter(d.lookahead_step + jitter(), d[col], s=9, color=color, alpha=0.3, edgecolors="none")
        plt.plot(*zip(*d.groupby("lookahead_step")[col].mean().items()), color=color, linewidth=2.5,
                 label=f"{label} (mean {d[col].mean():.2f} s)")
    plt.xlabel("Lookahead step")
    plt.ylabel("TTC absolute error (s)")
    plt.title(f"TTC error at every point: {d.episode.nunique()} episodes x {d.lookahead_step.nunique()} steps (lines = mean per step)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def heatmap_figure(run_dir: str, out: str, dream_label: str) -> None:
    """2D histogram: TTC absolute error vs. lookahead step, one panel per path, each step column sums to 1."""
    from matplotlib.colors import PowerNorm
    d = load(run_dir)
    if "real_abs_error" not in d:
        raise SystemExit(f"{run_dir} has no real_abs_error column: rerun the eval with the control.")
    n_steps = d.lookahead_step.nunique()
    x_edges = np.arange(0.5, n_steps + 1.5)
    y_edges = np.linspace(0, 15, 31)  # 0.5 s bins
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200, sharey=True)
    for ax, col, cmap, label in ((axes[0], "abs_error", "Reds", dream_label), (axes[1], "real_abs_error", "Blues", "Real frames (control)")):
        h, _, _ = np.histogram2d(d.lookahead_step, d[col].clip(0, 15), bins=[x_edges, y_edges])
        h = h / h.sum(axis=1, keepdims=True)  # fraction of episodes at each step
        m = ax.pcolormesh(x_edges, y_edges, h.T, cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=1))
        fig.colorbar(m, ax=ax, label="Fraction of episodes at that step")
        ax.set_title(f"{label} (mean {d[col].mean():.2f} s)")
        ax.set_xlabel("Lookahead step")
    axes[0].set_ylabel("TTC absolute error (s)")
    fig.suptitle(f"TTC error distribution vs. lookahead: {d.episode.nunique()} episodes, 0.5 s bins", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figures for the Euler/Heun and control-vs-dream comparisons")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solver", help="Euler vs. Heun histograms")
    s.add_argument("--euler", required=True, help="Euler eval folder")
    s.add_argument("--heun", required=True, help="Heun eval folder")
    s.add_argument("--heun-label", default="Heun")
    s.add_argument("--out", default="visualizations/figures/euler_vs_heun.png")
    c = sub.add_parser("control", help="Dream vs. real-frame control, MAE vs. lookahead")
    c.add_argument("--run", required=True, help="Eval folder produced with the control")
    c.add_argument("--dream-label", default="Generated frames (Euler)")
    c.add_argument("--out", default="visualizations/figures/control_vs_generated.png")
    q = sub.add_parser("points", help="All per-point TTC errors vs. lookahead step (dream vs. control)")
    q.add_argument("--run", required=True, help="Eval folder produced with the control")
    q.add_argument("--dream-label", default="Generated frames (Euler)")
    q.add_argument("--out", default="visualizations/figures/ttc_error_all_points.png")
    r = sub.add_parser("heatmap", help="2D histogram of TTC error vs. lookahead step (dream vs. control)")
    r.add_argument("--run", required=True, help="Eval folder produced with the control")
    r.add_argument("--dream-label", default="Generated frames (Euler)")
    r.add_argument("--out", default="visualizations/figures/ttc_error_2d_histogram.png")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.cmd == "solver":
        solver_figure(a.euler, a.heun, a.out, a.heun_label)
    elif a.cmd == "points":
        points_figure(a.run, a.out, a.dream_label)
    elif a.cmd == "heatmap":
        heatmap_figure(a.run, a.out, a.dream_label)
    else:
        control_figure(a.run, a.out, a.dream_label)
