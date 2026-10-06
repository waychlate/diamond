import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = "#2457A6", "#B34712"


def load(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "eval_ttc_predictions.csv"))


def solver_figure(euler_dir: str, heun_dir: str, out: str, heun_label: str) -> None:
    """Overlaid histograms of per-frame pixel MSE and per-frame TTC absolute error."""
    e, h = load(euler_dir), load(heun_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)

    lo = min(e.pixel_mse.min(), h.pixel_mse.min())
    hi = max(e.pixel_mse.max(), h.pixel_mse.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    for df, color, name in ((e, BLUE, "Euler"), (h, ORANGE, heun_label)):
        ax1.hist(df.pixel_mse, bins=bins, color=color, alpha=0.6,
                 label=f"{name} (mean {df.pixel_mse.mean():.3f})")
    ax1.set_xscale("log")
    ax1.set_xlabel("Pixel MSE (per generated frame, log scale)")
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
    fig.suptitle(f"Euler vs. {heun_label}: {len(e)} points each (runs used different random episodes)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def control_figure(run_dir: str, out: str, dream_label: str) -> None:
    """TTC MAE vs. lookahead step for generated frames and the real-frame control, with SEM bands."""
    d = load(run_dir)
    if "real_abs_error" not in d:
        raise SystemExit(f"{run_dir} has no real_abs_error column: rerun the eval with the control.")
    g = d.groupby("lookahead_step")
    n = g.episode.nunique()
    steps = n.index.values
    plt.figure(figsize=(8.5, 5), dpi=200)
    for col, color, label in (("abs_error", ORANGE, dream_label), ("real_abs_error", BLUE, "Real frames (control)")):
        m, s = g[col].mean(), g[col].std() / np.sqrt(n)
        plt.plot(steps, m, color=color, marker="o", markersize=3.5, linewidth=2, label=f"{label} (mean {d[col].mean():.2f} s)")
        plt.fill_between(steps, m - s, m + s, color=color, alpha=0.2)
    plt.xlabel("Lookahead step")
    plt.ylabel("TTC MAE (s)")
    plt.title(f"TTC MAE vs. lookahead, {int(n.max())} episodes (band = ±1 SEM)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
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
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.cmd == "solver":
        solver_figure(a.euler, a.heun, a.out, a.heun_label)
    else:
        control_figure(a.run, a.out, a.dream_label)
