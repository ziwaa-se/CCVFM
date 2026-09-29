#!/usr/bin/env python3
"""Render the README charts from results/paper_headline.csv.

    python tools/make_readme_figures.py        # -> assets/fid_vs_nfe.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# categorical slots 1-3 of a CVD-validated palette, assigned in fixed order
COLORS = {"CCVFM": "#2a78d6", "HRF2": "#eb6834", "Rectified Flow": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"


def load():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "paper_headline.csv"))))
    for r in rows:
        r["nfe"], r["fid"] = int(r["nfe"]), float(r["fid"])
    return rows


def panel(ax, rows, dataset, methods, title, logy, label_dy=None):
    for m in methods:
        pts = sorted((r["nfe"], r["fid"]) for r in rows
                     if r["dataset"] == dataset and r["method"] == m)
        xs, ys = zip(*pts)
        lw = 2.4 if m == "CCVFM" else 2.0
        ax.plot(xs, ys, "-o", color=COLORS[m], lw=lw, ms=8, mec="white", mew=2,
                label=m, zorder=3 if m == "CCVFM" else 2)
        dy = (label_dy or {}).get(m, 0)
        ax.annotate(f"{m}", (xs[-1], ys[-1]), xytext=(8, dy), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("function evaluations (NFE)", color=MUTED)
    ax.set_ylabel("FID$_{50k}$  (lower is better)", color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def main():
    rows = load()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    panel(axes[0], rows, "MNIST", ["Rectified Flow", "HRF2", "CCVFM"],
          "MNIST: FID 0.75 at 51 NFE vs. HRF2's 2.57 at 500 NFE", logy=True)
    axes[0].set_xlim(4, 1500)
    panel(axes[1], rows, "ImageNet-32", ["HRF2", "CCVFM"],
          "ImageNet-32: same U-Net, only the source changes", logy=False,
          label_dy={"HRF2": 9, "CCVFM": -9})
    axes[1].set_xlim(8, 90)
    axes[1].set_xticks([11, 21, 51])
    axes[1].set_xticklabels(["11", "21", "51"])
    for nfe, (h, c), off in zip([11, 21, 51], [(20.29, 12.55), (12.49, 9.51), (9.02, 8.76)],
                                [(6, 0), (6, 0), (-36, 14)]):
        axes[1].annotate(f"−{(h - c) / h:.0%}", (nfe, (h + c) / 2), xytext=off,
                         textcoords="offset points", fontsize=9, color=MUTED, va="center")
    fig.tight_layout()
    out = os.path.join(ROOT, "assets", "fid_vs_nfe.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print("saved", out)
    source_ablation()


def source_ablation():
    """Matched-budget MNIST ablation: surrogate source (CCVFM) vs N(0, I) source (HRF2)."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "mnist_source_ablation.csv"))))
    series = {"ccvfm_surrogate": ("CCVFM (coreset surrogate source)", COLORS["CCVFM"]),
              "gaussian_hrf2": ("Gaussian source (HRF2)", COLORS["HRF2"])}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, (label, color) in series.items():
        sub = [r for r in rows if r["source"] == key]
        nfes = sorted({int(r["nfe"]) for r in sub})
        per = {n: [float(r["fid_50k"]) for r in sub if int(r["nfe"]) == n] for n in nfes}
        means = [sum(v) / len(v) for v in per.values()]
        for n, vals in per.items():
            ax.scatter([n] * len(vals), vals, s=14, color=color, alpha=0.35, lw=0, zorder=2)
        n_seeds = len(next(iter(per.values())))
        ax.plot(nfes, means, "-o", color=color, lw=2.2, ms=8, mec="white", mew=2, zorder=3,
                label=f"{label}, {n_seeds} seeds")
        ax.annotate(f"{means[1]:.2f}", (nfes[1], means[1]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_xticks([6, 11, 21])
    ax.set_xticklabels(["6", "11", "21"])
    ax.set_title("MNIST, same network, same Stage III budget (80k steps):\nonly the inner-flow source differs",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("function evaluations (NFE)", color=MUTED)
    ax.set_ylabel("FID$_{50k}$  (lower is better)", color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    out = os.path.join(ROOT, "assets", "source_ablation.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
