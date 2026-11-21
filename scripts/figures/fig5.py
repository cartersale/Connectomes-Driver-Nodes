#!/usr/bin/env python
"""Figure 5: spectral gap, q1 and strength assortativity in connectomes and null networks.

Weakening links within the core is in figS_corecut.py.

A  Spectral gap l1 - l2 by scale: human connectomes and the three null models (strength-preserving,
   geometry-preserving, weight-shuffled). Small dots: subjects (nulls: mean of each subject's 50 nulls). Large dots:
   means over subjects.
B  q1 of degree-strength sets (mean over k = 2, 3, 5; c = 0.1) by scale: human, nulls at the same c, and nulls at a
   c matched to the human decay ratio. Nulls: mean over the three null models for each subject.
C  Per subject, the excess of the human connectome over the mean of its 50 strength-preserving nulls in edge-weighted
   strength assortativity (x) and in spectral gap (y), coloured by scale, with Spearman rho per scale.
Reads null_models/{sets.csv.gz, rayleigh/networks.csv.gz, rayleigh/per_subject.csv}. Run: python scripts/figures/fig5.py --run <name>
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from hoc import figstyle as fs

HUMAN = fs.SCALE_COLORS[114]  # one purple for human data in A and B (the x-axis already gives the scale)
NETS = {  # network: (label, colour, marker)
    "real": ("Human", HUMAN, "o"),
    "strength": ("Strength-preserving", "#5f5e59", "s"),
    "geometric": ("Geometry-preserving", "#85847e", "D"),
    "weights": ("Weight-shuffled", "#a9a8a2", "^"),
}
Q1 = {  # condition: (label, colour)
    "human": ("Human", HUMAN),
    "fixed": ("Null, same c", "#a9a8a2"),
    "gap": ("Null, gap-matched c", "#5f5e59"),  # drawn hollow
}


def strips(ax, groups: list, width: float = 0.66, ms: float = 4.5, dot: float = 2.5, stat=np.mean, ylim=None) -> None:
    """groups: list (per condition) of (dict scale -> per-subject values, colour, marker). x = scale index."""
    rng = np.random.default_rng(1)
    n = len(groups)
    off = (np.arange(n) - (n - 1) / 2) * width / n
    for g, grp in enumerate(groups):
        vals, color, marker = grp[:3]
        hollow = len(grp) > 3 and grp[3]
        for i, sc in enumerate(fs.SCALES):
            v = np.asarray(vals[sc])
            x = i + off[g]
            color = fs.SCALE_COLORS[sc] if grp[1] == "scale" else grp[1]
            if ylim is not None:  # points beyond the axis are drawn at its edge
                v_show = np.clip(v, ylim[0], ylim[1])
            else:
                v_show = v
            if hollow:
                ax.scatter(x + rng.uniform(-0.035, 0.035, len(v)), v_show, s=dot * 1.6, facecolor="none", edgecolor=color,
                           alpha=0.5, lw=0.4, marker=marker, rasterized=True)
                ax.plot(x, stat(v), marker, ms=ms, mfc="white", mec=color, mew=0.9, zorder=5)
            else:
                ax.scatter(x + rng.uniform(-0.035, 0.035, len(v)), v_show, s=dot, color=color, alpha=0.45, lw=0,
                           marker=marker, rasterized=True)
                ax.plot(x, stat(v), marker, ms=ms, color=color, mec="white" if color != "#cfcec9" else fs.MUTED,
                        mew=0.6, zorder=5)
    ax.set_xticks(range(len(fs.SCALES)), [str(s) for s in fs.SCALES])
    ax.set_xlim(-0.5, len(fs.SCALES) - 0.5)
    ax.tick_params(axis="x", length=0)
    fs.style(ax)


def handles(items, ms=4):
    items = [(l, fs.SCALE_COLORS[114] if c == "scale" else c, m) for l, c, m in items]
    return [mpl.lines.Line2D([], [], ls="", marker=m, ms=ms, color=c, mec="white" if c != "#cfcec9" else fs.MUTED,
                             mew=0.5, label=l) for l, c, m in items]


def panel_gap(ax, ray: pd.DataFrame) -> None:
    per = ray.groupby(["null", "scale", "subject"]).gap.mean()
    groups = [({sc: per[(n, sc)].to_numpy() for sc in fs.SCALES}, c, m) for n, (_, c, m) in NETS.items()]
    strips(ax, groups)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Spectral gap  ℓ1 − ℓ2")
    ax.set_xlabel("Regions")
    ax.legend(handles=handles([(l, c, m) for l, c, m in NETS.values()]), loc="upper right", ncol=1, fontsize=6,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    for sc in fs.SCALES:
        print(f"5A  gap {sc}: " + ", ".join(f"{n} {per[(n, sc)].mean():.2f}" for n in NETS))


def panel_q1(ax, sets: pd.DataFrame) -> None:
    d = sets[(sets.c_ref == 0.1) & (sets.k == "avg2+")]
    per = d.groupby(["null", "c_mode", "scale", "subject_id"]).degree_q1.mean()
    human = per.loc["real", "fixed"]
    nulls = per.drop("real", level="null").groupby(["c_mode", "scale", "subject_id"]).mean()  # mean over null models
    vals = {"human": human, "fixed": nulls.loc["fixed"], "gap": nulls.loc["gap"]}
    groups = [({sc: vals[c].loc[sc].to_numpy() for sc in fs.SCALES}, col, "o", c == "gap") for c, (_, col) in Q1.items()]
    strips(ax, groups)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("$q_1$ of degree-strength sets")
    ax.set_xlabel("Regions")
    hs = handles([(Q1["human"][0], HUMAN, "o"), (Q1["fixed"][0], Q1["fixed"][1], "o")])
    hs.append(mpl.lines.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=Q1["gap"][1], mew=0.9,
                               label=Q1["gap"][0]))
    ax.legend(handles=hs, loc="upper right", ncol=1, fontsize=6,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    for sc in fs.SCALES:
        print(f"5B  q1 {sc}: " + ", ".join(f"{c} {vals[c].loc[sc].mean():.2f}" for c in Q1))


def panel_assort(ax, per_subject: pd.DataFrame) -> None:
    """Excess assortativity vs excess gap, human minus the mean of that subject's strength-preserving nulls."""
    d = per_subject[per_subject.null == "strength"]
    for i, sc in enumerate(fs.SCALES):
        g = d[d.scale == sc]
        ax.scatter(g.d_r_W, g.d_gap, s=7, color=fs.SCALE_COLORS[sc], marker=fs.SCALE_MARKERS[sc], lw=0, alpha=0.85)
        rho = stats.spearmanr(g.d_r_W, g.d_gap).statistic
        ax.text(0.04, 0.96 - 0.08 * i, f"{sc}: ρ = {rho:.2f}", transform=ax.transAxes, fontsize=6,
                color=fs.SCALE_COLORS[sc], ha="left", va="top")
        print(f"5C  {sc}: rho(excess r_W, excess gap) vs strength nulls = {rho:.2f}")
    ax.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=fs.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Excess strength assortativity")
    ax.set_ylabel("Excess spectral gap")
    fs.style(ax)


def main():
    ctx, root = fs.open_run("figures/fig5", ("null_models/02_analyze", "null_models/04_rayleigh_bound"))
    fs.setup()
    nulls = (root / "null_models")
    sets = pd.read_csv(nulls / "sets.csv.gz", usecols=["null", "scale", "subject_id", "c_mode", "c_ref", "k", "degree_q1"],
                       dtype={"k": str})
    ray = pd.read_csv(nulls / "rayleigh" / "networks.csv.gz", usecols=["null", "scale", "subject", "gap", "r_W"])
    per_subject = pd.read_csv(nulls / "rayleigh" / "per_subject.csv")

    fig = plt.figure(figsize=(fs.FULL, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    panel_gap(ax, ray)
    fs.panel_label(ax, "A", x=-0.2, y=1.02)
    ax = fig.add_subplot(gs[0, 1])
    panel_q1(ax, sets)
    fs.panel_label(ax, "B", x=-0.2, y=1.02)
    ax = fig.add_subplot(gs[0, 2])
    panel_assort(ax, per_subject)
    fs.panel_label(ax, "C", x=-0.2, y=1.02)
    fs.finish(fig, ctx, "fig5")

if __name__ == "__main__":
    main()
