#!/usr/bin/env python
"""Figure 4: the next slowest modes divide the core and fade faster by the spectral gap.

A  The front-back split of the core: consensus axis a of the v2-v3 plane on the medial surface (114 regions; red =
   front, blue = back), outlining the top 20% of regions by v1. Right: which eigenvector holds the split in each
   subject (v2, v3, later), by scale.
B  Removing v1 (projecting it out of the Gramian): degree-minus-cycle q1 before and after, and how far degree sets
   still exceed random sets after. Mean over k = 2, 3, 5 at c = 0.1.
C  Excess q1 of degree sets over random sets (mean over k = 1, 2, 3, 5) as damping c grows; mean and 95% CI over
   subjects. Dotted lines: c = 0.1 and the nctpy default c = 1.
D  Spectral gap vs excess q1 at c = 0.1, one dot per subject, with Spearman rho per scale.

Reads revision/mechanism/second_subspace_{axes,subjects}.csv and revision/q1/{p3_perp, p1_vs_random}.csv,
decay_curves.csv.gz. Run: python scripts/revision/figures/fig4.py --run <name> [--draft]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

SCALE = 114
SLOT = ("#2a78d6", "#eb6834", "#cfcec9")  # v2, v3, later (direct-labelled)


def panel_split(fig, cell, root: Path, axes_tab: pd.DataFrame, subj: pd.DataFrame) -> None:
    t = axes_tab[axes_tab.scale == SCALE].sort_values("node_index")
    geo = fs.surface(SCALE, t["region_label"].tolist())
    a = t["a"].to_numpy()
    v1 = t["v1_mean"].to_numpy()
    top = v1 >= np.sort(v1)[::-1][int(np.ceil(0.2 * len(v1))) - 1]
    m = float(np.abs(a).max())
    norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
    sub = cell.subgridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.22)
    for j, view in enumerate(("L_medial", "R_medial")):
        ax = fig.add_subplot(sub[0, j])
        fs.draw_view(ax, geo, view, a, fs.DIV, norm, outline=top)
        ax.set_title("Left medial" if j == 0 else "Right medial", fontsize=6.5, color=fs.INK2, pad=1)
        if j == 0:
            fs.panel_label(ax, "A", x=-0.02, y=1.04)
            ax.text(0.02, 0.08, "back", transform=ax.transAxes, color=fs.DIV(0.05), fontsize=6, ha="left", va="top")
            ax.text(0.98, 0.08, "front", transform=ax.transAxes, color=fs.DIV(0.95), fontsize=6, ha="right", va="top")
        else:
            ax.text(0.02, 0.08, "front", transform=ax.transAxes, color=fs.DIV(0.95), fontsize=6, ha="left", va="top")
            ax.text(0.98, 0.08, "back", transform=ax.transAxes, color=fs.DIV(0.05), fontsize=6, ha="right", va="top")
    # which eigenvector holds the split
    ax = fig.add_subplot(sub[0, 2])
    names = ("v2", "v3", "later")
    for yi, sc in enumerate(fs.SCALES):
        g = subj[subj.scale == sc]
        counts = [(g.a_best == 2).sum(), (g.a_best == 3).sum(), (g.a_best >= 4).sum()]
        left = 0
        for k, cnt in enumerate(counts):
            ax.barh(yi, cnt, left=left, color=SLOT[k], height=0.62, edgecolor="white", linewidth=1.0)
            if cnt >= 8:
                ax.text(left + cnt / 2, yi, str(cnt), ha="center", va="center", fontsize=5.8, color="white" if k < 2 else fs.INK2)
            left += cnt
        print(f"4A  split slot {sc}: v2 {counts[0]}, v3 {counts[1]}, later {counts[2]} of {len(g)}")
    g = subj[subj.scale == fs.SCALES[0]]
    first = [(g.a_best == 2).sum(), (g.a_best == 3).sum(), (g.a_best >= 4).sum()]
    centres = np.cumsum(first) - np.array(first) / 2
    for k, name in enumerate(names):  # header over the top bar's segments
        ax.text(centres[k], -0.45, name, fontsize=6, color=SLOT[k] if k < 2 else fs.INK2, ha="center", va="bottom", fontweight="bold")
    ax.set_yticks(range(3), [f"{s}" for s in fs.SCALES])
    ax.set_title("Eigenvector holding the split", fontsize=6.5, color=fs.INK2, pad=9)
    ax.set_xlabel("Subjects")
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 35, 70])
    ax.invert_yaxis()
    fs.style(ax)
    ax.tick_params(axis="y", length=0)


def panel_removal(ax, p3: pd.DataFrame, p1: pd.DataFrame) -> None:
    k = "avg2+"
    d = p3[(p3.metric == "q1") & (p3.c == 0.1) & (p3.k.astype(str) == k)].set_index("scale")
    r = p1[(p1.family == "degree_strength") & (p1.c == 0.1) & (p1.k.astype(str) == k) & (p1.metric == "q1_perp")].set_index("scale")
    bars = [("Degree − cycle", lambda s: -d.loc[s, "full_mean"], fs.INK),
            ("… after removing v1", lambda s: -d.loc[s, "perp_mean"], fs.MUTED),
            ("Degree − random,\nafter removing v1", lambda s: r.loc[s, "value_mean"] - r.loc[s, "random_mean"], fs.HUB)]
    w = 0.26
    for i, sc in enumerate(fs.SCALES):
        for j, (_, f, col) in enumerate(bars):
            ax.bar(i + (j - 1) * w, f(sc), width=w - 0.03, color=col)
        print(f"4B  {sc}: degree-cycle {-d.loc[sc, 'full_mean']:.3f} -> {-d.loc[sc, 'perp_mean']:.3f} "
              f"({100 * d.loc[sc, 'fraction_removed']:.0f}% removed); degree-random after {bars[2][1](sc):.3f}")
    ax.set_xticks(range(3), [f"{s}" for s in fs.SCALES])
    ax.set_xlabel("Regions")
    ax.set_ylabel("Difference in q1")
    ax.tick_params(axis="x", length=0)
    handles = [mpl.patches.Patch(color=c, label=l) for l, _, c in bars]
    ax.legend(handles=handles, loc="upper right", fontsize=5.6, handlelength=1, borderaxespad=0.1)
    ax.set_ylim(0, 0.48)
    fs.style(ax)


def panel_sweep(ax, curves: pd.DataFrame) -> None:
    for sc in fs.SCALES:
        g = curves[curves.scale == sc].groupby("c").dr_q1
        m, se, n = g.mean(), g.sem(), g.count()
        half = se * stats.t.ppf(0.975, n - 1)
        ax.fill_between(m.index, m - half, m + half, color=fs.SCALE_COLORS[sc], alpha=0.25, lw=0)
        ax.plot(m.index, m, color=fs.SCALE_COLORS[sc], lw=1.3, label=str(sc))
        vals = {c: m.iloc[np.argmin(np.abs(np.log(m.index / c)))] for c in (0.01, 0.1, 1.0)}
        print(f"4C  excess q1 {sc}: " + ", ".join(f"c={c}: {v:.2f}" for c, v in vals.items()))
    ax.set_xscale("log")
    fs.reference_lines(ax)
    ax.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Damping c")
    ax.set_ylabel("Excess q1, degree − random")
    ax.legend(title="Regions", loc="lower left", title_fontsize=6, handlelength=1.2, borderaxespad=0.1)
    fs.style(ax)


def panel_gap(ax, curves: pd.DataFrame) -> None:
    d = curves[np.isclose(curves.c, 0.1)]
    for i, sc in enumerate(fs.SCALES):
        g = d[d.scale == sc]
        ax.scatter(g.gap, g.dr_q1, s=7, color=fs.SCALE_COLORS[sc], marker=fs.SCALE_MARKERS[sc], lw=0, alpha=0.85)
        rho = stats.spearmanr(g.gap, g.dr_q1).statistic
        ax.text(0.98, 0.04 + 0.08 * (2 - i), f"{sc}: ρ = {rho:.2f}", transform=ax.transAxes, fontsize=6, color=fs.SCALE_COLORS[sc],
                ha="right", va="bottom")
        print(f"4D  gap vs excess q1 at c = 0.1, {sc}: rho = {rho:.2f}")
    ax.set_xlabel("Spectral gap  ℓ1 − ℓ2")
    ax.set_ylabel("Excess q1 at c = 0.1")
    fs.style(ax)


def main():
    ctx, root, draft = fs.open_run("revision/figures/fig4", ("revision/mechanism/05_second_subspace", "revision/q1/02_tables",
                                                             "revision/q1/03_decay_collapse"))
    fs.setup()
    mech, q1 = fs.rev_dir(root, "mechanism", draft), fs.rev_dir(root, "q1", draft)
    axes_tab = pd.read_csv(mech / "second_subspace_axes.csv")
    subj = pd.read_csv(mech / "second_subspace_subjects.csv")
    p3 = pd.read_csv(q1 / "p3_perp.csv", dtype={"k": str})
    p1 = pd.read_csv(q1 / "p1_vs_random.csv", dtype={"k": str})
    curves = pd.read_csv(q1 / "decay_curves.csv.gz")

    fig = plt.figure(figsize=(fs.FULL, 4.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.42)
    panel_split(fig, gs[0], root, axes_tab, subj)
    low = gs[1].subgridspec(1, 3, wspace=0.42)
    for j, (letter, fn, data) in enumerate((("B", panel_removal, (p3, p1)), ("C", panel_sweep, (curves,)), ("D", panel_gap, (curves,)))):
        ax = fig.add_subplot(low[0, j])
        fn(ax, *data)
        fs.panel_label(ax, letter, x=-0.25, y=1.02)
    fs.finish(fig, ctx, root, "fig4", draft)


if __name__ == "__main__":
    main()
