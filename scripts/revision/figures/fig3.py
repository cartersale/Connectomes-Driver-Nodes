#!/usr/bin/env python
"""Figure 3: the shared pattern is a mode of the midline core.

A  v1 on the left medial surface: group mean and four individual subjects, chosen at the min / 25th / 50th / 75th
   percentile of their correlation with the group mean (114 regions). Right: that correlation for every subject at
   every scale.
B  Strength s_i vs m_i, the strength-weighted mean v1 loading of the region's neighbours, for every region and
   subject (114 regions; log axes). Line: median m_i within strength deciles. Text: median within-subject Spearman rho.
C  Loading on v1 (L_S) of each criterion's driver set: subject means, with the middle 90% of random sets shaded.
D  Cosine between the Gramian's top eigenvector (the most reachable pattern) and v1, same layout as C.
C and D: c = 0.1, k = 3, 114 regions; rows ordered by mean L_S.

Reads processed/connectomes_scale-*.npz, revision/mechanism/{eigvec_subjects, eigvec_regions, gramian_alignment,
node_v1_strength}.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402

SCALE, K = 114, 3
ROWS = ["degree_strength", "eigenvector_centrality", "top_avg_ctrl", "cycle_criticality", "betweenness",
        "participation_coefficient", "cycle_participation", "triangle_participation", "cycle_resid", "top_modal_gu",
        "low_perron"]


def panel_subjects(fig, cell, root: Path, regions: pd.DataFrame, subj: pd.DataFrame) -> None:
    labels = pd.read_csv(root / "processed" / f"atlas_labels_scale-{SCALE}.csv").sort_values("node_index")["region_label"].tolist()
    geo = fs.surface(SCALE, labels)
    stack = np.load(root / "processed" / f"connectomes_scale-{SCALE}.npz")["adjacency"]
    s = subj[subj.scale == SCALE].sort_values("v1_consistency").reset_index(drop=True)
    picks = [(q, s.iloc[int(round(q * (len(s) - 1)))]) for q in (0.1, 0.5, 0.9)]
    mean = regions[regions.scale == SCALE].sort_values("node_index")["v1_mean"].to_numpy()
    maps = [("Group mean", mean, "")]
    for q, p in picks:
        si = int(p.subject_id.split("-")[1])
        maps.append((f"r = {p.v1_consistency:.2f}", fs.leading_eigvec(stack[si])[1], f"{int(q*100)}th pct."))
    sub = cell.subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.35, 1.1], wspace=0.08)
    vmax = max(m[1].max() for m in maps)
    norm = mpl.colors.Normalize(0, vmax)
    for j, (title, v, _) in enumerate(maps):
        ax = fig.add_subplot(sub[0, j])
        fs.draw_view(ax, geo, "L_medial", v, fs.SEQ, norm)
        if j > 0:
            ax.text(0.5, -0.06, maps[j][2], transform=ax.transAxes, fontsize=6, color=fs.INK2, ha="center", va="top")
        ax.set_title(title, fontsize=6.5, color=fs.INK if j == 0 else fs.INK2, pad=1)
        if j == 0:
            fs.panel_label(ax, "A", x=0.0, y=1.08)
            cax = ax.inset_axes([0.3, -0.1, 0.45, 0.05])
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=fs.SEQ), cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            cb.set_ticks([])
            cax.text(-0.04, 0.5, "v1 low", transform=cax.transAxes, ha="right", va="center", fontsize=6, color=fs.INK2)
            cax.text(1.04, 0.5, "high", transform=cax.transAxes, ha="left", va="center", fontsize=6, color=fs.INK2)
        if j == 1:
            ax.text(0.0, 1.22, "Individual subjects, left medial (percentile of r)", transform=ax.transAxes, fontsize=6.5, color=fs.INK2, ha="left")
    # strip of r with the group mean, every scale
    ax = fig.add_subplot(sub[0, 5])
    rng = np.random.default_rng(0)
    for i, sc in enumerate(fs.SCALES):
        r = subj[subj.scale == sc]["v1_consistency"].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(r)), r, s=5, color=fs.SCALE_COLORS[sc], lw=0, alpha=0.8)
        ax.plot([i - 0.28, i + 0.28], [np.median(r)] * 2, color=fs.INK, lw=1.0)
        print(f"3A  v1 vs group mean, {sc}: median r = {np.median(r):.3f} (range {r.min():.2f}-{r.max():.2f})")
    ax.set_xticks(range(3), [str(s) for s in fs.SCALES])
    ax.set_xlabel("Regions")
    ax.set_ylabel("r with group mean")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    fs.style(ax)


def panel_strength(ax, nodes: pd.DataFrame) -> None:
    """Strength s_i against neighbours' loading m_i: do strong regions connect to high-loading regions?"""
    d = nodes[(nodes.strength > 0) & (nodes.m > 0)]
    rho = d.groupby(["scale", "subject_id"])[["strength", "m"]].apply(lambda g: stats.spearmanr(g.strength, g.m)[0])
    med = rho.groupby("scale").median()
    d = d[d.scale == SCALE]
    ax.scatter(d["strength"], d["m"], s=2.5, lw=0, alpha=0.35, color=fs.RANDOM, rasterized=True)
    # median m_i within strength deciles (pooled over subjects)
    bins = np.quantile(d["strength"], np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(d["strength"], bins[1:-1]), 0, 9)
    g = d.groupby(idx)
    ax.plot(g["strength"].median(), g["m"].median(), color=fs.INK, lw=1.4, marker="o", ms=2.5, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Strength $s_i$")
    ax.set_ylabel("Neighbours' $v_1$ loading $m_i$")
    fs.style(ax)
    ax.text(0.04, 0.96, f"median within-subject\nSpearman $\\rho$ = {med[SCALE]:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color=fs.INK2)
    print("3B  median within-subject rho(s, m): " + "  ".join(f"{s}: {v:.2f}" for s, v in med.items())
          + f"; positive in {int((rho > 0).sum())}/{len(rho)} subject-scales")


def dotplot(ax, align: pd.DataFrame, col: str, xlabel: str, show_names: bool) -> None:
    a = align[(align.scale == SCALE) & (align.k == K)]
    rand = a[a.family == "random"].groupby("subject_id")[col].mean().to_numpy()  # one random mean per subject, like the dots
    lo, hi = np.percentile(rand, [5, 95])
    ax.axvspan(lo, hi, color=fs.RANDOM, alpha=0.25, lw=0)
    ax.axvline(np.mean(rand), color=fs.RANDOM, lw=1.0)
    for i, fam in enumerate(ROWS):
        v = a[a.family == fam][col].to_numpy()
        name, color = fs.CRITERIA[fam]
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(v)), s=3, color=color, alpha=0.35, lw=0)
        ax.plot(v.mean(), i, "o", ms=4.5, color=color, mec="white", mew=0.6)
    ax.set_yticks(range(len(ROWS)), [fs.CRITERIA[f][0] for f in ROWS] if show_names else [])
    ax.text(np.mean(rand), -0.75, "random", fontsize=6, color=fs.INK2, va="bottom", ha="center")
    ax.set_ylim(len(ROWS) - 0.4, -0.9)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    fs.style(ax)
    ax.spines["left"].set_visible(False)


def main():
    ctx = stage_from_argv("revision/figures/fig3")
    ctx.require_stage("revision/mechanism/04_leading_eigenvectors")
    ctx.require_stage("revision/mechanism/08_gramian_alignment")
    ctx.begin()
    fs.setup()
    mech = ctx.root / "revision" / "mechanism"
    subj = pd.read_csv(mech / "eigvec_subjects.csv")
    regions = pd.read_csv(mech / "eigvec_regions.csv")
    align = pd.read_csv(mech / "gramian_alignment.csv")
    nodes = pd.read_csv(mech / "node_v1_strength.csv")
    # rows ordered by mean L_S (descending)
    a = align[(align.scale == SCALE) & (align.k == K)]
    ROWS.sort(key=lambda f: -a[a.family == f]["loading"].mean())

    fig = plt.figure(figsize=(fs.FULL, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.45], hspace=0.38)
    panel_subjects(fig, gs[0], ctx.root, regions, subj)
    low = gs[1].subgridspec(1, 3, width_ratios=[1, 1.15, 0.85], wspace=0.12)
    ax = fig.add_subplot(low[0, 0])
    panel_strength(ax, nodes)
    fs.panel_label(ax, "B", x=-0.2, y=1.02)
    ax = fig.add_subplot(low[0, 1])
    dotplot(ax, align, "loading", "Loading on v1, $L_S$", show_names=True)
    fs.panel_label(ax, "C", x=-0.75, y=1.02)
    ax = fig.add_subplot(low[0, 2])
    dotplot(ax, align, "cos_top_v1", "cos(most reachable pattern, v1)", show_names=False)
    fs.panel_label(ax, "D", x=-0.05, y=1.02)
    # place C next to B, leaving room for C's row names
    pos = fig.axes[-2].get_position()
    fig.axes[-2].set_position([pos.x0 + 0.1, pos.y0, pos.width - 0.1, pos.height])

    for fam in ("degree_strength", "betweenness", "cycle_participation", "random"):
        for k in (2, 3, 5):
            g = align[(align.family == fam) & (align.k == k)].groupby(["scale", "subject_id"])["cos_top_v1"].mean().groupby("scale").mean()
            print(f"3D  {fam:22s} k={k}: " + "  ".join(f"{s}: {v:.2f}" for s, v in g.items()))
    paths = fs.save(fig, ctx.root, "fig3")
    ctx.finish(outputs=paths)


if __name__ == "__main__":
    main()
