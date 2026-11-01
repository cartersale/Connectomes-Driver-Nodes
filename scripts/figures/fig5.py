#!/usr/bin/env python
"""Figure 5: strong links among core regions give human connectomes a large spectral gap.

A  Spectral gap l1 - l2 by scale: human connectomes and the three null models (strength-preserving,
   geometry-preserving, weight-shuffled). Small dots: subjects (nulls: mean of each subject's 50 nulls). Large dots:
   means over subjects.
B  q1 of degree-strength sets (mean over k = 2, 3, 5; c = 0.1) by scale: human, nulls at the same c, and nulls at a
   c matched to the human decay ratio. Nulls: mean over the three null models for each subject.
C  Per subject, the excess of the human connectome over the mean of its 50 strength-preserving nulls in edge-weighted
   strength assortativity (x) and in spectral gap (y), coloured by scale, with Spearman rho per scale.
D  Left: the core's front and back halves (anatomical labels) and the links between them on the medial surface
   (cohort-mean weights, 114 regions). Right, top: % of the gap closed by halving the front-back links, against the
   same weight taken from within the halves or from outside the core (matched edge by edge on weight and end
   strength; mean over 20 draws), one dot per subject, large dots = medians; values beyond the axis are drawn at its
   edge. Open diamonds: the same front-back cut with halves defined by the sign of each subject's v2 (a reference;
   chosen using v2). Right, bottom: the change in l1 and l2 for the same 50% cuts (label halves; medians).

Reads null_models/{sets.csv.gz, rayleigh/networks.csv.gz, rayleigh/per_subject.csv} and mechanisms/{core_cut.csv.gz,
core_cut_regions.csv}. Run: python scripts/figures/fig5.py --run <name>
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from hoc import figstyle as fs  # noqa: E402

ANT, POST = "#b8312f", "#184f95"  # diverging poles: front (red) and back (blue), as in Fig 4A
NETS = {  # network: (label, colour, marker)
    "real": ("Human", "scale", "o"),
    "strength": ("Strength-preserving", "#5f5e59", "s"),
    "geometric": ("Geometry-preserving", "#85847e", "D"),
    "weights": ("Weight-shuffled", "#a9a8a2", "^"),
}
Q1 = {  # condition: (label, colour)
    "human": ("Human", "scale"),
    "fixed": ("Null, same c", "#a9a8a2"),
    "gap": ("Null, gap-matched c", "scale"),  # drawn hollow
}
COND = {  # core-cut condition: (label, colour)
    "cross": ("Front-back links", fs.HUB),
    "within_matched": ("Within-half links", "#8a8984"),
    "other_matched": ("Outside-core links", "#cfcec9"),
}
EDGE_SCALE = 114
ALPHA = 0.5


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
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Spectral gap  ℓ1 − ℓ2")
    ax.set_xlabel("Regions")
    ax.legend(handles=handles([(l, c, m) for l, c, m in NETS.values()]), loc="upper center", ncol=2, fontsize=6,
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
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("$q_1$ of degree-strength sets")
    ax.set_xlabel("Regions")
    hs = handles([(Q1["human"][0], "scale", "o"), (Q1["fixed"][0], Q1["fixed"][1], "o")])
    hs.append(mpl.lines.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=fs.SCALE_COLORS[114], mew=0.9,
                               label=Q1["gap"][0]))
    ax.legend(handles=hs, loc="upper center", ncol=3, fontsize=6,
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


def cut_table(cut: pd.DataFrame) -> pd.DataFrame:
    d = cut[(cut.alpha == ALPHA) & cut.condition.isin(list(COND))]
    d = d[(d.partition == "labels") | ((d.partition == "v2sign") & (d.condition == "cross"))]
    d = d.assign(condition=np.where(d.partition == "v2sign", "v2_cross", d.condition))
    d = d.assign(closed=-100 * d.dgap / (d.gap - d.dgap))
    return d.groupby(["condition", "scale", "subject_id"])[["closed", "dl1", "dl2"]].mean()


def panel_cut(fig, cell, cut: pd.DataFrame) -> list:
    t = cut_table(cut)
    sub = cell.subgridspec(2, 1, height_ratios=[1.15, 1], hspace=0.55)
    ax = fig.add_subplot(sub[0])
    groups = [({sc: t.loc[(c, sc), "closed"].to_numpy() for sc in fs.SCALES}, col, "o") for c, (_, col) in COND.items()]
    groups.append(({sc: t.loc[("v2_cross", sc), "closed"].to_numpy() for sc in fs.SCALES}, fs.HUB, "D", True))
    lim = (-30, 80)
    strips(ax, groups, stat=np.median, ylim=lim)
    ax.set_ylim(lim[0] - 3, lim[1] + 3)
    n_out = int(((t.closed < lim[0]) | (t.closed > lim[1])).sum())
    print(f"5D  gap closed: {n_out} of {len(t)} subject values beyond {lim} (drawn at the edge)")
    ax.axhline(0, color=fs.GRID, lw=0.6, zorder=0)
    ax.set_ylabel("Gap closed (%)")
    hs = handles([(l, c, "o") for l, c in COND.values()])
    hs.append(mpl.lines.Line2D([], [], ls="", marker="D", ms=3.5, mfc="white", mec=fs.HUB, mew=0.9,
                               label="Front-back links, $v_2$ halves"))
    ax.legend(handles=hs, loc="lower left", fontsize=6, ncol=2, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.0, bbox_to_anchor=(-0.05, 1.04))
    axes = [ax]
    low = sub[1].subgridspec(1, 2, wspace=0.7)
    for j, (col, lab) in enumerate((("dl1", "Δℓ1"), ("dl2", "Δℓ2"))):
        ax = fig.add_subplot(low[0, j])
        groups = [({sc: t.loc[(c, sc), col].to_numpy() for sc in fs.SCALES}, cc, "o") for c, (_, cc) in COND.items()]
        strips(ax, groups, ms=3.5, dot=1.5, stat=np.median)
        ax.axhline(0, color=fs.MUTED, lw=0.6, zorder=0)
        ax.set_ylabel(lab, labelpad=1)
        ax.set_xlabel("Regions")
        axes.append(ax)
    for sc in fs.SCALES:
        print(f"5D  {sc}: median % closed " + ", ".join(f"{c} {np.median(t.loc[(c, sc), 'closed']):.0f}" for c in (*COND, "v2_cross"))
              + "; median dl1 " + ", ".join(f"{c} {np.median(t.loc[(c, sc), 'dl1']):+.3f}" for c in COND)
              + "; median dl2 " + ", ".join(f"{c} {np.median(t.loc[(c, sc), 'dl2']):+.3f}" for c in COND))
    return axes


def panel_links(fig, cell, root: Path, regions: pd.DataFrame) -> list:
    reg = regions[regions.scale == EDGE_SCALE].sort_values("node_index")
    labels = pd.read_csv(root / "processed" / f"atlas_labels_scale-{EDGE_SCALE}.csv").sort_values("node_index")["region_label"].tolist()
    geo = fs.surface(EDGE_SCALE, labels)
    a = np.asarray(np.load(root / "processed" / f"connectomes_scale-{EDGE_SCALE}.npz")["adjacency"], float).mean(axis=0)
    side = reg["labels"].to_numpy()
    rgb = np.tile(mpl.colors.to_rgb("#ecebe7"), (len(side), 1))
    rgb[side == "anterior"] = mpl.colors.to_rgb("#f0b8b5")
    rgb[side == "posterior"] = mpl.colors.to_rgb("#b5cbe8")
    hemi = reg["hemi"].to_numpy()
    wmax = max(a[i, j] for i in np.where(side == "anterior")[0] for j in np.where(side == "posterior")[0])
    sub = cell.subgridspec(2, 1, hspace=0.12)
    axes = []
    for k, (view, h) in enumerate((("L_medial", "L"), ("R_medial", "R"))):
        ax = fig.add_subplot(sub[k, 0])
        fs.draw_view(ax, geo, view, np.zeros(len(side)), None, None, face_rgb=rgb)
        ant = [i for i in np.where((side == "anterior") & (hemi == h))[0]]
        post = [j for j in np.where((side == "posterior") & (hemi == h))[0]]
        pos = {n: fs.centroid(geo, view, n) for n in ant + post}
        for i in ant:
            for j in post:
                w = a[i, j]
                if w <= 0:
                    continue
                p, q = pos[i], pos[j]
                mid = (p + q) / 2 + np.array([0, 0.18 * np.linalg.norm(q - p)])  # bow the link upward
                t = np.linspace(0, 1, 30)[:, None]
                curve = (1 - t) ** 2 * p + 2 * (1 - t) * t * mid + t**2 * q
                ax.plot(*curve.T, color=fs.HUB, lw=0.2 + 2.2 * w / wmax, alpha=0.35 + 0.5 * w / wmax, solid_capstyle="round", zorder=4)
        for n in ant + post:
            ax.plot(*pos[n], "o", ms=3.5, color=ANT if side[n] == "anterior" else POST, mec="white", mew=0.6, zorder=5)
        ax.set_title("Left medial" if h == "L" else "Right medial", fontsize=6.5, color=fs.INK2, pad=1)
        axes.append(ax)
    hs = [mpl.lines.Line2D([], [], ls="", marker="o", ms=4, color=ANT, mec="white", label="Front half"),
          mpl.lines.Line2D([], [], ls="", marker="o", ms=4, color=POST, mec="white", label="Back half")]
    axes[1].legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6,
                   handletextpad=0.1, columnspacing=0.8, borderaxespad=0.0)
    return axes


def main():
    ctx, root = fs.open_run("figures/fig5", ("null_models/02_analyze", "null_models/04_rayleigh_bound",
                                                             "mechanisms/07_core_cut"))
    fs.setup()
    nulls = (root / "null_models")
    mech = (root / "mechanisms")
    sets = pd.read_csv(nulls / "sets.csv.gz", usecols=["null", "scale", "subject_id", "c_mode", "c_ref", "k", "degree_q1"],
                       dtype={"k": str})
    ray = pd.read_csv(nulls / "rayleigh" / "networks.csv.gz", usecols=["null", "scale", "subject", "gap", "r_W"])
    per_subject = pd.read_csv(nulls / "rayleigh" / "per_subject.csv")
    cut = pd.read_csv(mech / "core_cut.csv.gz")
    regions = pd.read_csv(mech / "core_cut_regions.csv")

    fig = plt.figure(figsize=(fs.FULL, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.38)
    top = gs[0].subgridspec(1, 2, wspace=0.28)
    ax = fig.add_subplot(top[0, 0])
    panel_gap(ax, ray)
    fs.panel_label(ax, "A", x=-0.12, y=1.02)
    ax = fig.add_subplot(top[0, 1])
    panel_q1(ax, sets)
    fs.panel_label(ax, "B", x=-0.12, y=1.02)
    bottom = gs[1].subgridspec(1, 3, width_ratios=[1.0, 0.62, 1.05], wspace=0.3)
    ax = fig.add_subplot(bottom[0, 0])
    panel_assort(ax, per_subject)
    fs.panel_label(ax, "C", x=-0.2, y=1.02)
    brains = panel_links(fig, bottom[0, 1], root, regions)
    fs.panel_label(brains[0], "D", x=0.02, y=1.1)
    panel_cut(fig, bottom[0, 2], cut)
    fs.finish(fig, ctx, "fig5")


if __name__ == "__main__":
    main()
