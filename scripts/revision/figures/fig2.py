#!/usr/bin/env python
"""Former Figure 2 (now written as figS_energy): energy cannot tell driver sets apart; combining hubs concentrates
control. The main-text Figure 2 is drawn by fig2_combine.py.

A  Average control energy against its floor N/eps: median log10 E_avg over sets for each subject, k and eps, coloured
   by scale (c = 0.1). The dashed line is E_avg = N/eps.
B  One subject (median gap, 114 regions, k = 3, c = 0.1): q1 of its 100 random sets (histogram) and of each
   criterion's set.
C  Mean within-subject percentile of each criterion's q1 among the random sets, by scale and k (c = 0.1). Rows are
   ordered by mean loading on v1.
D  Excess q1 of degree sets over random sets, by k: a single hub concentrates less than a random region, two or more
   concentrate more.

Reads revision/regime/task6_landscape.csv and revision/q1/{set_q1.csv.gz, p1_vs_random.csv}.
Run: python scripts/revision/figures/fig2.py --run <name> [--draft]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

SCALE, K, C = 114, 3, 0.1
LABELLED = ("degree_strength", "betweenness", "cycle_participation")


def panel_energy(ax, land: pd.DataFrame) -> None:
    lims = [land.log10_N_over_eps.min() - 0.6, land.log10_N_over_eps.max() + 0.6]
    ax.plot(lims, lims, color=fs.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    rng = np.random.default_rng(0)
    for sc in fs.SCALES:
        d = land[land.scale == sc]
        ax.scatter(d.log10_N_over_eps + rng.uniform(-0.15, 0.15, len(d)), d.median_log10E, s=3, color=fs.SCALE_COLORS[sc],
                   lw=0, alpha=0.6, label=str(sc), rasterized=True)
    dev = (land.median_log10E - land.log10_N_over_eps).abs()
    ax.text(0.03, 0.97, f"all within {dev.max():.2f}\nof log N/ε", transform=ax.transAxes, fontsize=6, color=fs.INK2, va="top")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("log$_{10}$ N/ε")
    ax.set_ylabel("log$_{10}$ average energy")
    ax.legend(title="Regions", loc="lower right", title_fontsize=6, handletextpad=0.1, borderaxespad=0.1, markerscale=2)
    fs.style(ax)
    for eps, g in land.groupby("eps"):
        d = (g.median_log10E - g.log10_N_over_eps).abs()
        print(f"2A  eps = {eps:g}: |log10 E - log10 N/eps| max {d.max():.3f}, median {d.median():.3f}; "
              f"share of eigenvalues below eps {g.mean_fraction_below_eps.min():.2f}-{g.mean_fraction_below_eps.max():.2f}")


def panel_subject(ax, sq: pd.DataFrame, sid: str) -> None:
    d = sq[(sq.subject_id == sid) & (sq.scale == SCALE) & np.isclose(sq.c, C) & (sq.k == K)]
    rand = d[d.family == "random"].q1.to_numpy()
    ax.hist(rand, bins=np.linspace(0, 1, 41), color=fs.RANDOM, alpha=0.6, lw=0)
    top = ax.get_ylim()[1]
    crit = d[d.family.isin(fs.CRITERIA)].drop_duplicates("family")
    for _, row in crit.iterrows():
        name, color = fs.CRITERIA[row.family]
        lab = row.family in LABELLED
        ax.plot([row.q1] * 2, [top * 1.02, top * (1.17 if lab else 1.1)], color=color if lab else fs.MUTED, lw=1.4 if lab else 0.8)
        if lab:
            ax.text(row.q1, top * 1.19, name.split(" (")[0], fontsize=6, color=fs.INK2, ha="center", va="bottom", rotation=0)
    ax.text(np.median(rand), top * 0.5, "random\nsets", fontsize=6, color=fs.INK2, ha="center")
    ax.set_ylim(0, top * 1.35)
    ax.set_xlim(0.25, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("q1 (share on the most reachable pattern)")
    fs.style(ax)
    ax.spines["left"].set_visible(False)
    print(f"2B  {sid}: random median {np.median(rand):.2f}; " + ", ".join(f"{f} {crit.set_index('family').q1[f]:.2f}" for f in LABELLED))


def panel_heatmap(ax, p1: pd.DataFrame, order: list[str]) -> None:
    d = p1[(p1.metric == "q1") & np.isclose(p1.c, C) & p1.k.isin(["1", "2", "3", "5"])]
    cols = [(sc, k) for sc in fs.SCALES for k in ("1", "2", "3", "5")]
    M = np.full((len(order), len(cols)), np.nan)
    for i, fam in enumerate(order):
        for j, (sc, k) in enumerate(cols):
            v = d[(d.family == fam) & (d.scale == sc) & (d.k == k)].pct_mean
            if len(v):
                M[i, j] = 100 * v.iloc[0]
    im = ax.imshow(M, cmap=fs.DIV, norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100), aspect="auto")
    ax.set_yticks(range(len(order)), [fs.CRITERIA[f][0] for f in order], fontsize=6)
    ax.set_xticks(range(len(cols)), [k for _, k in cols], fontsize=6)
    for j in (3.5, 7.5):
        ax.axvline(j, color="white", lw=2)
    for g, sc in enumerate(fs.SCALES):
        ax.text(1.5 + 4 * g, -0.9, f"{sc} regions", ha="center", va="bottom", fontsize=6, color=fs.INK2)
    ax.set_xlabel("Drivers k")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = ax.inset_axes([-0.5, -0.17, 0.28, 0.035])  # under the row names
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, 50, 100])
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cax.set_title("Percentile vs random sets", fontsize=6, color=fs.INK2, pad=2)


def panel_excess(ax, p1: pd.DataFrame) -> None:
    ks = ["1", "2", "3", "5"]
    d = p1[(p1.family == "degree_strength") & (p1.metric == "q1") & np.isclose(p1.c, C) & p1.k.isin(ks)]
    for sc in fs.SCALES:
        g = d[d.scale == sc].set_index("k").loc[ks]
        ex = g.value_mean - g.random_mean
        ax.plot([int(k) for k in ks], ex, color=fs.SCALE_COLORS[sc], marker=fs.SCALE_MARKERS[sc], ms=3.5, lw=1.2, label=str(sc))
        print(f"2D  degree excess q1 {sc}: " + ", ".join(f"k={k}: {v:+.2f}" for k, v in ex.items()))
    ax.axhline(0, color=fs.MUTED, lw=0.6, ls=":")
    ax.text(5, 0.01, "random", fontsize=6, color=fs.INK2, ha="right", va="bottom")
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xlabel("Drivers k")
    ax.set_ylabel("Excess q1, degree − random")
    ax.legend(title="Regions", loc="lower right", bbox_to_anchor=(1, 0.3), title_fontsize=6, handlelength=1.4, borderaxespad=0.1)
    fs.style(ax)


def main():
    ctx, root, draft = fs.open_run("revision/figures/fig2", ("revision/regime/04_energy_floor", "revision/q1/02_tables"))
    fs.setup()
    land = pd.read_csv(fs.rev_dir(root, "regime", draft) / "task6_landscape.csv")
    q1 = fs.rev_dir(root, "q1", draft)
    sq = pd.read_csv(q1 / "set_q1.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "q1", "loading"])
    p1 = pd.read_csv(q1 / "p1_vs_random.csv", dtype={"k": str})
    # one illustrative subject: median spectral gap at the primary scale
    subj = pd.read_csv(fs.rev_dir(root, "mechanism", draft) / "eigvec_subjects.csv")
    s = subj[subj.scale == SCALE].sort_values("gap").reset_index(drop=True)
    sid = s.subject_id.iloc[len(s) // 2]
    # criteria ordered by mean loading on v1 (all scales and k, c = 0.1)
    named = sq[np.isclose(sq.c, C) & sq.family.isin(fs.CRITERIA)]
    order = named.groupby("family").loading.mean().sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=(fs.FULL, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.6, 1], height_ratios=[1, 1.15], hspace=0.55, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    panel_energy(ax, land)
    fs.panel_label(ax, "A", x=-0.3, y=1.02)
    ax = fig.add_subplot(gs[0, 1:])
    panel_subject(ax, sq, sid)
    fs.panel_label(ax, "B", x=-0.06, y=1.02)
    ax = fig.add_subplot(gs[1, :2])
    panel_heatmap(ax, p1, order)
    fs.panel_label(ax, "C", x=-0.42, y=1.08)
    ax = fig.add_subplot(gs[1, 2])
    panel_excess(ax, p1)
    fs.panel_label(ax, "D", x=-0.3, y=1.02)
    fs.finish(fig, ctx, root, "figS_energy", draft)  # was Figure 2; now a supplementary candidate


if __name__ == "__main__":
    main()
