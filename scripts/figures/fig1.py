#!/usr/bin/env python
"""Figure 1: where inputs enter shapes what they can reach (orienting figure).

A  The control model on the real network: the cohort-reference connectome seen from above, node size = strength,
   strongest 5% of links drawn. Inputs u(t) enter the 3 strongest regions (hub set).
B  The leading eigenvector v1 on the cortex (lateral and medial, both hemispheres), with the hub set marked.
C  Loading on v1 and q1 (share of controllability on the most reachable pattern) for the hub set against 1000
   random 3-region sets (grey points; black line = median).

Data: cohort-reference connectome at 114 regions, c = 0.1, k = 3, infinite horizon.
Run: python scripts/figures/fig1.py --run <name>
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

from hoc import figstyle as fs  # noqa: E402
from hoc.rankings import strength_scores  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402

SCALE, C, K = 114, 0.1, 3
N_RANDOM = 1000
EDGE_QUANTILE = 0.95  # links drawn in A
NODE = "#c9c7c0"


def mark(ax, xy, hub: bool = True, size: float = 44) -> None:
    ax.scatter(*xy, s=size, color=fs.HUB, edgecolor="white", lw=0.9, zorder=6)


def node_xy(geo: dict) -> np.ndarray:
    """Top-down position (left-right, back-front) of each region: mean of its pial vertices."""
    xy = np.zeros((geo["n_nodes"], 2))
    for hemi in ("L", "R"):
        g = geo[hemi]
        for n in np.unique(g["v2n"][g["v2n"] >= 0]):
            xy[n] = g["coords"][g["v2n"] == n][:, :2].mean(axis=0)
    return xy


def panel_network(ax, geo, a0, hub) -> None:
    # brain silhouette from above
    for hemi in ("L", "R"):
        g = geo[hemi]
        tri = g["coords"][g["faces"]][:, :, :2]
        ax.add_collection(PolyCollection(tri, facecolors="#f1f0ec", edgecolors="none", rasterized=True, zorder=0))
    xy = node_xy(geo)
    s = strength_scores(a0)
    iu = np.triu_indices(len(a0), 1)
    w = a0[iu]
    keep = w > np.quantile(w[w > 0], EDGE_QUANTILE)
    segs = np.stack([xy[iu[0][keep]], xy[iu[1][keep]]], axis=1)
    lw = 0.25 + 1.1 * (w[keep] / w[keep].max())
    ax.add_collection(LineCollection(segs, colors=fs.MUTED, linewidths=lw, alpha=0.45, zorder=1))
    size = 3 + 40 * (s / s.max()) ** 2
    drivers = set(hub)
    others = [i for i in range(len(a0)) if i not in drivers]
    ax.scatter(xy[others, 0], xy[others, 1], s=size[others], color=NODE, edgecolor="white", lw=0.4, zorder=2)
    for d in hub:
        mark(ax, xy[d], True, size=size[d])
    # inputs u(t): a wave and an arrow pointing in from outside each hemisphere
    for d in drivers:
        p = xy[d]
        u = np.array([0.0, 1.0]) if abs(p[0]) < 12 else np.array([np.sign(p[0]), 0.0])  # midline regions: from the front
        start = p + 17 * u
        ax.annotate("", xy=p + 5.5 * u, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=0.7, mutation_scale=5), zorder=5)
        t = np.linspace(0, 1, 60)
        nrm = np.array([-u[1], u[0]])
        wave = start[None] + 13 * t[:, None] * u[None] + 2.6 * np.sin(4 * np.pi * t)[:, None] * nrm
        ax.plot(*wave.T, color=fs.INK, lw=0.8, zorder=5)
    ax.text(0, 75, "front", ha="center", va="bottom", fontsize=6, color=fs.INK2)
    ax.text(-74, -104, "L", ha="center", va="bottom", fontsize=6, color=fs.INK2)
    ax.text(74, -104, "R", ha="center", va="bottom", fontsize=6, color=fs.INK2)
    ax.text(0.5, -0.04, r"$\dot{x} = Ax + Bu$", transform=ax.transAxes, ha="center", va="top", color=fs.INK, fontsize=8)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-108, 80)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_brain(fig, cell, geo, strength, v1, hub) -> dict:
    """Two rows over the same four views: each region's strength (the degree score) and its v1 value.
    The hub drivers (top 3 by strength) are outlined in both rows."""
    views = (("L_lateral", "Left lateral"), ("L_medial", "Left medial"), ("R_medial", "Right medial"), ("R_lateral", "Right lateral"))
    rows = (("Strength", strength), ("v1", v1))
    sub = cell.subgridspec(2, 4, wspace=0.03, hspace=-0.25)
    outline = np.zeros(len(v1), bool)
    outline[hub] = True
    axes = {}
    for r, (name, vals) in enumerate(rows):
        # strength: weakest to strongest region; v1: from 0, compressed so the low tail stays visible
        norm = (mpl.colors.Normalize(float(vals.min()), float(vals.max())) if name == "Strength"
                else mpl.colors.PowerNorm(0.6, 0, float(vals.max())))
        for c, (view, title) in enumerate(views):
            ax = fig.add_subplot(sub[r, c])
            fs.draw_view(ax, geo, view, vals, fs.SEQ, norm, outline=outline, outline_color=fs.HUB, outline_width=0.9)
            axes[(r, c)] = ax
        axes[(r, 0)].text(-0.04, 0.5, name, transform=axes[(r, 0)].transAxes, rotation=90, ha="right", va="center",
                          fontsize=6.5, color=fs.INK)
    # hemisphere labels over each lateral-medial pair (the views themselves show lateral vs medial)
    for c, hemi in ((0, "Left"), (2, "Right")):
        axes[(0, c)].text(1.02, 1.04, hemi, transform=axes[(0, c)].transAxes, ha="center", va="bottom", fontsize=6.5, color=fs.INK2)
    # one shared scale: each row runs from its weakest to its strongest region
    cax = axes[(0, 3)].inset_axes([1.05, -0.62, 0.05, 1.25])
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=fs.SEQ), cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([])
    cax.text(0.5, 1.03, "high", transform=cax.transAxes, ha="center", va="bottom", fontsize=5.5, color=fs.INK2)
    cax.text(0.5, -0.03, "low", transform=cax.transAxes, ha="center", va="top", fontsize=5.5, color=fs.INK2)
    return axes


def panel_strips(ax, vals: dict, rng) -> None:
    """Hub set (orange dot) against the random sets (grey strip, median line)."""
    for g, (name, (h, r)) in enumerate(vals.items()):
        ax.scatter(g - 0.17, h, s=28, color=fs.HUB, edgecolor="white", lw=0.6, zorder=4)
        x = g + 0.17 + rng.uniform(-0.1, 0.1, len(r))
        ax.scatter(x, r, s=1.5, color=fs.RANDOM, lw=0, alpha=0.5, rasterized=True, zorder=2)
        ax.plot([g + 0.03, g + 0.31], [np.median(r)] * 2, color=fs.INK, lw=0.9, zorder=3)
    ax.set_xticks([0, 1], ["Loading on v1", "q1"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="x", length=0)
    fs.style(ax)


def set_values(a0, sets) -> tuple[np.ndarray, np.ndarray]:
    lam, u = np.linalg.eigh(a0)
    v1 = u[:, -1]
    mu = lam - lam[-1] - C
    kernel = -1.0 / (mu[:, None] + mu[None, :])

    def q1(s):
        w = np.linalg.eigvalsh(kernel * (u[s].T @ u[s]))
        return float(w[-1] / np.clip(w, 0, None).sum())

    return np.array([np.sum(v1[s] ** 2) for s in sets]), np.array([q1(s) for s in sets])


def main():
    ctx = stage_from_argv("figures/fig1")
    ctx.require_stage("primary/00_prepare_data")
    ctx.begin()
    fs.setup()
    a0 = np.asarray(np.load(ctx.root / "processed" / f"connectomes_scale-{SCALE}.npz")["cohort_reference"], float)
    labels = pd.read_csv(ctx.root / "processed" / f"atlas_labels_scale-{SCALE}.csv").sort_values("node_index")["region_label"].tolist()
    _, v1 = fs.leading_eigvec(a0)
    hub = list(np.argsort(-strength_scores(a0))[:K])
    rng = np.random.default_rng(0)
    rand_sets = [rng.choice(len(a0), K, replace=False) for _ in range(N_RANDOM)]
    hub_load, hub_q1 = set_values(a0, [hub])
    r_load, r_q1 = set_values(a0, rand_sets)
    geo = fs.surface(SCALE, labels)

    fig = plt.figure(figsize=(fs.FULL, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 3.1, 0.9], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    panel_network(ax, geo, a0, hub)
    fs.panel_label(ax, "A", x=0.02, y=1.0)
    axes = panel_brain(fig, gs[1], geo, strength_scores(a0), v1, hub)
    fs.panel_label(axes[(0, 0)], "B", x=-0.12, y=1.08)
    ax = fig.add_subplot(gs[2])
    vals = {"loading": (hub_load[0], r_load), "q1": (hub_q1[0], r_q1)}
    panel_strips(ax, vals, np.random.default_rng(1))
    fs.panel_label(ax, "C", x=-0.25, y=1.02)
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=fs.HUB, mec="white", label="3 strongest regions (hub drivers; outlined in B)"),
               Line2D([], [], ls="", marker="o", ms=3, color=fs.RANDOM, label="1000 random sets of 3")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=6.5, handletextpad=0.3, columnspacing=2)

    for k, (h, r) in vals.items():
        print(f"{k}: hub {h:.2f}  random median {np.median(r):.2f} (5-95% {np.quantile(r, 0.05):.2f}-{np.quantile(r, 0.95):.2f}), "
              f"hub above {100 * np.mean(r < h):.1f}% of random sets")
    print("hub:", [labels[i] for i in hub])
    paths = fs.save(fig, ctx.root, "fig1")
    ctx.finish(outputs=paths)


if __name__ == "__main__":
    main()
