#!/usr/bin/env python
"""Figure 3B on its own: strength vs neighbours' loading m_i.

Drawn separately so it can replace panel B in the hand-assembled Figure 3. m_i is the strength-weighted mean v1 loading of region
i's neighbours. Grey dots: every region in every subject (114 regions; log axes; the one zero-strength region is
dropped). Black line: median m_i within strength deciles. Text: median within-subject Spearman rho.

Writes <run>/figures/fig3b.{pdf,svg,png}. Run: python scripts/figures/fig3b.py --run <name>
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hoc import figstyle as fs  # noqa: E402

SCALE = 114
AX_W, AX_H = 1.67, 1.65  # inches: the size of panel B's axes in the full Figure 3


def panel(ax, nodes: pd.DataFrame) -> None:
    d = nodes[(nodes.strength > 0) & (nodes.m > 0)]
    rho = d.groupby(["scale", "subject_id"])[["strength", "m"]].apply(lambda g: g.strength.corr(g.m, method="spearman"))
    med = rho.groupby("scale").median()
    d = d[d.scale == SCALE]
    ax.scatter(d["strength"], d["m"], s=2.5, lw=0, alpha=0.35, color=fs.RANDOM, rasterized=True)
    bins = np.quantile(d["strength"], np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(d["strength"], bins[1:-1]), 0, 9)
    g = d.groupby(idx)
    ax.plot(g["strength"].median(), g["m"].median(), color=fs.INK, lw=1.4, marker="o", ms=2.5, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlabel("Strength $s_i$")
    ax.set_ylabel("Mean $v_1$ of neighbors")
    fs.style(ax)
    ax.text(0.04, 0.96, f"median within-subject\nSpearman $\\rho$ = {med[SCALE]:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color=fs.INK2)
    print(f"{len(d)} regions plotted; median within-subject rho(s, m): "
          + "  ".join(f"{s}: {v:.2f}" for s, v in med.items()) + f"; positive in {int((rho > 0).sum())}/{len(rho)}")


def main():
    ctx, root = fs.open_run("figures/fig3b", ("mechanisms/08_gramian_alignment",))
    fs.setup()
    nodes = pd.read_csv(root / "mechanisms" / "node_v1_strength.csv")
    fig = plt.figure(figsize=(AX_W + 0.7, AX_H + 0.6))
    ax = fig.add_axes([0.55 / (AX_W + 0.7), 0.45 / (AX_H + 0.6), AX_W / (AX_W + 0.7), AX_H / (AX_H + 0.6)])
    panel(ax, nodes)
    fs.finish(fig, ctx, "fig3b")


if __name__ == "__main__":
    main()
