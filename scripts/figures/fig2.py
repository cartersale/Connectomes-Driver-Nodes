#!/usr/bin/env python
"""Figure 2: combining hubs concentrates controllability onto a shared pattern.

A  Normalized Gramian spectra, lambda_i(W) / tr W (the fraction of total controllability along each of the Gramian's
   eigenvectors, largest first), for one representative subject: one hub vs one random region (top), three hubs vs
   three random regions (bottom). The first six eigenvalues are shown; the rest of the spectrum holds at most 0.2% of
   the total in every case (printed when the script runs), so the bars sum to one to within that. q1 is the first bar.
B  q1 against the number of drivers k (1, 2, 3, 5) for degree-strength, cycle-participation and random sets. Small
   dots are subjects (random: the mean of each subject's 100 random sets); large dots are means with 95% CIs, joined
   by thin lines to guide the eye (only these k were measured).
C  q1 of each criterion's set at k = 3: subjects (small) and means (large), with the random mean (line) and the
   5th-95th percentile of per-subject random means (band). Rows in the same order as Figure 3 (by mean loading on v1).

Representative subject: the one whose degree-strength q1 at k = 3 is closest to the cohort median. Random sets: of
that subject's 100 random sets at each k, the one whose q1 is closest to the subject's median random q1. Single hub:
the top-ranked degree-strength region. All panels: 114 regions, c = 0.1, infinite horizon.

Writes <run>/figures/fig2.{pdf,svg,png}. Run: python scripts/figures/fig2.py --run <name>
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from hoc import figstyle as fs  # noqa: E402
from hoc.analysis.null_models import Spectral, load_stack  # noqa: E402

SCALE, C, K_SHOW = 114, 0.1, 3
KS = (1, 2, 3, 5)
N_EIG = 6
FAMS_B = [("degree_strength", "Degree (strength)", fs.HUB), ("cycle_participation", "Cycle participation", fs.SPREAD),
          ("random", "Random", fs.RANDOM)]


def nodes_of(s) -> list[int]:
    return [int(t) for t in str(s).split(";")]


class Example:
    """One representative subject: its hub and random sets, and their Gramians in A0's eigenbasis."""

    def __init__(self, root: Path, sets: pd.DataFrame):
        x = sets[(sets.scale == SCALE) & (sets.c == C)]
        deg3 = x[(x.family == "degree_strength") & (x.k == K_SHOW)].set_index("subject_id").q1
        self.sid = (deg3 - deg3.median()).abs().idxmin()
        si = int(self.sid.split("-")[1])
        x = x[x.subject_id == self.sid]
        self.q1_table = x
        self.spec = Spectral(load_stack(SCALE, root)[si])
        self.kern = self.spec.kernel(C)
        self.hub1 = nodes_of(x[(x.family == "degree_strength") & (x.k == 1)].nodes.iloc[0])
        self.hubs = nodes_of(x[(x.family == "degree_strength") & (x.k == K_SHOW)].nodes.iloc[0])
        self.rand = {}
        for k in (1, K_SHOW):
            r = x[(x.family == "random") & (x.k == k)]
            self.rand[k] = nodes_of(r.iloc[(r.q1 - r.q1.median()).abs().argsort().iloc[0]].nodes)

    def spectrum(self, nodes: list[int]) -> np.ndarray:
        U = self.spec.U[nodes, :]
        with np.errstate(all="ignore"):  # spurious BLAS warnings on macOS; checked below
            g = self.kern * (U.T @ U)
        assert np.isfinite(g).all()
        w = np.clip(np.linalg.eigvalsh(g)[::-1], 0, None)
        return w / w.sum()


def panel_spectra(fig, cell, ex: Example) -> None:
    sub = cell.subgridspec(2, 2, hspace=0.55, wspace=0.22)
    rows = [("One region", ex.hub1, ex.rand[1]), ("Three regions", ex.hubs, ex.rand[K_SHOW])]
    xs = np.arange(1, N_EIG + 1)
    axes = []
    for i, (rname, hub, rnd) in enumerate(rows):
        for j, (nodes, color, cname) in enumerate(((hub, fs.HUB, "Hubs"), (rnd, fs.RANDOM, "Random"))):
            ax = fig.add_subplot(sub[i, j])
            w = ex.spectrum(nodes)
            ax.bar(xs, w[:N_EIG], width=0.7, color=color, lw=0)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(xs, [str(t) for t in xs], fontsize=6)
            ax.set_yticks([0, 0.5, 1], ["0", "", "1"] if j == 0 else [])
            ax.text(0.97, 0.95, f"$q_1$ = {w[0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6,
                    color=fs.INK2)
            fs.style(ax)
            if i == 0:
                ax.set_title(cname, fontsize=7, color=color if color != fs.RANDOM else fs.INK2, pad=3)
            if j == 0:
                ax.set_ylabel(f"{rname}\nfraction of total", fontsize=6.5)
            axes.append(ax)
    fig.text((axes[2].get_position().x0 + axes[3].get_position().x1) / 2, axes[2].get_position().y0 - 0.12,
             "Gramian eigenvector $i$ (largest first)", ha="center", va="top", fontsize=6.5,
             color=fs.INK)
    fs.panel_label(axes[0], "A", x=-0.45, y=1.12)


def panel_k(ax, sets: pd.DataFrame) -> None:
    x = sets[(sets.scale == SCALE) & (sets.c == C)]
    rng = np.random.default_rng(0)
    for off, (fam, name, color) in zip((-0.22, 0.0, 0.22), FAMS_B):
        means = []
        for k in KS:
            v = x[(x.family == fam) & (x.k == k)].groupby("subject_id").q1.mean().to_numpy()
            ax.scatter(k + off + rng.uniform(-0.06, 0.06, len(v)), v, s=2.5, color=color, alpha=0.3, lw=0)
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(k + off, m, yerr=1.96 * se, fmt="o", ms=4, color=color, mec="white", mew=0.6, elinewidth=1,
                        capsize=0, zorder=5)
            means.append(m)
        ax.plot(np.array(KS) + off, means, color=color, lw=0.8, zorder=4)
        ax.plot([], [], "o-", ms=4, lw=0.8, color=color, mec="white", mew=0.6, label=name)
    ax.set_xticks(KS)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False, fontsize=6, handletextpad=0.3, borderaxespad=0.2)
    ax.set_xlabel("Number of drivers $k$")
    ax.set_ylabel(r"$q_1 = \lambda_{\max}(W)/\mathrm{tr}\,W$")
    fs.style(ax)
    fs.panel_label(ax, "B", x=-0.18, y=1.03)


def panel_criteria(ax, align: pd.DataFrame) -> None:
    a = align[(align.scale == SCALE) & (align.k == K_SHOW)]
    rows = sorted(fs.CRITERION_ROWS, key=lambda f: -a[a.family == f]["loading"].mean())
    rand = a[a.family == "random"].groupby("subject_id").q1.mean().to_numpy()
    lo, hi = np.percentile(rand, [5, 95])
    top = -0.55  # the line starts just under its label
    ax.fill_betweenx([top, len(rows) - 0.4], lo, hi, color=fs.RANDOM, alpha=0.45, lw=0)
    ax.plot([rand.mean()] * 2, [top, len(rows) - 0.4], color=fs.INK2, lw=0.9)
    ax.text(rand.mean(), top - 0.05, "random", fontsize=6, color=fs.INK2, va="bottom", ha="center")
    for i, fam in enumerate(rows):
        v = a[a.family == fam].q1.to_numpy()
        name, color = fs.CRITERIA[fam]
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.2, 0.2, len(v)), s=3, color=color,
                   alpha=0.35, lw=0)
        ax.plot(v.mean(), i, "o", ms=4.5, color=color, mec="white", mew=0.6)
    ax.set_yticks(range(len(rows)), [fs.CRITERIA[f][0] for f in rows], fontsize=6.5)
    ax.set_ylim(len(rows) - 0.4, -1.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"$q_1$ ($k$ = {K_SHOW})")
    ax.tick_params(axis="y", length=0)
    fs.style(ax)
    ax.spines["left"].set_visible(False)
    fs.panel_label(ax, "C", x=-0.62, y=1.03)
    print(f"C  random mean {rand.mean():.3f}, band {lo:.3f}-{hi:.3f}")


def main():
    ctx, root = fs.open_run("figures/fig2", ("concentration/01_compute_q1",
                                                                     "mechanisms/08_gramian_alignment"))
    fs.setup()
    sets = pd.read_csv((root / "concentration") / "set_q1.csv.gz")
    align = pd.read_csv((root / "mechanisms") / "gramian_alignment.csv")
    ex = Example(root, sets)
    print(f"representative subject {ex.sid}; hub1 {ex.hub1}; hubs {ex.hubs}; random k=1 {ex.rand[1]}, k=3 {ex.rand[K_SHOW]}")
    for name, n in (("hub1", ex.hub1), ("hubs", ex.hubs), ("rand1", ex.rand[1]), ("rand3", ex.rand[K_SHOW])):
        w = ex.spectrum(n)
        print(f"  {name}: q1 {w[0]:.3f}, top six {np.round(w[:N_EIG], 3)}, rest {w[N_EIG:].sum():.4f}")

    fig = plt.figure(figsize=(fs.FULL, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.45, 0.45, 1.0, 1.2, 0.95], wspace=0.0)
    panel_spectra(fig, gs[0], ex)
    panel_k(fig.add_subplot(gs[2]), sets)
    panel_criteria(fig.add_subplot(gs[4]), align)
    fs.finish(fig, ctx, "fig2")


if __name__ == "__main__":
    main()
