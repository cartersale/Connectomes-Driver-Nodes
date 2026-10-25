#!/usr/bin/env python
"""Figure: anatomy of the leading eigenvector and of the second-mode directions, plus how stable v2 is.

Rows A-C: cortical surface maps (fsaverage5 pial, Cammoun annotation; left lateral, left medial,
right medial, right lateral) of
  A  v1, group mean (sequential)
  B  consensus axis a of the v2-v3 plane (05_second_subspace; diverging, + = frontal side)
  C  consensus axis b of the v2-v3 plane (diverging)
B and C outline the top 20% of regions by group-mean v1 (the regions panel A highlights): both
second-mode patterns split that same set. Red and blue regions move in opposite directions; the
overall sign of an eigenvector is arbitrary.
Row D: eigenvalue ladder l1..l6 (every subject + median); which of v2..v4+ the axis a is in each
subject (all scales); cos^2(a, v2) against l2 - l3 (is v2/v3 swapping driven by small spacing?).

Reads runs/<name>/revision/mechanism/second_subspace_{axes,subjects}.csv. Writes
runs/<name>/figures/revision/mechanism/fig_eigvecs_scale-{68,114,219}.{png,pdf}
(choose the run with ``--run <name>``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from nilearn import datasets
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION  # noqa: E402
from hoc.runs import figures_dir, stage_from_argv  # noqa: E402

SCALES = (68, 114, 219)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
WALL = "#cfcec9"  # medial wall / unlabelled vertices
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#5598e7", "#f0efec", "#ec7b7a", "#b8312f"])
CAT = ("#2a78d6", "#eb6834", "#1baf7a")  # v2, v3, v4+ (validated; direct-labelled)
# (hemi, view, viewer direction along x, horizontal image axis = sign * y)
VIEWS = (("L", "lateral", -1, -1), ("L", "medial", +1, +1), ("R", "medial", -1, -1), ("R", "lateral", +1, +1))


def surface(atlas_dir: Path, scale: int, labels: list[str]) -> dict:
    fsavg = datasets.load_fsaverage(mesh="fsaverage5")
    geo, names = {}, []
    node = 0
    for hemi, part in (("L", "left"), ("R", "right")):
        mesh = fsavg["pial"].parts[part]
        coords, faces = np.asarray(mesh.coordinates, float), np.asarray(mesh.faces, int)
        annot = atlas_dir / f"atl-Cammoun2012_space-fsaverage5_res-{SCALE_TO_RESOLUTION[scale]}_hemi-{hemi}_deterministic.annot"
        vlab, _, anames = nib.freesurfer.read_annot(str(annot))
        v2n = np.full(vlab.shape, -1)
        for i, raw in enumerate(anames):
            name = raw.decode()
            if name in EXCLUDED_LABELS:
                continue
            v2n[vlab == i] = node
            names.append(f"{hemi}_{name}")
            node += 1
        geo[hemi] = {"coords": coords, "faces": faces, "v2n": v2n}
    if names != labels:
        raise ValueError(f"annotation labels do not match the region order at scale {scale}")
    return geo


def draw_view(ax, g: dict, values: np.ndarray, cmap, norm, view_x: int, hsign: int, outline: np.ndarray | None = None) -> None:
    c, f = g["coords"], g["faces"]
    tri = c[f]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    facing = normal[:, 0] * view_x
    order = np.argsort(-tri[:, :, 0].mean(1) * view_x)  # painter: far faces first
    keep = order[facing[order] > -0.2]
    shade = 0.62 + 0.38 * np.clip(facing[keep], 0, 1)[:, None]
    node = g["v2n"][f[keep]]
    labelled = (node >= 0).all(axis=1)
    rgb = np.tile(mpl.colors.to_rgb(WALL), (len(keep), 1))
    fv = values[node[labelled]].mean(axis=1)
    rgb[labelled] = cmap(norm(fv))[:, :3]
    poly = np.stack([hsign * tri[keep][:, :, 1], tri[keep][:, :, 2]], axis=-1)
    ax.add_collection(PolyCollection(poly, facecolors=np.clip(rgb * shade, 0, 1), edgecolors="none", rasterized=True))
    if outline is not None:
        # boundary edges between faces inside and outside the outlined regions, on faces facing the viewer
        vis = keep[facing[keep] > 0.05]
        nodes = g["v2n"][f[vis]]
        inside = ((nodes >= 0) & outline[np.clip(nodes, 0, None)]).sum(axis=1) >= 2
        edges = np.sort(np.concatenate([f[vis][:, [0, 1]], f[vis][:, [1, 2]], f[vis][:, [2, 0]]]), axis=1)
        flag = np.tile(inside, 3)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        lo = np.ones(len(uniq), bool)
        hi = np.zeros(len(uniq), bool)
        np.logical_and.at(lo, inv.ravel(), flag)
        np.logical_or.at(hi, inv.ravel(), flag)
        border = uniq[hi & ~lo]
        seg = np.stack([hsign * c[border][:, :, 1], c[border][:, :, 2]], axis=-1)
        ax.add_collection(LineCollection(seg, colors=INK, linewidths=0.7, zorder=3))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def brain_row(fig, gs_row, geo, axes_tab, col, cmap, norm, title, cbar_ends, outline: np.ndarray | None = None) -> None:
    sub = gs_row.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.02)
    vals = axes_tab.sort_values("node_index")[col].to_numpy()
    for j, (hemi, view, vx, hs) in enumerate(VIEWS):
        ax = fig.add_subplot(sub[0, j])
        draw_view(ax, geo[hemi], vals, cmap, norm, vx, hs, outline)
        ax.set_title(f"{'Left' if hemi == 'L' else 'Right'} {view}", fontsize=7, color=INK2, pad=1)
        if j == 0:
            ax.text(-0.04, 0.5, title, transform=ax.transAxes, ha="right", va="center", fontsize=8, color=INK, rotation=90)
    cax = fig.add_subplot(sub[0, 4])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, colors=INK2, length=2)
    if cbar_ends:
        cb.set_label(cbar_ends, fontsize=6, color=INK2, labelpad=3)


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(labelsize=6.5, colors=INK2, length=2, width=0.6)


def make_figure(scale: int, axes_all: pd.DataFrame, subj_all: pd.DataFrame, atlas_dir: Path, out: Path) -> None:
    axes_tab = axes_all[axes_all["scale"] == scale]
    subj = subj_all[subj_all["scale"] == scale]
    geo = surface(atlas_dir, scale, axes_tab.sort_values("node_index")["region_label"].tolist())
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "svg.fonttype": "none", "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 6.4), facecolor="white")
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.35], hspace=0.22, top=0.95)
    vmax1 = float(axes_tab["v1_mean"].max())
    brain_row(fig, gs[0], geo, axes_tab, "v1_mean", SEQ, Normalize(0, vmax1), "A  v1 (slow mode)", "loading")
    v1 = axes_tab.sort_values("node_index")["v1_mean"].to_numpy()
    top = v1 >= np.sort(v1)[::-1][int(np.ceil(0.2 * len(v1))) - 1]
    for row, col, label in ((1, "a", "B  Split 1: front vs back"), (2, "b", "C  Split 2 (varies)")):
        m = float(np.abs(axes_tab[col]).max())
        ends = "− posterior · frontal +" if col == "a" else "loading (sign arbitrary)"
        brain_row(fig, gs[row], geo, axes_tab, col, DIV, TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m), label, ends, outline=top)

    low = gs[3].subgridspec(1, 3, wspace=0.45, width_ratios=[1, 1.05, 1])
    # D: eigenvalue ladder
    ax = fig.add_subplot(low[0, 0])
    L = subj[[f"l{i}" for i in range(1, 7)]].to_numpy()
    x = np.arange(1, 7)
    for row in L:
        ax.plot(x, row, color=MUTED, lw=0.5, alpha=0.35)
    med = np.median(L, axis=0)
    ax.plot(x, med, color=CAT[0], lw=2, marker="o", ms=4, mec="white", mew=0.8)
    top = L.max()
    for i, txt in ((0, "gap"), (1, "ℓ2 − ℓ3")):
        ax.text(i + 1.56, top * (1.0 - 0.09 * i), f"{txt} {med[i] - med[i + 1]:.2f}", fontsize=6, color=INK, ha="left", va="bottom")
        ax.plot([i + 1.5, i + 1.5], [med[i + 1], top * (1.0 - 0.09 * i)], color=INK, lw=0.6, ls=":")
    ax.set_xticks(x, [f"ℓ{i}" for i in x])
    ax.set_ylabel("Eigenvalue of A0", fontsize=7, color=INK2)
    ax.set_title("D  Only ℓ1 stands apart", fontsize=7.5, color=INK, loc="left")
    style(ax)
    # E: where axis a sits, every scale
    ax = fig.add_subplot(low[0, 1])
    names = ("v2", "v3", "v4+")
    for yi, sc in enumerate(SCALES):
        g = subj_all[subj_all["scale"] == sc]
        counts = [(g.a_best == 2).sum(), (g.a_best == 3).sum(), (g.a_best >= 4).sum()]
        left = 0
        for k, cnt in enumerate(counts):
            ax.barh(yi, cnt, left=left, color=CAT[k], height=0.62, edgecolor="white", linewidth=1.5)
            if k < 2:
                ax.text(left + cnt / 2, yi, f"{names[k]} {cnt}", ha="center", va="center", fontsize=6, color="white")
            else:
                ax.text(left + cnt + 1, yi, f"{names[k]} {cnt}", ha="left", va="center", fontsize=6, color=INK2)
            left += cnt
    ax.set_yticks(range(len(SCALES)), [f"{s} regions" for s in SCALES])
    ax.set_xlim(0, 84)
    ax.set_xticks([0, 35, 70])
    ax.set_xlabel("Subjects", fontsize=7, color=INK2)
    ax.invert_yaxis()
    ax.set_title("E  Which eigenvector is split 1?", fontsize=7.5, color=INK, loc="left")
    style(ax)
    # F: does small spacing drive the swap?
    ax = fig.add_subplot(low[0, 2])
    ax.scatter(subj["d23"], subj["cos2_a_v2"], s=14, color=CAT[0], edgecolor="white", linewidth=0.6)
    r = stats.spearmanr(subj["d23"], subj["cos2_a_v2"])
    ax.text(0.0, 1.0, f"Spearman ρ = {r.statistic:+.2f}, p = {r.pvalue:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlabel("ℓ2 − ℓ3", fontsize=7, color=INK2)
    ax.set_ylabel("cos²(split 1, v2)", fontsize=7, color=INK2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("F  Swaps don't track spacing", fontsize=7.5, color=INK, loc="left", pad=12)
    style(ax)

    fig.text(0.02, 0.035, "Outlined in B and C: the top 20% of regions by v1 (panel A). Red and blue regions move in opposite "
             "directions; which side is red is arbitrary.", fontsize=6.5, color=INK2, ha="left", va="top")
    fig.suptitle(f"Leading eigenvectors of the structural connectome ({scale} regions, n = {len(subj)})", fontsize=9, color=INK, x=0.02, ha="left", y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_eigvecs_scale-{scale}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ctx = stage_from_argv("revision/mechanism/06_make_eigvec_figure")
    ctx.require_stage("revision/mechanism/05_second_subspace")
    ctx.begin()
    mech = ctx.root / "revision" / "mechanism"
    axes_all = pd.read_csv(mech / "second_subspace_axes.csv")
    subj_all = pd.read_csv(mech / "second_subspace_subjects.csv")
    out = figures_dir(ctx.root, "revision", "mechanism")
    atlas_dir = REPO_ROOT / ctx.config["paths"]["atlas_dir"]
    for scale in SCALES:
        make_figure(scale, axes_all, subj_all, atlas_dir, out)
        print(scale, "done", flush=True)
    ctx.finish(outputs=[out / f"fig_eigvecs_scale-{s}.{e}" for s in SCALES for e in ("png", "pdf")])


if __name__ == "__main__":
    main()
