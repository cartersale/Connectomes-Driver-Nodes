"""Shared look for the main-text figures (Figs 1-5): fonts, colors, axis styling, saving, and brain surfaces.

Each figure script draws all its panels on one canvas, then calls ``save``. That writes PDF, SVG and a PNG
preview to ``runs/<name>/paper/figs/`` and copies the PDF and SVG into
``paper/network_neuro_resubmission/figs/``.

Brain surfaces are drawn with plain matplotlib on the fsaverage5 pial mesh, with Cammoun parcels. The mesh
is projected onto the y-z plane: lateral and medial views look along x.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from hoc.atlas import EXCLUDED_LABELS, SCALE_TO_RESOLUTION  # noqa: E402

PAPER_FIGS = REPO_ROOT / "paper" / "network_neuro_resubmission" / "figs"
ATLAS_DIR = REPO_ROOT / "data" / "external" / "atlas_cache" / "atl-cammoun2012" / "fsaverage5"

# Page widths in inches (Network Neuroscience: one column ~3.4 in, full width ~7 in).
ONE_COL = 3.4
FULL = 7.0

# Ink and surfaces
INK, INK2, MUTED, GRID = "#0b0b0b", "#0b0b0b", "#8a8984", "#e4e3df"  # INK2 = INK: all text black
WALL = "#cfcec9"  # medial wall / unlabelled vertices
RANDOM = "#b4b2ab"  # random driver sets

# Categorical: hub-oriented, betweenness, cycle / spread-out (validated colorblind-safe; always direct-labelled)
HUB, MID, SPREAD = "#eb6834", "#1baf7a", "#2a78d6"
# Ordered scales: light to dark purple
SCALES = (68, 114, 219)
SCALE_COLORS = {68: "#b3a6d6", 114: "#7a64b5", 219: "#3f1f7a"}
SCALE_MARKERS = {68: "o", 114: "s", 219: "D"}

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#5598e7", "#f0efec", "#ec7b7a", "#b8312f"])

# Driver-selection criteria: display name and color group.
CRITERIA = {
    "degree_strength": ("Degree (strength)", HUB),
    "eigenvector_centrality": ("Eigenvector centrality", HUB),
    "top_avg_ctrl": ("Average controllability", HUB),
    "betweenness": ("Betweenness", MID),
    "cycle_criticality": ("Cycle criticality", RANDOM),
    "cycle_participation": ("Cycle participation", SPREAD),
    "degree_normalized_cycle_participation": ("Cycle part. / degree", RANDOM),
    "cycle_resid": ("Cycle part. (strength removed)", RANDOM),
    "triangle_participation": ("Triangle participation", RANDOM),
    "tetrahedron_participation": ("Tetrahedron participation", RANDOM),
    "participation_coefficient": ("Participation coefficient", RANDOM),
    "low_perron": ("Low v1 loading", RANDOM),
    "top_modal_gu": ("Modal controllability", RANDOM),
    "top_modal_ct": ("Modal ctrl. (cont. time)", RANDOM),
}

C_PRIMARY = 0.1
C_NCTPY = 1.0


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],  # Arial has a bold face here; Helvetica does not
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "axes.labelcolor": INK2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.linewidth": 1.2,
            "figure.facecolor": "white",
        }
    )


def style(ax, grid: bool = False) -> None:
    """Recessive axes: no top/right spines, muted thin spines and ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, length=2, width=0.6)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def panel_label(ax, letter: str, title: str = "", x: float = -0.02, y: float = 1.0) -> None:
    """Bold panel letter at the top left, with an optional short title after it."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color=INK)
    if title:
        ax.text(x + 0.02, y, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK)


def reference_lines(ax, labels: bool = True) -> None:
    """Dotted verticals at the primary damping (c = 0.1) and the nctpy default (c = 1)."""
    for c, name in ((C_PRIMARY, "c = 0.1"), (C_NCTPY, "nctpy")):
        ax.axvline(c, color=MUTED, lw=0.6, ls=":", zorder=0)
        if labels:
            ax.text(c, 1.0, f" {name}", transform=ax.get_xaxis_transform(), fontsize=6, color=INK2, ha="left", va="top")


def save(fig, run_root: Path, name: str, copy_to_paper: bool = True) -> list[Path]:
    """Write PDF, SVG and a PNG preview into <run>/paper/figs/; copy the PDF and SVG into the manuscript folder."""
    out = Path(run_root) / "paper" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png"):
        p = out / f"{name}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300 if ext == "png" else 600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        paths.append(p)
    plt.close(fig)
    if copy_to_paper:
        PAPER_FIGS.mkdir(parents=True, exist_ok=True)
        for p in paths[:2]:
            shutil.copy2(p, PAPER_FIGS / p.name)
    return paths


# ------------------------------------------------------------------------------------------- run handling

DRAFT_FROM_LEGACY = ("q1", "nulls")  # revision folders a --draft render reads from results/revision/


def open_run(stage: str, requires: tuple[str, ...] = ()):
    """Start a figure stage. Returns (ctx, run_root, draft).

    With ``--draft`` there is no manifest entry, and q1/ and nulls/ are read from the legacy results/revision/
    tree. This previews a layout before the canonical stages finish. Draft output is never copied into the paper.
    """
    from hoc.runs import reader_run_dir, stage_from_argv

    if "--draft" in sys.argv:
        return None, reader_run_dir(), True
    ctx = stage_from_argv(stage)
    for r in requires:
        ctx.require_stage(r)
    ctx.begin()
    return ctx, ctx.root, False


def rev_dir(root: Path, sub: str, draft: bool) -> Path:
    """<run>/revision/<sub>, or results/revision/<sub> for the q1/nulls folders in a draft."""
    if draft and sub.split("/")[0] in DRAFT_FROM_LEGACY:
        return REPO_ROOT / "results" / "revision" / sub
    return Path(root) / "revision" / sub


def finish(fig, ctx, root: Path, name: str, draft: bool) -> None:
    if draft:
        paths = save(fig, root, f"drafts/{name}_draft", copy_to_paper=False)
        print("draft written:", paths[-1])
        return
    paths = save(fig, root, name)
    ctx.finish(outputs=paths)


# ------------------------------------------------------------------------------------------- brain surfaces

# (hemi, view name, viewer direction along x, sign of y on the horizontal image axis)
VIEWS = {
    "L_lateral": ("L", -1, -1),
    "L_medial": ("L", +1, +1),
    "R_medial": ("R", -1, -1),
    "R_lateral": ("R", +1, +1),
}


def surface(scale: int, labels: list[str] | None = None) -> dict:
    """fsaverage5 pial mesh per hemisphere with each vertex's node index (-1 = unlabelled).

    Node order is left then right, annotation order, excluded labels dropped. This is the order of the processed
    connectomes. When ``labels`` is given (e.g. ``L_superiorfrontal_1``), check that it matches.
    """
    from nilearn import datasets

    fsavg = datasets.load_fsaverage(mesh="fsaverage5")
    geo, names, node = {}, [], 0
    for hemi, part in (("L", "left"), ("R", "right")):
        mesh = fsavg["pial"].parts[part]
        coords, faces = np.asarray(mesh.coordinates, float), np.asarray(mesh.faces, int)
        annot = ATLAS_DIR / f"atl-Cammoun2012_space-fsaverage5_res-{SCALE_TO_RESOLUTION[scale]}_hemi-{hemi}_deterministic.annot"
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
    if labels is not None and list(labels) != names:
        raise ValueError(f"annotation labels do not match the region order at scale {scale}")
    geo["n_nodes"] = node
    return geo


def draw_view(ax, geo: dict, view: str, values: np.ndarray, cmap, norm, outline: np.ndarray | None = None,
              face_rgb: np.ndarray | None = None, outline_color: str = INK, outline_width: float = 0.6) -> None:
    """Draw one hemisphere view of node ``values``, optionally outlining the nodes where ``outline`` is True.

    ``face_rgb`` (n_nodes x 3) overrides the colormap, e.g. to grey out everything outside a region set.
    """
    hemi, view_x, hsign = VIEWS[view]
    g = geo[hemi]
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
    if face_rgb is not None:
        rgb[labelled] = face_rgb[node[labelled]].mean(axis=1)
    else:
        rgb[labelled] = cmap(norm(values[node[labelled]].mean(axis=1)))[:, :3]
    poly = np.stack([hsign * tri[keep][:, :, 1], tri[keep][:, :, 2]], axis=-1)
    ax.add_collection(PolyCollection(poly, facecolors=np.clip(rgb * shade, 0, 1), edgecolors="none", rasterized=True))
    if outline is not None:
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
        ax.add_collection(LineCollection(seg, colors=outline_color, linewidths=outline_width, zorder=3))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def anchor(geo: dict, view: str, node: int) -> np.ndarray | None:
    """Image position of a parcel's most viewer-facing vertex in ``view``; None if the parcel is in the other hemisphere."""
    hemi, view_x, hsign = VIEWS[view]
    g = geo[hemi]
    mask = g["v2n"] == node
    if not mask.any():
        return None
    xyz = g["coords"][mask]
    v = xyz[np.argmax(xyz[:, 0] * view_x)]
    return np.array([hsign * v[1], v[2]])


def centroid(geo: dict, view: str, node: int) -> np.ndarray | None:
    """Image position of a parcel's mean vertex in ``view``; None if the parcel is in the other hemisphere."""
    hemi, _, hsign = VIEWS[view]
    g = geo[hemi]
    mask = g["v2n"] == node
    if not mask.any():
        return None
    v = g["coords"][mask].mean(axis=0)
    return np.array([hsign * v[1], v[2]])


def leading_eigvec(a: np.ndarray) -> tuple[float, np.ndarray]:
    """Largest eigenvalue and its eigenvector, signed to be positive."""
    lam, u = np.linalg.eigh(a)
    v = u[:, -1]
    return float(lam[-1]), v * np.sign(v.sum())
