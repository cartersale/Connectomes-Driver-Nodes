#!/usr/bin/env python
"""Is v2 a single, well-defined mode, or one direction in a near-degenerate v2-v3 plane?

When l2 - l3 is small, the individual eigenvectors v2 and v3 are not stable across subjects: small
differences in wiring rotate them into each other while the plane they span stays put. This stage
separates the two cases on the real connectomes (70 subjects x 68/114/219 regions).

Consensus plane. Per scale, Pbar = mean over subjects of the projector onto span(v2, v3). Its two
leading eigenvectors are the consensus axes a and b (oriented: a's frontal mean > 0, b's left
hemisphere mean > 0). The eigenvalues of Pbar (each <= 1) say how consistent the plane is (1 = every
subject has exactly this plane); for comparison, the leading eigenvalue of mean(v2 v2') says how
consistent v2 is on its own.

Per subject (second_subspace_subjects.csv):
  l1..l6, gap = l1 - l2, d23 = l2 - l3, d34 = l3 - l4, bulk_spacing = mean spacing of l2..l11,
  r23 = d23 / gap, d23_vs_bulk = d23 / bulk_spacing
  plane_overlap   ||[a b]' [v2 v3]||_F^2 / 2   (1 = the subject's v2-v3 plane is the consensus plane)
  plane_overlap4  same with span(v2, v3, v4) against the consensus 3-plane (is a third mode involved?)
  v2_cos_a        |cos(v2, a)|: is the subject's v2 the consensus axis a?
  theta_deg       rotation of v2 within the consensus plane, atan2(|v2.b|, |v2.a|) in degrees
                  (0 = v2 is a, 90 = v2 is b; the subject's v2 and v3 have swapped)
  cos2_a_v{2..6}  squared cosine of the consensus axis a with each of the subject's v2..v6
  a_best         which of v2..v6 matches a best (2 = a is the subject's v2; 3 = it is v3, ...)
  split_a, split_b  cross/within weight ratio of the v1-dominant set (top 20% by v1) split by the sign
                  of the consensus axis a (resp. b); the subject-independent version of
                  04_leading_eigenvectors' split_cross_within
Axes (second_subspace_axes.csv): per region, a, b, and the consensus 3-plane's third axis c.
Summary (second_subspace_summary.md): plane consistency, spacing, theta vs d23, the anatomy of a and
b, and the gap correlations of split_a / split_b.

Outputs (runs/<name>/revision/mechanism/; choose the run with ``--run <name>``):
second_subspace_subjects.csv, second_subspace_axes.csv, second_subspace_summary.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

NULLS_DIR = Path(__file__).resolve().parents[1] / "nulls"  # shared helpers: nulls/nulls_common.py
if str(NULLS_DIR) not in sys.path:
    sys.path.insert(0, str(NULLS_DIR))
from nulls_common import SCALES, load_stack  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402

CORE_FRAC = 0.20  # same v1-dominant set as 04_leading_eigenvectors
N_EIG = 11


def region_table(root: Path, scale: int) -> pd.DataFrame:
    meta = pd.read_csv(root / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "region_label", "hemi", "anatomical_group", "method"])
    meta = meta[(meta["scale"] == scale) & (meta["method"] == "degree_strength") & (meta["subject_id"] == "sub-000")]
    meta = meta.drop(columns=["subject_id", "method"]).sort_values("node_index").reset_index(drop=True)
    xyz = pd.read_csv(root / "revision" / "spectra" / f"centroids_scale-{scale}.csv").sort_values("node_index").reset_index(drop=True)
    if not (xyz["region_label"].to_numpy() == meta["region_label"].to_numpy()).all():
        raise ValueError(f"region order of centroids and node_scores differ at scale {scale}")
    return meta.join(xyz[["x", "y", "z"]])


def consensus(Vs: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Leading eigenvectors / eigenvalues of the mean projector onto each subject's subspace."""
    with np.errstate(all="ignore"):  # spurious BLAS warnings on macOS Accelerate; result is finite
        P = np.mean([V @ V.T for V in Vs], axis=0)
    if not np.isfinite(P).all():
        raise FloatingPointError("non-finite mean projector")
    w, U = np.linalg.eigh(P)
    return U[:, ::-1][:, :dim], w[::-1][:dim]


def split_ratio(A: np.ndarray, top: np.ndarray, axis: np.ndarray) -> float:
    pos, neg = top[axis[top] > 0], top[axis[top] <= 0]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    within = np.r_[A[np.ix_(pos, pos)][np.triu_indices(len(pos), 1)], A[np.ix_(neg, neg)][np.triu_indices(len(neg), 1)]]
    return float(A[np.ix_(pos, neg)].mean() / within.mean())


def analyze_scale(root: Path, scale: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    regions = region_table(root, scale)
    stack = load_stack(scale, root)
    n_sub, n = stack.shape[:2]
    lams, V1, V23, V234, Vtop = [], [], [], [], []
    for s in range(n_sub):
        lam, U = np.linalg.eigh(stack[s])
        Vtop.append(U[:, ::-1][:, 1:6])
        lams.append(lam[::-1][:N_EIG])
        V1.append(U[:, -1] * np.sign(U[:, -1].sum()))
        V23.append(U[:, [-2, -3]])
        V234.append(U[:, [-2, -3, -4]])
    lams, V1 = np.array(lams), np.array(V1)
    ab, w_plane = consensus(V23, 2)
    abc, w_plane3 = consensus(V234, 3)
    frontal = (regions["anatomical_group"] == "frontal").to_numpy()
    left = (regions["hemi"] == "L").to_numpy()
    a, b = ab[:, 0], ab[:, 1]
    a *= np.sign(a[frontal].mean())
    b *= np.sign(b[left].mean() - b[~left].mean())
    # third axis of the 3-plane: the part orthogonal to span(a, b)
    c = abc @ np.linalg.svd(ab.T @ abc)[2][-1]
    c /= np.linalg.norm(c)
    c *= np.sign(c[frontal].mean())
    _, w_v2 = consensus([V[:, :1] for V in V23], 1)

    n_set = int(np.ceil(CORE_FRAC * n))
    rows = []
    for s in range(n_sub):
        v2, V = V23[s][:, 0], V23[s]
        ca, cb = abs(v2 @ a), abs(v2 @ b)
        top = np.argsort(-V1[s])[:n_set]
        L = lams[s]
        cos2 = (Vtop[s].T @ a) ** 2
        rows.append({
            "subject_id": f"sub-{s:03d}", "scale": scale,
            **{f"l{i + 1}": L[i] for i in range(6)},
            "gap": L[0] - L[1], "d23": L[1] - L[2], "d34": L[2] - L[3], "bulk_spacing": (L[1] - L[N_EIG - 1]) / (N_EIG - 2),
            "plane_overlap": np.sum((ab.T @ V) ** 2) / 2,
            "plane_overlap4": np.sum((abc.T @ V234[s]) ** 2) / 3,
            "v2_cos_a": ca, **{f"cos2_a_v{i + 2}": cos2[i] for i in range(5)}, "a_best": int(np.argmax(cos2)) + 2, "theta_deg": np.degrees(np.arctan2(cb, ca)),
            "split_a": split_ratio(stack[s], top, a), "split_b": split_ratio(stack[s], top, b),
        })
    subj = pd.DataFrame(rows)
    subj["r23"] = subj["d23"] / subj["gap"]
    subj["d23_vs_bulk"] = subj["d23"] / subj["bulk_spacing"]
    axes = regions.assign(scale=scale, a=a, b=b, c=c, v1_mean=V1.mean(axis=0))
    info = {"w_plane": w_plane, "w_plane3": w_plane3, "w_v2": float(w_v2[0]), "lam_mean": lams.mean(axis=0)}
    return subj, axes, info


def describe_axis(axes: pd.DataFrame, col: str) -> str:
    r = {ax: np.corrcoef(axes[col], axes[ax])[0, 1] for ax in ("x", "y", "z")}
    left = axes["hemi"] == "L"
    hi = ", ".join(axes.sort_values(col, ascending=False)["region_label"].head(5))
    lo = ", ".join(axes.sort_values(col)["region_label"].head(5))
    return (f"r with x (L->R) {r['x']:+.2f}, y (post->ant) {r['y']:+.2f}, z (inf->sup) {r['z']:+.2f}; "
            f"mean L {axes.loc[left, col].mean():+.3f} vs R {axes.loc[~left, col].mean():+.3f}. "
            f"Positive: {hi}. Negative: {lo}.")


def summary(subj: pd.DataFrame, axes: pd.DataFrame, infos: dict) -> str:
    lines = ["# v2 vs the v2-v3 plane (auto-generated by 05_second_subspace.py)", ""]
    lines += ["## Eigenvalue ladder (subject means)", "", "| scale | l1 | l2 | l3 | l4 | l5 | l6 |", "|---|---|---|---|---|---|---|"]
    for scale, info in infos.items():
        lines.append(f"| {scale} | " + " | ".join(f"{v:.3f}" for v in info["lam_mean"][:6]) + " |")
    lines += ["", "## Consistency across subjects", "",
              "| scale | v2 alone | v2-v3 plane (axis a, b) | v2-v4 3-plane (3rd axis) | median plane overlap | median 3-plane overlap |",
              "|---|---|---|---|---|---|"]
    for scale, info in infos.items():
        g = subj[subj["scale"] == scale]
        lines.append(f"| {scale} | {info['w_v2']:.2f} | {info['w_plane'][0]:.2f}, {info['w_plane'][1]:.2f} | {info['w_plane3'][2]:.2f} | "
                     f"{g['plane_overlap'].median():.2f} | {g['plane_overlap4'].median():.2f} |")
    lines += ["", "Eigenvalues of the mean projector (1 = identical in every subject). 'v2 alone' uses v2 only.", "",
              "## Spacing and mixing (medians; rho across subjects)", "",
              "| scale | gap | d23 | d23 / gap | d23 / bulk spacing | theta (deg) | subjects with theta > 45 | rho(theta, d23) | rho(theta, d23/gap) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for scale, g in subj.groupby("scale"):
        r1, r2 = stats.spearmanr(g["theta_deg"], g["d23"]), stats.spearmanr(g["theta_deg"], g["r23"])
        lines.append(f"| {scale} | {g.gap.median():.3f} | {g.d23.median():.3f} | {g.r23.median():.2f} | {g.d23_vs_bulk.median():.2f} | "
                     f"{g.theta_deg.median():.0f} | {(g.theta_deg > 45).sum()}/70 | {r1.statistic:+.2f} (p={r1.pvalue:.1g}) | {r2.statistic:+.2f} (p={r2.pvalue:.1g}) |")
    lines += ["", "## Where the consensus axis a sits in each subject", "",
              "| scale | median cos^2(a, v2) | v3 | v4 | share of a in v2..v6 | a is v2 / v3 / v4+ | rho(cos^2(a, v2), d23) |",
              "|---|---|---|---|---|---|---|"]
    for scale, g in subj.groupby("scale"):
        r = stats.spearmanr(g["cos2_a_v2"], g["d23"])
        share = g[[f"cos2_a_v{i}" for i in range(2, 7)]].sum(axis=1).median()
        counts = f"{(g.a_best == 2).sum()} / {(g.a_best == 3).sum()} / {(g.a_best >= 4).sum()}"
        lines.append(f"| {scale} | {g.cos2_a_v2.median():.2f} | {g.cos2_a_v3.median():.2f} | {g.cos2_a_v4.median():.2f} | {share:.2f} | {counts} | "
                     f"{r.statistic:+.2f} (p={r.pvalue:.1g}) |")
    lines += ["", "## Consensus axes", ""]
    for scale, g in axes.groupby("scale"):
        lines += [f"**{scale}**", "", f"- a: {describe_axis(g, 'a')}", f"- b: {describe_axis(g, 'b')}", f"- c (3rd axis): {describe_axis(g, 'c')}", ""]
    lines += ["## Gap vs splitting the v1-dominant set along a fixed axis (Spearman rho across subjects)", "",
              "| scale | split_a: gap | l1 | l2 | split_b: gap | l1 | l2 |", "|---|---|---|---|---|---|---|"]
    for scale, g in subj.groupby("scale"):
        cells = []
        for col in ("split_a", "split_b"):
            for y in ("gap", "l1", "l2"):
                r = stats.spearmanr(g[y], g[col], nan_policy="omit")
                cells.append(f"{r.statistic:+.2f}{'*' if r.pvalue < 0.01 else ''}")
        lines.append(f"| {scale} | " + " | ".join(cells) + " |")
    lines += ["", "\\* p < 0.01 (uncorrected)."]
    return "\n".join(lines) + "\n"


def main():
    ctx = stage_from_argv("revision/mechanism/05_second_subspace")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.require_stage("revision/spectra/01_compute_spectra")
    ctx.begin()
    out = ctx.root / "revision" / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    subj, axes, infos = [], [], {}
    for scale in SCALES:
        s, a, info = analyze_scale(ctx.root, scale)
        subj.append(s)
        axes.append(a)
        infos[scale] = info
    subj, axes = pd.concat(subj, ignore_index=True), pd.concat(axes, ignore_index=True)
    paths = [out / "second_subspace_subjects.csv", out / "second_subspace_axes.csv", out / "second_subspace_summary.md"]
    subj.to_csv(paths[0], index=False)
    axes.to_csv(paths[1], index=False)
    paths[2].write_text(summary(subj, axes, infos))
    print(paths[2].read_text())
    ctx.finish(outputs=paths)


if __name__ == "__main__":
    main()
