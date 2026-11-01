#!/usr/bin/env python
"""What are the two leading eigenvectors of A0, and what does the spectral gap measure anatomically?

Real connectomes only (70 subjects x 68/114/219 regions). Per subject: v1 (Perron vector, made
nonnegative) and v2 of A0. v2 has an arbitrary sign, so it is aligned across subjects to a template
(the subject mean, re-estimated three times) and the template is oriented so that its frontal
regions are positive.

"v1-dominant set": the 20% of regions with the largest v1 loading in that subject (CORE_FRAC). It is
the set of regions the slow mode lives on; within it, v2's sign splits the regions into two groups.

Per subject (eigvec_subjects.csv):
  l1, l2, l3, gap = l1 - l2, l2_l3 = l2 - l3
  v1_consistency      Pearson r of v1 with the group-mean v1
  v2_consistency      |r| of v2 with the template (low when v2 and v3 swap: l2 - l3 is small)
  rho_v1_strength     Spearman rho(v1, strength); rho_v1_size: with parcel size (n_vertices)
  ipr_v1, ipr_v2      inverse participation ratio sum v^4 (1/N if spread evenly)
  v1_left_share       share of v1's squared mass in the left hemisphere
  v2_hemi_match       share of regions whose v2 sign matches their hemisphere (0.5 = no L/R split)
  r_v2_{x,y,z}        Pearson r of v2 with centroid coordinates (y: posterior -> anterior)
  v2_on_v1set         share of v2's squared mass on the v1-dominant set (CORE_FRAC if unrelated)
  absr_v2_y_v1set     |r(v2, y)| within the v1-dominant set: how anterior-posterior the split is
  v1set_density       mean weight among v1-dominant regions
  split_cross_within  mean weight between the two v2-sign halves of the v1-dominant set divided by
                      the mean weight within halves (high = halves tightly bound together)
  inter_frac          share of total weight on inter-hemispheric connections
Across subjects (eigvec_corr.csv): Spearman rho of gap, l1 and l2 with each subject-level measure.
Per region (eigvec_regions.csv): group-mean v1, v2 template, mean |v2|, labels, lobe, size.

Outputs (runs/<name>/mechanisms/; choose the run with ``--run <name>``): eigvec_subjects.csv,
eigvec_regions.csv, eigvec_corr.csv, eigvec_summary.md
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

from hoc.analysis.null_models import SCALES, load_stack  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402

CORE_FRAC = 0.20
SUBJECT_MEASURES = [
    "v1_consistency", "v2_consistency", "rho_v1_strength", "rho_v1_size", "ipr_v1", "ipr_v2", "v1_left_share",
    "v2_hemi_match", "r_v2_x", "r_v2_y", "r_v2_z", "v2_on_v1set", "absr_v2_y_v1set", "v1set_density",
    "split_cross_within", "inter_frac", "l2_l3",
]
CORR_MEASURES = ["v1set_density", "split_cross_within", "v2_on_v1set", "absr_v2_y_v1set", "ipr_v1", "ipr_v2", "inter_frac", "l2_l3"]


def region_table(root: Path, scale: int) -> pd.DataFrame:
    meta = pd.read_csv(root / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "region_label", "hemi", "anatomical_group", "method"])
    meta = meta[(meta["scale"] == scale) & (meta["method"] == "degree_strength") & (meta["subject_id"] == "sub-000")]
    meta = meta.drop(columns=["subject_id", "method"]).sort_values("node_index").reset_index(drop=True)
    xyz = pd.read_csv(root / "spectra" / f"centroids_scale-{scale}.csv").sort_values("node_index").reset_index(drop=True)
    if not (xyz["region_label"].to_numpy() == meta["region_label"].to_numpy()).all():
        raise ValueError(f"region order of centroids and node_scores differ at scale {scale}")
    return meta.join(xyz[["n_vertices", "x", "y", "z"]])


def eigvecs(stack: np.ndarray, regions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sub, n = stack.shape[:2]
    V1, V2, L = np.zeros((n_sub, n)), np.zeros((n_sub, n)), np.zeros((n_sub, 3))
    for s in range(n_sub):
        lam, U = np.linalg.eigh(stack[s])
        V1[s] = U[:, -1] * np.sign(U[:, -1].sum())
        V2[s] = U[:, -2]
        L[s] = lam[-1], lam[-2], lam[-3]
    template = V2[0].copy()
    for _ in range(3):
        V2 *= np.sign((V2 * template).sum(axis=1))[:, None]
        template = V2.mean(axis=0)
    if template[(regions["anatomical_group"] == "frontal").to_numpy()].mean() < 0:
        V2, template = -V2, -template
    return V1, V2, L


def subject_rows(stack, V1, V2, L, regions, scale) -> pd.DataFrame:
    left = (regions["hemi"] == "L").to_numpy()
    X, Y, Z, size = (regions[c].to_numpy(float) for c in ("x", "y", "z", "n_vertices"))
    v1_mean, template = V1.mean(axis=0), V2.mean(axis=0)
    n_set = int(np.ceil(CORE_FRAC * V1.shape[1]))
    rows = []
    for s in range(stack.shape[0]):
        A, v1, v2 = stack[s], V1[s], V2[s]
        top = np.argsort(-v1)[:n_set]
        pos, neg = top[v2[top] > 0], top[v2[top] <= 0]
        within = np.r_[A[np.ix_(pos, pos)][np.triu_indices(len(pos), 1)], A[np.ix_(neg, neg)][np.triu_indices(len(neg), 1)]]
        cross = A[np.ix_(pos, neg)].mean() if len(pos) and len(neg) else np.nan
        inter = A[np.ix_(left, ~left)].sum()
        rows.append({
            "subject_id": f"sub-{s:03d}", "scale": scale,
            "l1": L[s, 0], "l2": L[s, 1], "l3": L[s, 2], "gap": L[s, 0] - L[s, 1], "l2_l3": L[s, 1] - L[s, 2],
            "v1_consistency": np.corrcoef(v1, v1_mean)[0, 1],
            "v2_consistency": abs(np.corrcoef(v2, template)[0, 1]),
            "rho_v1_strength": stats.spearmanr(v1, A.sum(axis=0)).statistic,
            "rho_v1_size": stats.spearmanr(v1, size).statistic,
            "ipr_v1": np.sum(v1**4), "ipr_v2": np.sum(v2**4),
            "v1_left_share": np.sum(v1[left] ** 2),
            "v2_hemi_match": np.mean(np.sign(v2) == np.where(left, 1, -1)),
            "r_v2_x": np.corrcoef(v2, X)[0, 1], "r_v2_y": np.corrcoef(v2, Y)[0, 1], "r_v2_z": np.corrcoef(v2, Z)[0, 1],
            "v2_on_v1set": np.sum(v2[top] ** 2),
            "absr_v2_y_v1set": abs(np.corrcoef(v2[top], Y[top])[0, 1]),
            "v1set_density": A[np.ix_(top, top)][np.triu_indices(n_set, 1)].mean(),
            "split_cross_within": cross / within.mean() if within.size else np.nan,
            "inter_frac": inter / (inter + (A.sum() / 2 - inter)),
        })
    return pd.DataFrame(rows)


def correlations(subj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, g in subj.groupby("scale"):
        for m in CORR_MEASURES:
            for y in ("gap", "l1", "l2"):
                r = stats.spearmanr(g[y], g[m], nan_policy="omit")
                rows.append({"scale": scale, "measure": m, "outcome": y, "rho": float(r.statistic), "p": float(r.pvalue), "n": int(g[m].notna().sum())})
    return pd.DataFrame(rows)


def summary(subj: pd.DataFrame, reg: pd.DataFrame, corr: pd.DataFrame) -> str:
    med = subj.groupby("scale")[SUBJECT_MEASURES].median().T
    lines = ["# Leading eigenvectors of A0 (auto-generated by 04_leading_eigenvectors.py)", "",
             f"v1-dominant set = top {CORE_FRAC:.0%} of regions by v1 loading, per subject. Medians over 70 subjects.", "",
             "| measure | " + " | ".join(map(str, med.columns)) + " |", "|---|" + "---|" * len(med.columns)]
    lines += [f"| {m} | " + " | ".join(f"{v:.3f}" for v in med.loc[m]) + " |" for m in med.index]
    lines += ["", "## Regions (group mean)", ""]
    for scale, g in reg.groupby("scale"):
        hi = lambda col, asc: ", ".join(g.sort_values(col, ascending=asc)["region_label"].head(8))  # noqa: E731
        lines += [f"**{scale}**", "", f"- top v1: {hi('v1_mean', False)}", f"- v2 positive: {hi('v2_template', False)}",
                  f"- v2 negative: {hi('v2_template', True)}", ""]
    lines += ["## Across-subject Spearman rho", "", "| scale | measure | gap | l1 | l2 |", "|---|---|---|---|---|"]
    star = lambda r: f"{r.rho:+.2f}{'*' if r.p < 0.01 else ''}"  # noqa: E731
    for (scale, m), g in corr.groupby(["scale", "measure"], sort=False):
        d = {r.outcome: star(r) for r in g.itertuples()}
        lines.append(f"| {scale} | {m} | {d['gap']} | {d['l1']} | {d['l2']} |")
    lines += ["", "\\* p < 0.01 (uncorrected)."]
    return "\n".join(lines) + "\n"


def main():
    ctx = stage_from_argv("mechanisms/04_leading_eigenvectors")
    ctx.require_stage("primary/00_prepare_data")
    ctx.require_stage("primary/01_primary_analysis")
    ctx.require_stage("spectra/01_compute_spectra")
    ctx.begin()
    out = ctx.root / "mechanisms"
    out.mkdir(parents=True, exist_ok=True)
    subj, reg = [], []
    for scale in SCALES:
        regions = region_table(ctx.root, scale)
        stack = load_stack(scale, ctx.root)
        V1, V2, L = eigvecs(stack, regions)
        subj.append(subject_rows(stack, V1, V2, L, regions, scale))
        reg.append(regions.assign(v1_mean=V1.mean(0), v2_template=V2.mean(0), v2_abs_mean=np.abs(V2).mean(0)))
    subj, reg = pd.concat(subj, ignore_index=True), pd.concat(reg, ignore_index=True)
    corr = correlations(subj)
    paths = [out / "eigvec_subjects.csv", out / "eigvec_regions.csv", out / "eigvec_corr.csv", out / "eigvec_summary.md"]
    subj.to_csv(paths[0], index=False)
    reg.to_csv(paths[1], index=False)
    corr.to_csv(paths[2], index=False)
    paths[3].write_text(summary(subj, reg, corr))
    print(paths[3].read_text())
    ctx.finish(outputs=paths)


if __name__ == "__main__":
    main()
