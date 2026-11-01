#!/usr/bin/env python
"""Does cutting the anterior-posterior links of the medial core close the spectral gap of A0?

04_leading_eigenvectors found that v1 lives on a medial core and v2 splits that core front from back.
This stage tests the causal version of that reading without using v2 to choose anything: define the
core and its two halves from anatomical labels, weaken only the connections between the halves, and
see whether l1 - l2 shrinks more than when the same weight is taken from other connections.

Core (CORE, by Desikan-Killiany base label, both hemispheres, every sub-parcel at 114/219):
  anterior   superiorfrontal, rostralanteriorcingulate, caudalanteriorcingulate
  posterior  posteriorcingulate, isthmuscingulate, precuneus, paracentral
Two splits of that core (partition column):
  labels     the label groups above (primary)
  ycut       the same core regions split at the median centroid y of the core (robustness check)
  v2sign     reference only, and circular: the same core split by the sign of the subject's own v2. It
             shows how much the gap can close when the cut follows v2 exactly; the gap between it and
             the anatomical splits measures how far v2's split of the core departs from the labels.

Edge classes (upper triangle, nonzero weight): cross = anterior-posterior; within = anterior-anterior
or posterior-posterior; other = every edge with at least one endpoint outside the core.

Conditions, per subject and cut level alpha (ALPHAS). W = alpha x total cross weight is the weight
removed in every condition:
  cross            every cross edge scaled by (1 - alpha)
  within_uniform   every within edge scaled by the same factor, chosen to remove W (NaN if W exceeds
                   the within-core weight)
  other_uniform    the same for the other edges
  within_matched,  N_DRAWS random edge sets: each cross edge is paired with a pool edge chosen at random
  other_matched    among its K_NEAR nearest by (log weight, log s_i + log s_j) (s = node strength in the
                   unperturbed graph), without replacement while the pool lasts; each pool edge then
                   loses t x (summed weight of its cross partners), capped at its own weight, with t set
                   so the total removed is W. removed_ratio < 1 marks draws that could not reach W.
Every condition records l1, l2, l3 of the perturbed matrix, gap = l1 - l2, and v2_overlap = |cos| of
the perturbed v2 with the unperturbed v2 (low = v2 changed identity, e.g. swapped with v3).

Outputs (runs/<name>/revision/mechanism/; choose the run with ``--run`` <name>):
  core_cut_regions.csv     core membership per region and partition
  core_cut_baseline.csv    per subject: l1..l3, class weights, v1 mass on the core, v2 sign agreement
                           with the partition (descriptive only), and first-order dl1 / dl2 per unit of
                           weight removed from the cross and the within edges (fo_*: -2 <v_k,i v_k,j>
                           weighted by w; shows which eigenvalue each edge class moves)
  core_cut_v2_side.csv     per core parcel: share of subjects in which v2 (oriented to agree with the
                           label split) puts the parcel on the side opposite its label
  core_cut.csv.gz          per subject x partition x alpha x condition (x draw): eigenvalues
  core_cut_compare.csv     per scale x partition x alpha x control: paired cross-vs-control tests on
                           dl1, dl2 and dgap separately
  core_cut_summary.md
At alpha = 1 the strength-matched controls often cannot remove the full cross weight (the cross edges are
among the heaviest in the graph), so alpha = 0.5 is the cleaner matched comparison; see removed_ratio.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import linalg, stats

NULLS_DIR = Path(__file__).resolve().parents[1] / "nulls"  # shared helpers: nulls/nulls_common.py
if str(NULLS_DIR) not in sys.path:
    sys.path.insert(0, str(NULLS_DIR))
from nulls_common import SCALES, load_stack  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402

CORE = {
    "anterior": ("superiorfrontal", "rostralanteriorcingulate", "caudalanteriorcingulate"),
    "posterior": ("posteriorcingulate", "isthmuscingulate", "precuneus", "paracentral"),
}
PARTITIONS = ("labels", "ycut")  # anatomical
REFERENCE = "v2sign"  # circular reference split, see docstring
ALPHAS = (0.25, 0.5, 0.75, 1.0)
N_DRAWS = 20
K_NEAR = 5
SEED = 20261026
HEADLINE_ALPHAS = (0.5, 1.0)  # 0.5: matched controls reach the target weight; at 1.0 they often fall short
CONTROLS = ("within_uniform", "other_uniform", "within_matched", "other_matched")


def region_table(root: Path, scale: int) -> pd.DataFrame:
    meta = pd.read_csv(root / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "region_label", "hemi", "anatomical_group", "method"])
    meta = meta[(meta["scale"] == scale) & (meta["method"] == "degree_strength") & (meta["subject_id"] == "sub-000")]
    meta = meta.drop(columns=["subject_id", "method"]).sort_values("node_index").reset_index(drop=True)
    xyz = pd.read_csv(root / "revision" / "spectra" / f"centroids_scale-{scale}.csv").sort_values("node_index").reset_index(drop=True)
    if not (xyz["region_label"].to_numpy() == meta["region_label"].to_numpy()).all():
        raise ValueError(f"region order of centroids and node_scores differ at scale {scale}")
    reg = meta.join(xyz[["n_vertices", "x", "y", "z"]])
    base = reg["region_label"].str.replace(r"^[LR]_", "", regex=True).str.replace(r"_\d+$", "", regex=True)
    reg["base_label"] = base
    reg["labels"] = np.select([base.isin(CORE["anterior"]), base.isin(CORE["posterior"])], ["anterior", "posterior"], "none")
    core = reg["labels"] != "none"
    y_med = reg.loc[core, "y"].median()
    reg["ycut"] = np.where(~core, "none", np.where(reg["y"] > y_med, "anterior", "posterior"))
    return reg


def top3(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    lam, U = linalg.eigh(A, subset_by_index=[n - 3, n - 1])
    return lam[::-1], U[:, ::-1]


def edge_classes(A: np.ndarray, group: np.ndarray):
    iu, ju = np.triu_indices(A.shape[0], 1)
    w = A[iu, ju]
    keep = w > 0
    iu, ju, w = iu[keep], ju[keep], w[keep]
    gi, gj = group[iu], group[ju]
    core = (gi != "none") & (gj != "none")
    cls = np.where(~core, "other", np.where(gi != gj, "cross", "within"))
    return iu, ju, w, cls


def apply_removal(A: np.ndarray, iu, ju, removed) -> np.ndarray:
    B = A.copy()
    B[iu, ju] -= removed
    B[ju, iu] -= removed
    return B


def matched_partners(feat_x, feat_pool, rng) -> np.ndarray:
    """For each cross edge (random order), a pool index among its K_NEAR nearest unused pool edges."""
    used = np.zeros(len(feat_pool), bool)
    out = np.empty(len(feat_x), int)
    for e in rng.permutation(len(feat_x)):
        d = np.sum((feat_pool - feat_x[e]) ** 2, axis=1)
        if used.all():
            used[:] = False  # pool exhausted: start reusing
        d[used] = np.inf
        cand = np.argsort(d)[:K_NEAR]
        cand = cand[np.isfinite(d[cand])]
        out[e] = rng.choice(cand)
        used[out[e]] = True
    return out


def matched_removal(w_pool, partners, w_x, target) -> np.ndarray:
    """Removal per pool edge: min(w_c, t * partner weight), t solved by bisection to remove `target`."""
    demand = np.bincount(partners, weights=w_x, minlength=len(w_pool))
    f = lambda t: np.minimum(w_pool, t * demand).sum()  # noqa: E731
    if f(1e6) <= target:
        return np.minimum(w_pool, 1e6 * demand)
    lo, hi = 0.0, 1.0
    while f(hi) < target:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return np.minimum(w_pool, hi * demand)


def run_subject(A, group, rng, base_row):
    lam0, U0 = top3(A)
    v2_0 = U0[:, 1]
    s = A.sum(axis=0)
    iu, ju, w, cls = edge_classes(A, group)
    feat = np.c_[np.log(w), np.log(s[iu]) + np.log(s[ju])]
    feat = (feat - feat.mean(0)) / feat.std(0)
    idx = {c: np.flatnonzero(cls == c) for c in ("cross", "within", "other")}
    W_class = {c: w[i].sum() for c, i in idx.items()}
    x = idx["cross"]
    partners = {pool: [matched_partners(feat[x], feat[idx[pool]], rng) for _ in range(N_DRAWS)] for pool in ("within", "other")}

    rows = []

    def record(alpha, cond, draw, sel, removed, target):
        lam, U = top3(apply_removal(A, iu[sel], ju[sel], removed))
        rows.append({**base_row, "alpha": alpha, "condition": cond, "draw": draw,
                     "removed_ratio": removed.sum() / target if target > 0 else np.nan,
                     "l1": lam[0], "l2": lam[1], "l3": lam[2], "gap": lam[0] - lam[1],
                     "dl1": lam[0] - lam0[0], "dl2": lam[1] - lam0[1], "dgap": (lam[0] - lam[1]) - (lam0[0] - lam0[1]),
                     "v2_overlap": abs(U[:, 1] @ v2_0)})

    for alpha in ALPHAS:
        target = alpha * W_class["cross"]
        record(alpha, "cross", 0, x, alpha * w[x], target)
        for pool in ("within", "other"):
            p = idx[pool]
            if target <= W_class[pool]:
                record(alpha, f"{pool}_uniform", 0, p, w[p] * target / W_class[pool], target)
            else:
                rows.append({**base_row, "alpha": alpha, "condition": f"{pool}_uniform", "draw": 0, "removed_ratio": np.nan})
            for d, part in enumerate(partners[pool]):
                record(alpha, f"{pool}_matched", d, p, matched_removal(w[p], part, w[x], target), target)

    v1 = U0[:, 0] * np.sign(U0[:, 0].sum())
    g = np.where(group == "anterior", 1.0, np.where(group == "posterior", -1.0, 0.0))
    core = g != 0
    baseline = {**base_row, "l1": lam0[0], "l2": lam0[1], "l3": lam0[2], "gap": lam0[0] - lam0[1],
                "n_anterior": int((g > 0).sum()), "n_posterior": int((g < 0).sum()),
                "n_cross": len(x), "n_within": len(idx["within"]), "n_other": len(idx["other"]),
                "w_cross": W_class["cross"], "w_within": W_class["within"], "w_other": W_class["other"],
                "w_cross_share": W_class["cross"] / w.sum(),
                "v1_mass_core": float(np.sum(v1[core] ** 2)), "v2_mass_core": float(np.sum(v2_0[core] ** 2)),
                "v2_sign_agree": float(abs(np.mean(np.sign(v2_0[core]) * g[core]))),
                "v2_split_corr": float(abs(np.corrcoef(v2_0[core], g[core])[0, 1])),
                "first_order_dgap_cross": float(-2 * np.sum(w[x] * (U0[iu[x], 0] * U0[ju[x], 0] - U0[iu[x], 1] * U0[ju[x], 1])))}
    for c in ("cross", "within"):
        e = idx[c]
        for k in (0, 1):
            baseline[f"fo_dl{k + 1}_{c}"] = float(-2 * np.sum(w[e] * U0[iu[e], k] * U0[ju[e], k]) / W_class[c]) if len(e) else np.nan
    return rows, baseline


def oriented_v2(A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """v2 of A, signed so that it agrees with the label split of the core (anterior positive)."""
    v2 = top3(A)[1][:, 1]
    g = np.where(labels == "anterior", 1.0, np.where(labels == "posterior", -1.0, 0.0))
    return v2 * (1.0 if v2 @ g >= 0 else -1.0)


def v2sign_group(v2: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.where(labels == "none", "none", np.where(v2 > 0, "anterior", "posterior"))


def v2_side_table(stack: np.ndarray, reg: pd.DataFrame, scale: int) -> pd.DataFrame:
    labels = reg["labels"].to_numpy()
    core = labels != "none"
    V2 = np.array([oriented_v2(A, labels) for A in stack])[:, core]
    g = np.where(labels[core] == "anterior", 1.0, -1.0)
    return reg.loc[core, ["region_label", "hemi", "base_label", "labels", "y"]].assign(
        scale=scale, v2_mean=V2.mean(0), wrong_side=(np.sign(V2) * g < 0).mean(0)).reset_index(drop=True)


def compare(df: pd.DataFrame) -> pd.DataFrame:
    per = df.groupby(["scale", "partition", "subject_id", "alpha", "condition"])[["dl1", "dl2", "dgap", "removed_ratio"]].mean().reset_index()
    rows = []
    for (scale, part, alpha), g in per.groupby(["scale", "partition", "alpha"]):
        wide = {m: g.pivot(index="subject_id", columns="condition", values=m) for m in ("dl1", "dl2", "dgap")}
        removed = g.pivot(index="subject_id", columns="condition", values="removed_ratio")
        for ctrl in CONTROLS:
            for m, tab in wide.items():
                d = (tab["cross"] - tab[ctrl]).dropna()
                rows.append({"scale": scale, "partition": part, "alpha": alpha, "control": ctrl, "measure": m, "n": len(d),
                             "median_cross": float(tab["cross"].median()), "median_control": float(tab[ctrl].median()),
                             "median_removed_ratio": float(removed[ctrl].median()),
                             "median_diff": float(d.median()) if len(d) else np.nan,
                             "frac_cross_lower": float((d < 0).mean()) if len(d) else np.nan,
                             "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if len(d) > 5 else np.nan})
    # per-subject rank of the cross cut among the matched draws (share of draws with a smaller dgap)
    out = pd.DataFrame(rows)
    draws = df[df["condition"].str.endswith("_matched")]
    cross = df[df["condition"] == "cross"].set_index(["scale", "partition", "subject_id", "alpha"])["dgap"]
    j = draws.join(cross.rename("dgap_cross"), on=["scale", "partition", "subject_id", "alpha"])
    j["below"] = j["dgap"] <= j["dgap_cross"]
    emp = j.groupby(["scale", "partition", "alpha", "condition", "subject_id"])["below"].mean().groupby(["scale", "partition", "alpha", "condition"]).median()
    emp = emp.rename("median_emp_p").reset_index().rename(columns={"condition": "control"}).assign(measure="dgap")
    return out.merge(emp, on=["scale", "partition", "alpha", "control", "measure"], how="left")


def summary(base: pd.DataFrame, df: pd.DataFrame, comp: pd.DataFrame, reg: pd.DataFrame, side: pd.DataFrame) -> str:
    L = ["# Anterior-posterior core cut (auto-generated by 07_core_cut.py)", "",
         "Core and split from anatomical labels only (v2 is never used to choose edges), except the v2sign",
         "partition: a circular reference that splits the same core by each subject's v2 sign.",
         f"Cut levels {ALPHAS}; matched controls: {N_DRAWS} draws, K = {K_NEAR} nearest by (log w, log s_i + log s_j).", "",
         "## Baseline (median over subjects)", "",
         "| scale | partition | n_ant | n_post | cross share of weight | v1 mass on core | v2 mass on core | |r(v2, split)| | gap | 1st-order dgap (alpha=1) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for (scale, part), g in base.groupby(["scale", "partition"]):
        m = g.median(numeric_only=True)
        L.append(f"| {scale} | {part} | {m.n_anterior:.0f} | {m.n_posterior:.0f} | {m.w_cross_share:.3f} | {m.v1_mass_core:.2f} | "
                 f"{m.v2_mass_core:.2f} | {m.v2_split_corr:.2f} | {m.gap:.3f} | {m.first_order_dgap_cross:+.3f} |")
    L += ["", "## Which eigenvalue each edge class moves (first order, per unit weight removed; median)", "",
          "Cross edges close the gap by lowering l1 while leaving l2 alone (or raising it: their ends sit on opposite",
          "signs of v2). Where the split does not follow v2, cross edges lower l2 too and close the gap less.", "",
          "| scale | partition | cross dl1 | cross dl2 | cross dgap | within dl1 | within dl2 | within dgap | cross > within |",
          "|---|---|---|---|---|---|---|---|---|"]
    fo = base.assign(fo_dgap_cross=base.fo_dl1_cross - base.fo_dl2_cross, fo_dgap_within=base.fo_dl1_within - base.fo_dl2_within)
    for (scale, part), g in fo.groupby(["scale", "partition"]):
        m = g.median(numeric_only=True)
        win = (g.fo_dgap_cross < g.fo_dgap_within).mean()
        L.append(f"| {scale} | {part} | {m.fo_dl1_cross:+.3f} | {m.fo_dl2_cross:+.3f} | {m.fo_dgap_cross:+.3f} | "
                 f"{m.fo_dl1_within:+.3f} | {m.fo_dl2_within:+.3f} | {m.fo_dgap_within:+.3f} | {win:.2f} |")
    L += ["", "cross > within = share of subjects where a unit of cross weight closes the gap more than a unit of within weight.",
          "", "## Core parcels v2 places against their label", "",
          "Share of subjects in which v2 (signed to agree with the label split) puts the parcel on the opposite side; "
          "parcels above 0.5 listed.", ""]
    for scale, g in side.groupby("scale"):
        bad = g[g.wrong_side > 0.5].sort_values("wrong_side", ascending=False)
        items = ", ".join(f"{r.region_label} ({r.labels[:4]}., {r.wrong_side:.2f})" for r in bad.itertuples()) or "none"
        L.append(f"- {scale} (median over parcels {g.wrong_side.median():.2f}): {items}")
    L += ["", "## Eigenvalue changes (median over subjects; matched = mean over draws first)", ""]
    per = df.groupby(["scale", "partition", "subject_id", "alpha", "condition"])[["dl1", "dl2", "dgap", "v2_overlap"]].mean().reset_index()
    per = per.merge(base[["scale", "partition", "subject_id", "gap"]].rename(columns={"gap": "gap0"}), on=["scale", "partition", "subject_id"])
    per["dgap_pct"] = 100 * per["dgap"] / per["gap0"]
    med = per.groupby(["scale", "partition", "alpha", "condition"])[["dl1", "dl2", "dgap", "dgap_pct", "v2_overlap"]].median()
    L += ["| scale | partition | alpha | condition | dl1 | dl2 | dgap | dgap % | v2 overlap |", "|---|---|---|---|---|---|---|---|---|"]
    order = ["cross", *CONTROLS]
    for (scale, part, alpha), g in med.groupby(level=[0, 1, 2]):
        g = g.droplevel([0, 1, 2])
        for c in order:
            if c in g.index:
                r = g.loc[c]
                L.append(f"| {scale} | {part} | {alpha} | {c} | {r.dl1:+.3f} | {r.dl2:+.3f} | {r.dgap:+.3f} | {r.dgap_pct:+.1f} | {r.v2_overlap:.2f} |")
    fmt = lambda r: f"{r.median_diff:+.3f} ({r.frac_cross_lower:.2f}, p={r.p_wilcoxon:.1g})" if r.n else "n/a"  # noqa: E731
    for alpha in HEADLINE_ALPHAS:
        L += ["", f"## Cross vs control at alpha = {alpha} (paired over subjects)", "",
              "Median difference cross - control (share of subjects where cross is lower, Wilcoxon p). "
              "removed = median share of the cross cut's weight the control actually removed.", "",
              "| scale | partition | control | removed | dl1 | dl2 | dgap | median emp. p (dgap) |", "|---|---|---|---|---|---|---|---|"]
        for (scale, part, ctrl), g in comp[comp["alpha"] == alpha].groupby(["scale", "partition", "control"], sort=False):
            d = {r.measure: r for r in g.itertuples()}
            emp = d["dgap"].median_emp_p
            L.append(f"| {scale} | {part} | {ctrl} | {d['dgap'].median_removed_ratio:.2f} | {fmt(d['dl1'])} | {fmt(d['dl2'])} | {fmt(d['dgap'])} | "
                     f"{'' if pd.isna(emp) else f'{emp:.2f}'} |")
    L += ["", "Emp. p = share of a subject's matched draws whose dgap is at or below the cross cut's (small = cross closes the gap more).",
          "", "## Core regions (scale 68)", ""]
    r68 = reg[reg["scale"] == min(SCALES)]
    for part in PARTITIONS:
        for side in ("anterior", "posterior"):
            L.append(f"- {part} / {side}: " + ", ".join(r68.loc[r68[part] == side, "region_label"]))
    return "\n".join(L) + "\n"


def main():
    ctx = stage_from_argv("revision/mechanism/07_core_cut")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.require_stage("revision/spectra/01_compute_spectra")
    ctx.begin()
    out = ctx.root / "revision" / "mechanism"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, base, regs, sides = [], [], [], []
    for scale in SCALES:
        reg = region_table(ctx.root, scale)
        regs.append(reg.assign(scale=scale))
        stack = load_stack(scale, ctx.root)
        if ctx.subject_limit:
            stack = stack[: ctx.subject_limit]
        sides.append(v2_side_table(stack, reg, scale))
        labels = reg["labels"].to_numpy()
        for part in (*PARTITIONS, REFERENCE):
            for s in range(stack.shape[0]):
                group = v2sign_group(oriented_v2(stack[s], labels), labels) if part == REFERENCE else reg[part].to_numpy()
                r, b = run_subject(stack[s], group, rng, {"subject_id": f"sub-{s:03d}", "scale": scale, "partition": part})
                rows += r
                base.append(b)
            print(f"scale {scale} {part}: done", flush=True)
    df, base, reg, side = pd.DataFrame(rows), pd.DataFrame(base), pd.concat(regs, ignore_index=True), pd.concat(sides, ignore_index=True)
    comp = compare(df.dropna(subset=["dgap"]))
    paths = [out / "core_cut_regions.csv", out / "core_cut_baseline.csv", out / "core_cut.csv.gz", out / "core_cut_compare.csv", out / "core_cut_summary.md",
             out / "core_cut_v2_side.csv"]
    reg[["scale", "node_index", "region_label", "hemi", "base_label", "y", *PARTITIONS]].to_csv(paths[0], index=False)
    base.to_csv(paths[1], index=False)
    df.to_csv(paths[2], index=False)
    comp.to_csv(paths[3], index=False)
    side.to_csv(paths[5], index=False)
    paths[4].write_text(summary(base, df.dropna(subset=["dgap"]), comp, reg, side))
    print(paths[4].read_text())
    ctx.finish(outputs=paths)


if __name__ == "__main__":
    main()
