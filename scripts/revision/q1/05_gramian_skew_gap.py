#!/usr/bin/env python
"""Does the graph spectral gap shape the Gramian spectrum? Across-subject test on the real connectomes.

q1 = lambda1(W)/tr W is a property of the Gramian W alone. The graph spectrum enters W through the
kernel of the fast Gramian (nulls_common.Spectral): in A0's eigenbasis G_S = K o (U_S^T U_S) with
K_ab = 1 / (2c + (l_max - l_a) + (l_max - l_b)), so the Perron mode gets weight 1/(2c) and the next
mode 1/(2c + 2 gap). A larger gap should make W's eigenvalue distribution more right-skewed.

For every subject (70), scale (68/114/219), c in {0.01, 0.1, 1}, k in {1, 2, 3, 5}, and driver set
(degree_strength and cycle_participation top-k from <run>/primary/node_scores.csv, plus the 100
primary random sets), compute from eig(W):
  skew      Fisher skewness of the raw eigenvalues (clipped at 0)
  skew_log  skewness of log10(lambda + eps), eps = 1e-5 (the pipeline's ER/PR floor); for small k
            many eigenvalues sit at the floor, so read this one with that in mind
  q1        lambda1 / tr W (raw W), as in 01_compute_q1.py
Random sets are summarized per subject by their mean. Network level: lambda1, lambda2, gap of A0.

Across subjects, within each scale x c x k (k = "avg" averages each metric over k first) x family:
  rho_gap_skew, rho_gap_skewlog, rho_gap_q1     Spearman rho with the gap
  prho_gap_skew_q1, prho_gap_skewlog_q1         partial Spearman given q1 (does skewness carry gap
                                                information beyond q1?)
  prho_gap_skew_l1                              partial Spearman given lambda1 (gap vs overall
                                                spectral scale)
  rho_skew_q1                                   how redundant skewness is with q1
p-values: Spearman (scipy) and t-test with n - 3 df for partials; Holm within each statistic
across scale x c x k x family.

Outputs (<run>/revision/q1/; choose the run with ``--run <name>``):
  skew_sets.csv.gz       one row per subject x scale x c x k x family (random = mean of 100 sets)
  skew_gap_corr.csv      the across-subject correlations
  skew_gap_summary.md    compact table at c = 0.1, k = avg
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

NULLS_DIR = Path(__file__).resolve().parents[1] / "nulls"
if str(NULLS_DIR) not in sys.path:
    sys.path.insert(0, str(NULLS_DIR))
from nulls_common import EPSILON, Spectral, load_stack, random_driver_sets  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

SCALES = (68, 114, 219)
C_VALUES = (0.01, 0.1, 1.0)
K_VALUES = (1, 2, 3, 5)
METHODS = ("degree_strength", "cycle_participation")
METRICS = ("skew", "skew_log", "q1")


def load_orders(root: Path, scale: int) -> dict:
    t = pd.read_csv(root / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "rank"])
    t = t[(t["scale"] == scale) & t["method"].isin(METHODS)]
    return {(sid, m): g.sort_values("rank")["node_index"].to_numpy(int) for (sid, m), g in t.groupby(["subject_id", "method"])}


def eig_metrics(w: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row metrics of ascending eigenvalue arrays (n_sets x N)."""
    w = np.clip(np.atleast_2d(w), 0.0, None)
    return {
        "skew": stats.skew(w, axis=1),
        "skew_log": stats.skew(np.log10(w + EPSILON), axis=1),
        "q1": w[:, -1] / w.sum(axis=1),
    }


def job(scale: int, si: int, A: np.ndarray, orders: dict) -> list[dict]:
    sid = f"sub-{si:03d}"
    spec = Spectral(A)
    net = {"subject_id": sid, "scale": scale, "lambda1": spec.lam[-1], "lambda2": spec.lam[-2], "gap": spec.gap}
    rows = []
    for c in C_VALUES:
        kernel = spec.kernel(c)
        for k in K_VALUES:
            named = [orders[(sid, m)][:k] for m in METHODS]
            rand = random_driver_sets("real", -1, scale, si, k)
            w, _, _ = spec.set_spectra(kernel, named + rand)
            met = eig_metrics(w)
            for j, m in enumerate(METHODS):
                rows.append({**net, "c": c, "k": k, "family": m, **{x: float(met[x][j]) for x in METRICS}})
            r = slice(len(METHODS), None)
            rows.append({**net, "c": c, "k": k, "family": "random_mean", **{x: float(met[x][r].mean()) for x in METRICS}})
    return rows


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman rho of x and y after regressing the ranks of z (n x p) out of both."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([np.ones(len(x)), *[stats.rankdata(col) for col in np.atleast_2d(np.asarray(z).T)]])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    dof = len(x) - 2 - (Z.shape[1] - 1)
    t = r * np.sqrt(dof / max(1e-300, 1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), dof))


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    avg = df.groupby(["subject_id", "scale", "c", "family"], as_index=False).agg(
        {**{x: "mean" for x in METRICS}, "gap": "first", "lambda1": "first"})
    both = pd.concat([df.assign(k=df["k"].astype(str)), avg.assign(k="avg")], ignore_index=True)
    rows = []
    for (scale, c, k, fam), g in both.groupby(["scale", "c", "k", "family"]):
        gap, q1, l1 = g["gap"].to_numpy(), g["q1"].to_numpy(), g["lambda1"].to_numpy()
        row = {"scale": scale, "c": c, "k": k, "family": fam, "n": len(g)}
        for name, x in (("skew", g["skew"]), ("skewlog", g["skew_log"]), ("q1", g["q1"])):
            res = stats.spearmanr(gap, x)
            row[f"rho_gap_{name}"], row[f"p_gap_{name}"] = float(res.statistic), float(res.pvalue)
        for name, x in (("skew", g["skew"]), ("skewlog", g["skew_log"])):
            row[f"prho_gap_{name}_q1"], row[f"p_gap_{name}_q1"] = partial_spearman(gap, x.to_numpy(), q1)
        row["prho_gap_skew_l1"], row["p_gap_skew_l1"] = partial_spearman(gap, g["skew"].to_numpy(), l1)
        res = stats.spearmanr(g["skew"], q1)
        row["rho_skew_q1"] = float(res.statistic)
        rows.append(row)
    out = pd.DataFrame(rows)
    for p in [col for col in out if col.startswith("p_")]:
        out[p + "_holm"] = multipletests(out[p].to_numpy(), method="holm")[1]
    return out


def summary(corr: pd.DataFrame) -> str:
    t = corr[(corr["c"] == 0.1) & (corr["k"] == "avg")]
    lines = [
        "# Gramian skewness vs graph spectral gap (across subjects; auto-generated by 05_gramian_skew_gap.py)",
        "",
        "c = 0.1, metrics averaged over k in {1,2,3,5}; n = 70 subjects per scale; Holm-adjusted p in brackets.",
        "prho | q1 = partial Spearman given q1; prho | l1 = partial Spearman given lambda1.",
        "",
        "| scale | set | rho(gap, skew) | rho(gap, skew_log) | rho(gap, q1) | prho(gap, skew \\| q1) | prho(gap, skew \\| l1) | rho(skew, q1) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    f = lambda r, p: f"{r:+.2f} [{p:.2g}]"  # noqa: E731
    for _, r in t.sort_values(["scale", "family"]).iterrows():
        lines.append(
            f"| {r.scale} | {r.family} | {f(r.rho_gap_skew, r.p_gap_skew_holm)} | {f(r.rho_gap_skewlog, r.p_gap_skewlog_holm)} | "
            f"{f(r.rho_gap_q1, r.p_gap_q1_holm)} | {f(r.prho_gap_skew_q1, r.p_gap_skew_q1_holm)} | "
            f"{f(r.prho_gap_skew_l1, r.p_gap_skew_l1_holm)} | {r.rho_skew_q1:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ctx = stage_from_argv("revision/q1/05_gramian_skew_gap")
    ctx.require_stage("00_prepare_data")
    ctx.require_stage("01_primary")
    ctx.begin()
    out = ctx.root / "revision" / "q1"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for scale in SCALES:
        stack = load_stack(scale, ctx.root)
        orders = load_orders(ctx.root, scale)
        res = Parallel(n_jobs=n_jobs(12))(delayed(job)(scale, si, stack[si], orders) for si in range(stack.shape[0]))
        frames.append(pd.DataFrame([r for rows in res for r in rows]))
        print(scale, "done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    corr = correlations(df)
    paths = [out / "skew_sets.csv.gz", out / "skew_gap_corr.csv", out / "skew_gap_summary.md"]
    df.to_csv(paths[0], index=False)
    corr.to_csv(paths[1], index=False)
    paths[2].write_text(summary(corr))
    print(paths[2].read_text())
    ctx.finish(outputs=paths)


if __name__ == "__main__":
    main()
