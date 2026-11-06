#!/usr/bin/env python
"""Is a driver set's most reachable pattern the core mode v1? Also per-node strength vs v1 (Fig 3B, 3D).

For every subject (70), scale (68/114/219) and k in {1, 2, 3, 5} at c = 0.1, infinite horizon:
  - the named criteria and the 100 primary random sets, built as in concentration/01_compute_q1.py
    (same rankings, same random seeds; matched sets are left out);
  - the Gramian's top eigenvector, and its cosine with v1.

In A0's eigenbasis the Gramian is G_S = K o (U_S^T U_S), and v1 is the last basis vector. So the
cosine is the absolute last component of G_S's top eigenvector. One set is cross-checked against
the full-space Gramian from hoc.control.

Also, per node: strength s_i, v1(i), and the strength-weighted mean loading of its neighbours
m_i = (A v1)_i / s_i. These satisfy v1(i) = s_i m_i / l1.

Outputs in <run>/mechanisms/:
  gramian_alignment.csv     subject_id, scale, k, family, sample, cos_top_v1, q1, loading
  node_v1_strength.csv      subject_id, scale, node_index, strength, v1, m
  gramian_alignment_summary.md
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from hoc import config as hoc_config  # noqa: E402
from hoc.analysis import concentration as q1mod  # noqa: E402
from hoc.analysis.null_models import Spectral, load_stack, random_driver_sets  # noqa: E402
from hoc.control import gramian_metrics  # noqa: E402
from hoc.preprocessing import stabilize_adjacency  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402
from hoc.utils import n_jobs  # noqa: E402

SCALES = hoc_config.scales()
C = 0.1
K_VALUES = hoc_config.k_values()


def sets_for(spec: Spectral, A: np.ndarray, sid: str, si: int, scale: int, k: int, orders: dict, scores: dict) -> list:
    """Named criteria and random sets, exactly as concentration/01_compute_q1.job builds them at this c."""
    U, lam = spec.U, spec.lam
    sigma = np.linalg.svd(A, compute_uv=False)[0]
    modal_gu = (U**2) @ (1.0 - (lam / (1.0 + sigma)) ** 2)
    strength = scores[(sid, "degree_strength")]
    cp = scores[(sid, "cycle_participation")]
    X = np.column_stack([np.ones_like(strength), strength])
    resid = cp - X @ np.linalg.lstsq(X, cp, rcond=None)[0]
    mu = lam - spec.lam_max - C
    avg = (U**2) @ (1.0 / (-2.0 * mu))
    modal_ct = (U**2) @ (1.0 - np.exp(2.0 * mu * q1mod.TAU))
    top = q1mod.top
    methods = sorted(m for (s, m) in orders if s == sid)
    out = [(m, -1, orders[(sid, m)][:k]) for m in methods]
    out += [
        ("cycle_resid", -1, top(resid, k)),
        ("top_avg_ctrl", -1, top(avg, k)),
        ("top_modal_gu", -1, top(modal_gu, k)),
        ("top_modal_ct", -1, top(modal_ct, k)),
        ("low_perron", -1, top(spec.perron_sq, k, ascending=True)),
    ]
    out += [("random", r, s) for r, s in enumerate(random_driver_sets("real", -1, scale, si, k))]
    return out


def job(scale: int, si: int, A: np.ndarray, orders: dict, scores: dict) -> tuple[list, list]:
    sid = f"sub-{si:03d}"
    spec = Spectral(A)
    kernel = spec.kernel(C)
    rows = []
    for k in K_VALUES:
        sets = sets_for(spec, A, sid, si, scale, k, orders, scores)
        stack = np.stack([kernel * (spec.U[s, :].T @ spec.U[s, :]) for _, _, s in sets])
        w, vec = np.linalg.eigh(stack)
        q1 = w[:, -1] / np.clip(w, 0, None).sum(axis=1)
        cos = np.abs(vec[:, -1, -1])  # last eigenvector's component on the last basis vector (v1)
        for j, (fam, r, s) in enumerate(sets):
            rows.append({"subject_id": sid, "scale": scale, "k": k, "family": fam, "sample": r,
                         "cos_top_v1": float(cos[j]), "q1": float(q1[j]), "loading": float(spec.perron_sq[s].sum())})
    s = A.sum(axis=1)
    v1 = spec.perron
    m = (A @ v1) / s
    nodes = [{"subject_id": sid, "scale": scale, "node_index": i, "strength": float(s[i]), "v1": float(v1[i]), "m": float(m[i])}
             for i in range(len(s))]
    return rows, nodes


def cross_check(A: np.ndarray, drivers: np.ndarray) -> float:
    """|cos| from the full-space Gramian (hoc.control) minus the eigenbasis value; should be ~0."""
    spec = Spectral(A)
    G = spec.kernel(C) * (spec.U[drivers, :].T @ spec.U[drivers, :])
    _, vec = np.linalg.eigh(G)
    fast = abs(vec[-1, -1])
    system, _ = stabilize_adjacency(A, C)
    lam, u = np.linalg.eigh(system)
    ub = u[drivers, :].T
    W = u @ ((ub @ ub.T) / -(lam[:, None] + lam[None, :])) @ u.T
    top = gramian_metrics(0.5 * (W + W.T), 1e-5)["eigenvectors"][:, -1]
    return abs(abs(float(top @ spec.perron)) - fast)


def summarize(df: pd.DataFrame) -> str:
    lines = ["# Gramian top eigenvector vs v1 (c = 0.1, infinite horizon)", "",
             "Cosine between the most reachable pattern (top eigenvector of W_S) and v1. Subject means; random = mean over",
             "the 100 random sets per subject, then over subjects.", ""]
    keep = ["degree_strength", "eigenvector_centrality", "top_avg_ctrl", "betweenness", "cycle_participation", "random"]
    for scale in SCALES:
        d = df[(df.scale == scale) & df.family.isin(keep)]
        t = d.groupby(["family", "subject_id", "k"]).cos_top_v1.mean().groupby(["family", "k"]).mean().unstack("k")
        lines += [f"## {scale} regions (columns: k)", "", "```", t.round(3).to_string(), "```", ""]
    return "\n".join(lines)


def main():
    ctx = stage_from_argv("mechanisms/08_gramian_alignment")
    ctx.require_stage("primary/00_prepare_data")
    ctx.require_stage("primary/01_primary_analysis")
    ctx.begin()
    out = ctx.root / "mechanisms"
    out.mkdir(parents=True, exist_ok=True)
    rows, nodes = [], []
    for scale in SCALES:
        stack = load_stack(scale, ctx.root)
        orders, scores = q1mod.load_orders(scale, ctx.root)
        res = Parallel(n_jobs=n_jobs(8))(delayed(job)(scale, si, stack[si], orders, scores) for si in range(stack.shape[0]))
        for r, n in res:
            rows += r
            nodes += n
        err = cross_check(stack[0], orders[("sub-000", "degree_strength")][:3])
        print(f"{scale} done; full-space cross-check |diff| = {err:.2e}", flush=True)
        if err > 1e-6:
            raise RuntimeError(f"eigenbasis cosine disagrees with the full Gramian at scale {scale}: {err}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "gramian_alignment.csv", index=False)
    pd.DataFrame(nodes).to_csv(out / "node_v1_strength.csv", index=False)
    (out / "gramian_alignment_summary.md").write_text(summarize(df))
    print(summarize(df))
    ctx.finish(outputs=[out / "gramian_alignment.csv", out / "node_v1_strength.csv", out / "gramian_alignment_summary.md"])


if __name__ == "__main__":
    main()
