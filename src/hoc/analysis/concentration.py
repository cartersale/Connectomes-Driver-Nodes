"""Driver-set orderings shared by the concentration (q1) and Gramian-alignment stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TAU = 1.0  # horizon of the finite-time modal-controllability score (top_modal_ct)
MATCHED = ("degree_matched", "spatial_matched", "degree_spatial_matched")


def top(score: np.ndarray, k: int, ascending: bool = False) -> np.ndarray:
    return np.argsort(score if ascending else -score, kind="stable")[:k]


def load_orders(scale: int, root: Path) -> dict:
    t = pd.read_csv(root / "primary" / "node_scores.csv", usecols=["subject_id", "scale", "node_index", "method", "rank", "score"])
    t = t[t["scale"] == scale]
    orders, scores = {}, {}
    for (sid, m), g in t.groupby(["subject_id", "method"]):
        g = g.sort_values("rank")
        orders[(sid, m)] = g["node_index"].to_numpy(int)
        scores[(sid, m)] = g.sort_values("node_index")["score"].to_numpy(float)
    return orders, scores


def load_matched(scale: int, root: Path) -> dict:
    t = pd.read_csv(root / "spectra" / "set_metrics.csv", usecols=["subject_id", "scale", "k", "family", "sample", "nodes"])
    t = t[(t["scale"] == scale) & t["family"].isin(MATCHED)]
    out = {}
    for (sid, k), g in t.groupby(["subject_id", "k"]):
        out[(sid, int(k))] = [(f, int(s), np.array(n.split(";"), dtype=int)) for f, s, n in zip(g["family"], g["sample"], g["nodes"])]
    return out
