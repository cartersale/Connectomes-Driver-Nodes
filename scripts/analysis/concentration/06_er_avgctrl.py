#!/usr/bin/env python
"""What does effective rank track under strong damping? Effective rank vs the drivers' average controllability.

At c = 1, for each subject, scale and k in {1, 2, 3, 5}: the Spearman correlation, across that subject's 100 random
sets, between each set's effective rank (from set_q1.csv.gz) and the mean average controllability of its drivers,
AC_i = sum_a v_a(i)^2 / (-2 mu_a) with mu_a = l_a - l_1 - c. Random sets only, so the correlation is not driven by
the named criteria. The paper reports the median over subjects (Results, effective-rank paragraph; Methods,
"Spectral gap and individual differences").

Reads <run>/concentration/set_q1.csv.gz and the processed connectomes. Outputs in <run>/concentration/:
  er_avgctrl.csv          subject_id, scale, k, rho
  er_avgctrl_summary.md   median rho over subjects by scale and k
Run: python scripts/analysis/concentration/06_er_avgctrl.py --run <name>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hoc import config as hoc_config  # noqa: E402
from hoc.analysis.null_models import Spectral, load_stack  # noqa: E402
from hoc.runs import stage_from_argv  # noqa: E402

C = 1.0
SCALES = hoc_config.scales()


def avg_ctrl(A: np.ndarray, c: float) -> np.ndarray:
    sp = Spectral(A)
    with np.errstate(all="ignore"):  # spurious BLAS warnings on macOS; checked below
        ac = (sp.U**2) @ np.diag(sp.kernel(c))  # K_aa = 1 / (-2 mu_a)
    assert np.isfinite(ac).all()
    return ac


def main():
    ctx = stage_from_argv("concentration/06_er_avgctrl")
    ctx.require_stage("concentration/01_compute_q1")
    ctx.begin()
    out = ctx.root / "concentration"
    sets = pd.read_csv(out / "set_q1.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "nodes", "er"])
    sets = sets[(sets.c == C) & (sets.family == "random")]
    rows = []
    for scale in SCALES:
        stack = load_stack(scale, ctx.root)
        for si in range(stack.shape[0]):
            sid = f"sub-{si:03d}"
            ac = avg_ctrl(stack[si], C)
            x = sets[(sets.scale == scale) & (sets.subject_id == sid)]
            mac = x.nodes.astype(str).map(lambda s: ac[[int(t) for t in s.split(";")]].mean())
            for k, idx in x.groupby("k").groups.items():
                rows.append({"subject_id": sid, "scale": scale, "k": int(k),
                             "rho": float(spearmanr(x.loc[idx, "er"], mac.loc[idx]).statistic)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "er_avgctrl.csv", index=False)
    med = df.groupby(["scale", "k"]).rho.median().unstack("k").round(3)
    text = ("# Effective rank vs mean average controllability of the drivers (c = 1, random sets)\n\n"
            "Median over subjects of the within-subject Spearman rho across each subject's 100 random sets.\n\n"
            + med.to_markdown() + f"\n\nRange over scales and k: {med.values.min():.2f}-{med.values.max():.2f}\n")
    (out / "er_avgctrl_summary.md").write_text(text)
    print(text)
    ctx.finish(outputs=[out / "er_avgctrl.csv", out / "er_avgctrl_summary.md"])


if __name__ == "__main__":
    main()
