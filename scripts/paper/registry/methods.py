"""Results quoted in the Methods (solver agreement, null fidelity, isolated regions, Rayleigh-bound fit).

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

SCALES = (68, 114, 219)


from .record import Number  # noqa: E402

# ----------------------------------------------------------------------------- helpers

def fx(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ceil_dec(x: float, d: int) -> float:
    """Round up to d decimals (for 'within'/'at most' bounds)."""
    f = 10.0 ** d
    return math.ceil(round(x * f, 9)) / f


def pow10_ceil(x: float) -> str:
    """Next power of ten at or above x, as ``10^{e}``."""
    return f"10^{{{math.ceil(math.log10(x))}}}"


def _mk(name, value, computed, printed, source, locus, context, bound=None):
    """Build a Number with the computed text; the token printed in the manuscript goes to context["printed"]."""
    ctx = dict(context)
    if bound is not None:  # bound = (holds: bool, description)
        ctx["bound"] = bound[1]
        ctx["bound_holds"] = bool(bound[0])
    ctx["printed"] = printed
    return Number(name, float(value), computed, source, tuple(locus), ctx)


# ----------------------------------------------------------------------------- collect

def collect(run: Path) -> list[Number]:
    run = Path(run)
    rev = run
    out: list[Number] = []

    # ---- subsec:model: "All connectomes were connected except that of one subject, which contained a
    #      single isolated region at each scale"; subsec:perron: "(excluding the single zero-strength region)"
    #      Recomputed from processed/ (connected components of A > 0).
    n_disc, n_iso, comp = {}, {}, {}
    for s in SCALES:
        A = np.load(run / "processed" / f"connectomes_scale-{s}.npz")["adjacency"]
        disc = []
        for i, a in enumerate(A):
            nc, lab = connected_components(a > 0, directed=False)
            if nc > 1:
                disc.append((i, sorted(np.bincount(lab).tolist())))
        n_disc[s] = len(disc)
        n_iso[s] = int((A.sum(2) == 0).sum())
        comp[s] = disc
    one_subject = len({tuple(i for i, _ in comp[s]) for s in SCALES}) == 1
    src = ("00_prepare_data: processed/connectomes_scale-<N>.npz['adjacency'] | subjects whose graph (A > 0) has >1 "
           f"connected component per scale {n_disc}; components {comp}")
    words = {1: "one"}
    out.append(_mk("MDisconnectedSubjects", n_disc[68], words.get(n_disc[68], str(n_disc[68])) if one_subject else "?",
                   "one", src, ("subsec:model",),
                   {"stat": "count of subjects", "scales": "68/114/219 (same subject at every scale)"}))
    per_subj_iso = {s: [c[0] for _, c in comp[s]] for s in SCALES}  # size of smallest component
    iso_ok = all(n_iso[s] == 1 and per_subj_iso[s] == [1] for s in SCALES)
    out.append(_mk("MIsolatedRegionCount", 1 if iso_ok else n_iso[68], "single" if iso_ok else str(n_iso[68]),
                   "single",
                   src + f"; zero-strength regions per scale {n_iso} (all in that subject)",
                   ("subsec:model", "subsec:perron"),
                   {"stat": "number of isolated (zero-strength) regions in the disconnected subject, each scale"}))

    # ---- methods:gramian: "This solution agreed with a direct Lyapunov solver to within 10^{-13}."
    v = json.loads((rev / "sensitivity" / "validation.json").read_text())["library_vs_eigbasis"]
    worst = max(v["max_abs_W"], v["rel_W"])
    out.append(_mk("MLyapunovAgreement", worst, pow10_ceil(worst), "10^{-13}",
                   f"sensitivity/00_validate: sensitivity/validation.json | library_vs_eigbasis max(max_abs_W="
                   f"{v['max_abs_W']:.2g}, rel_W={v['rel_W']:.2g}); next power of ten",
                   ("methods:gramian",), {"comparison": "closed-form eigenbasis Gramian vs scipy Lyapunov solver"},
                   bound=(worst <= 1e-13, "<=")))

    # ---- subsec:nulls: "mean correlation with the original strengths 0.998 to 0.999"
    nf = pd.read_csv(rev / "null_models" / "null_fidelity.csv")
    sr = nf[nf.null == "strength"].set_index("scale").strength_r
    src = ("null_models/02_analyze: null_models/null_fidelity.csv | null == 'strength', strength_r "
           "(mean over subjects x 50 nulls of corr(null, real strengths)), {} over scales, 3 dp")
    for nm, val, ms in (("MStrengthNullFidelityMin", sr.min(), "0.998"), ("MStrengthNullFidelityMax", sr.max(), "0.999")):
        out.append(_mk(nm, val, fx(val, 3), ms, src.format("min" if nm.endswith("Min") else "max"), ("subsec:nulls",),
                       {"null": "strength", "stat": "mean over subjects and nulls", "per_scale": sr.round(4).to_dict()}))

    # ---- subsec:nulls: "mean connection length within 1\% of the original"
    real = nf[nf.null == "real"].set_index("scale").mean_edge_length
    geo = nf[nf.null == "geometric"].set_index("scale").mean_edge_length
    rel = float((100 * (geo / real - 1)).abs().max())
    out.append(_mk("MGeomNullEdgeLengthBound", rel, fx(ceil_dec(rel, 0), 0), "1",
                   "null_models/02_analyze: null_models/null_fidelity.csv | max over scales of "
                   "100*|geometric/real - 1| of mean_edge_length (cohort means), percent",
                   ("subsec:nulls",), {"null": "geometric", "stat": "cohort-mean edge length, max over scales"},
                   bound=(rel < 1, "<")))

    # ---- subsec:nulls: "(median Pearson r = 0.998; minimum 0.95)" -- r_W vs s'A0 s across strength nulls
    ps = pd.read_csv(rev / "null_models" / "rayleigh" / "per_subject.csv")
    pr = ps[ps.null == "strength"].pearson_rW_sTWs
    src = ("null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == 'strength', "
           "pearson_rW_sTWs, {} over all subjects x scales (as scripts/paper/sections/s8.py inline_values)")
    out.append(_mk("MAssortRayleighPearsonMedian", pr.median(), fx(pr.median(), 3), "0.998", src.format("median"),
                   ("subsec:nulls",), {"null": "strength", "stat": "median over 70 subjects x 3 scales"}))
    out.append(_mk("MAssortRayleighPearsonMin", pr.min(), fx(pr.min(), 2), "0.95", src.format("min"),
                   ("subsec:nulls",), {"null": "strength", "stat": "minimum over 70 subjects x 3 scales"}))
    names = [n.name for n in out]
    assert len(names) == len(set(names))
    return out
