"""Main text: the next slowest modes, damping and the spectral gap (subsec:core_divisions), and the eigenvalues
of the v2-v3 plane (subsec:core_methods).

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .record import Number  # noqa: E402

SCALES = (68, 114, 219)
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
LOC = ("subsec:core_divisions",)
LOC_METHODS = ("subsec:core_methods",)


def fx(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def pc(x: float, d: int = 0) -> str:
    return fx(100 * x, d)


def cnt(x: float) -> str:
    return str(int(round(x)))


@lru_cache(maxsize=None)
def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"k": str}) if path.endswith(("q1_vs_random.csv", "q1_matched.csv", "q1_perp.csv")) \
        else pd.read_csv(path)


def csv(run: Path, rel: str) -> pd.DataFrame:
    return _csv(str((run / rel).resolve()))


def one(df: pd.DataFrame, **eq) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        col = df[k]
        m &= np.isclose(col, v) if isinstance(v, float) else (col == v).to_numpy()
    x = df[m]
    assert len(x) == 1, (eq, len(x))
    return x.iloc[0]


def rng_pair(prefix: str, vals: dict, fmt, source: str, locus, context: dict) -> list[Number]:
    """Two macros (<prefix>Min / <prefix>Max) for a range quoted 'across scales' (or another set)."""
    lo_k = min(vals, key=vals.get)
    hi_k = max(vals, key=vals.get)
    detail = "; values: " + ", ".join(f"{k}: {v:.4g}" for k, v in vals.items())
    return [
        Number(prefix + "Min", float(vals[lo_k]), fmt(vals[lo_k]), source + f" | min over the set (at {lo_k})" + detail,
               tuple(locus), {**context, "end": "min"}),
        Number(prefix + "Max", float(vals[hi_k]), fmt(vals[hi_k]), source + f" | max over the set (at {hi_k})" + detail,
               tuple(locus), {**context, "end": "max"}),
    ]


def collect(run: Path) -> list[Number]:
    run = Path(run)
    out: list[Number] = []

    # ------------------------------------------------------------------ paragraph 1: v2 and the v2-v3 subspace
    # "The second eigenvector, v2, placed a median of 83--95% of its squared weight on the 20% of regions with the
    #  highest v1 loadings."
    ev = csv(run, "mechanisms/eigvec_subjects.csv")
    med = ev.groupby("scale").v2_on_v1set.median().to_dict()
    out += rng_pair("MVTwoOnCorePct", med, pc,
                    "mechanisms/04_leading_eigenvectors: mechanisms/eigvec_subjects.csv | column v2_on_v1set "
                    "(sum of v2(i)^2 over the subject's top 20% regions by v1), median over 70 subjects per scale, as %",
                    LOC, {"stat": "median over subjects", "set": "3 scales"})

    # Methods: "The second and third eigenvalues were close (median l2 - l3 = 0.15--0.18 across scales"
    out += rng_pair("MLTwoLThreeMedian", ev.groupby("scale").l2_l3.median().to_dict(), fx,
                    "mechanisms/04_leading_eigenvectors: mechanisms/eigvec_subjects.csv | column l2_l3 (l2 - l3 of A0), "
                    "median over 70 subjects per scale", LOC_METHODS, {"stat": "median over subjects", "set": "3 scales"})

    # ------------------------------------------------------------------ paragraph 2: projecting v1 out (Fig. 4B)
    # "We assessed the contribution of v1 by removing it from each controllability Gramian and recalculating q1. This
    #  reduced the difference between degree-strength and cycle-participation sets by 52--69% for two to five drivers."
    p3 = csv(run, "concentration/q1_perp.csv")
    fr = {s: one(p3, scale=s, c=0.1, k="avg2+", metric="q1").fraction_removed for s in SCALES}
    out += rng_pair("MPerpFractionRemovedQOneKTwoPlusPct", fr, pc,
                    "concentration/02_tables: concentration/q1_perp.csv | metric == 'q1', c == 0.1, k == 'avg2+' (k in 2,3,5), "
                    "column fraction_removed = 1 - perp_mean/full_mean (cycle - degree, mean over subjects), as %",
                    LOC, {"c": 0.1, "k": "2,3,5", "comparison": "cycle - degree", "set": "3 scales"})

    # "However, degree-strength sets remained more concentrated than random sets in all 70 subjects (Fig. 4B)"
    p1 = csv(run, "concentration/q1_vs_random.csv")
    rows = {s: one(p1, family="degree_strength", scale=s, c=0.1, k="avg2+", metric="q1_perp") for s in SCALES}
    src = ("concentration/02_tables: concentration/q1_vs_random.csv | family == 'degree_strength', metric == 'q1_perp', "
           "c == 0.1, k == 'avg2+' (k in 2,3,5); column {}")
    npos = min(r["n_positive"] for r in rows.values())
    out.append(Number("MPerpDegreeExcessNPositive", float(npos), cnt(npos),
                      src.format("n_positive, min over scales (" + ", ".join(f"{s}: {int(r['n_positive'])}" for s, r in rows.items())
                                 + "); text 'in all 70 subjects'"),
                      LOC, {"c": 0.1, "k": "2,3,5", "stat": "count of 70"}))

    # ------------------------------------------------------------------ paragraph 3: decay rates and the gap
    # "At c=0.1, v1 decayed approximately five to six times more slowly than v2" (decay rates at the median gap)
    ss = csv(run, "mechanisms/second_subspace_subjects.csv")
    gap = ss.groupby("scale").gap.median().to_dict()
    src_gap = ("mechanisms/05_second_subspace: mechanisms/second_subspace_subjects.csv | column gap = l1 - l2 "
               "of A0, median over 70 subjects per scale (identical to concentration/decay_flips.csv gap)")
    r01 = {s: (0.1 + g) / 0.1 for s, g in gap.items()}
    lo, hi = min(r01.values()), max(r01.values())
    src_r = src_gap + " | decay-rate ratio (c + gap_median) / c at c = {c}; per scale: {vals}"
    vals01 = ", ".join(f"{s}: {v:.2f}" for s, v in r01.items())
    out.append(Number("MDecayRatioCTenthLow", lo, WORD[round(lo)], src_r.format(c=0.1, vals=vals01) + "; min, spelled-out rounded integer",
                      LOC, {"c": 0.1, "end": "min"}))
    out.append(Number("MDecayRatioCTenthHigh", hi, WORD[round(hi)], src_r.format(c=0.1, vals=vals01) + "; max, spelled-out rounded integer",
                      LOC, {"c": 0.1, "end": "max"}))

    # ------------------------------------------------------------------ paragraph 4: damping sweep (Fig. 4C)
    # "The excess q1 of degree-strength sets over random sets was greatest under weaker damping, reaching 0.35--0.40
    #  near c=0.01, compared with 0.21--0.25 at c=0.1"; "the excess declined to 0.30--0.38 at c=0.003 and 0.03--0.06
    #  at c=1"
    def dr(s, c):
        return one(p1, family="degree_strength", scale=s, c=c, k="avg", metric="q1")

    src_dr = ("concentration/02_tables: concentration/q1_vs_random.csv | family == 'degree_strength', metric == 'q1', k == 'avg' "
              "(mean over k in 1,2,3,5 of criterion q1 minus the subject's random mean), c == {c}; column 'mean' (over 70 "
              "subjects). Matches decay_curves.csv.gz dr_q1 means plotted in Fig. 4C")
    for c, tag in ((0.1, "Tenth"), (1.0, "One")):
        out += rng_pair(f"MDegExcessQOneC{tag}", {s: dr(s, c)["mean"] for s in SCALES}, fx, src_dr.format(c=c),
                        LOC, {"c": c, "k": "1,2,3,5", "comparison": "degree vs random mean", "set": "3 scales"})
    # The low-damping end of the Fig. 4C curves (25 values of c from 10^-2.5 to 10^0.5)
    dc = csv(run, "concentration/decay_curves.csv.gz")
    curve = dc.groupby(["scale", "c"]).dr_q1.mean()
    src_dc = ("concentration/03_decay_collapse: concentration/decay_curves.csv.gz | column dr_q1 (degree-strength q1 minus "
              "the subject's random mean, mean over k in 1,2,3,5), mean over 70 subjects per scale and c (the Fig. 4C "
              "line); {}")
    peak = curve.groupby("scale").max().to_dict()
    at_peak = {s: float(curve.loc[s].idxmax()) for s in SCALES}
    out += rng_pair("MDegExcessQOnePeak", peak, fx,
                    src_dc.format("maximum over c of the cohort-mean curve, per scale (at c = "
                                  + ", ".join(f"{s}: {c:.4f}" for s, c in at_peak.items()) + ")"),
                    LOC, {"c": "curve maximum", "k": "1,2,3,5", "comparison": "degree vs random mean", "set": "3 scales"})
    c_low = float(dc.c.min())
    assert np.isclose(c_low, 10 ** -2.5)
    out += rng_pair("MDegExcessQOneLowestC", curve.xs(c_low, level="c").to_dict(), fx,
                    src_dc.format("c == 10^-2.5 (the smallest c of the grid, printed 0.003)"),
                    LOC, {"c": c_low, "k": "1,2,3,5", "comparison": "degree vs random mean", "set": "3 scales"})

    # ------------------------------------------------------------------ paragraph 5: gap vs excess across subjects (Fig. 4D)
    # "Subjects with larger gaps showed greater excess q1 ... (Spearman rho=0.53--0.67 across scales; Fig. 4D)"
    gc = csv(run, "concentration/decay_gap_correlations.csv")
    out += rng_pair("MGapRhoDegExcess", {s: one(gc, scale=s, c=0.1, curve="dr_q1").rho_gap for s in SCALES}, fx,
                    "concentration/03_decay_collapse: concentration/decay_gap_correlations.csv | c == 0.1, curve == 'dr_q1', "
                    "column rho_gap (Spearman across 70 subjects, gap vs k-averaged degree - random q1); equals the rho "
                    "printed in Fig. 4D (fig4.py recomputes it from decay_curves.csv.gz)",
                    LOC, {"c": 0.1, "stat": "Spearman across subjects", "set": "3 scales"})
    # "This relationship persisted after controlling for mean strength, edge density, and strength variability
    #  (partial rho=0.54--0.82)"
    pcr = csv(run, "concentration/decay_gap_partial_correlations.csv")
    full = pcr[pcr.controls == "strength+density+cv"]
    out += rng_pair("MGapPartialRhoDegExcess",
                    {s: one(full, scale=s, c=0.1, curve="dr_q1").partial_rho_gap for s in SCALES}, fx,
                    "concentration/03_decay_collapse: concentration/decay_gap_partial_correlations.csv | controls == "
                    "'strength+density+cv', c == 0.1, curve == 'dr_q1', column partial_rho_gap",
                    LOC, {"c": 0.1, "stat": "partial Spearman across subjects", "set": "3 scales"})
    return out


# Tokens as printed in the manuscript.
PRINTED = {
    "MVTwoOnCorePctMin": "83",
    "MVTwoOnCorePctMax": "95",
    "MLTwoLThreeMedianMin": "0.15",
    "MLTwoLThreeMedianMax": "0.18",
    "MPerpFractionRemovedQOneKTwoPlusPctMin": "52",
    "MPerpFractionRemovedQOneKTwoPlusPctMax": "69",
    "MPerpDegreeExcessNPositive": "70",
    "MDecayRatioCTenthLow": "five",
    "MDecayRatioCTenthHigh": "six",
    "MDegExcessQOnePeakMin": "0.35",
    "MDegExcessQOnePeakMax": "0.40",
    "MDegExcessQOneCTenthMin": "0.21",
    "MDegExcessQOneCTenthMax": "0.25",
    "MDegExcessQOneLowestCMin": "0.30",
    "MDegExcessQOneLowestCMax": "0.38",
    "MDegExcessQOneCOneMin": "0.03",
    "MDegExcessQOneCOneMax": "0.06",
    "MGapRhoDegExcessMin": "0.53",
    "MGapRhoDegExcessMax": "0.67",
    "MGapPartialRhoDegExcessMin": "0.54",
    "MGapPartialRhoDegExcessMax": "0.82",
}
