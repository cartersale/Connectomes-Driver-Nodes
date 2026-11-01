"""Main text: the next slowest modes, damping and the spectral gap (subsec:core_divisions), and the caption
of Fig. 4.

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .record import Number  # noqa: E402

SCALES = (68, 114, 219)
SW = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
LOC = ("subsec:core_divisions",)
LOC_FIG = ("subsec:core_divisions", "fig:core_divisions")


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

    # "A front--back axis identified from the v2--v3 subspace across subjects was represented most strongly by v2 in
    #  43--48 of 70 subjects and by v3 in another 17--22 (Fig. 4A)."
    ss = csv(run, "mechanisms/second_subspace_subjects.csv")
    src_ab = ("mechanisms/05_second_subspace: mechanisms/second_subspace_subjects.csv | a_best == {} "
              "(which of the subject's v2..v6 has the largest cos^2 with consensus axis a of the mean v2-v3 projector), "
              "count of subjects per scale (same counts as Fig. 4A right, scripts/figures/fig4.py)")
    out += rng_pair("MFrontBackOnVTwoCount", (ss.a_best == 2).groupby(ss.scale).sum().to_dict(), cnt, src_ab.format(2),
                    LOC_FIG, {"stat": "count of 70", "set": "3 scales", "eigenvector": "v2"})
    out += rng_pair("MFrontBackOnVThreeCount", (ss.a_best == 3).groupby(ss.scale).sum().to_dict(), cnt, src_ab.format(3),
                    LOC_FIG, {"stat": "count of 70", "set": "3 scales", "eigenvector": "v3"})

    # ------------------------------------------------------------------ paragraph 2: projecting v1 out (Fig. 4B)
    # "Removing v1 from each controllability Gramian reduced the difference in q1 between degree-strength and
    #  cycle-participation sets by 52--69% for two to five drivers."
    p3 = csv(run, "concentration/q1_perp.csv")
    fr = {s: one(p3, scale=s, c=0.1, k="avg2+", metric="q1").fraction_removed for s in SCALES}
    out += rng_pair("MPerpFractionRemovedQOneKTwoPlusPct", fr, pc,
                    "concentration/02_tables: concentration/q1_perp.csv | metric == 'q1', c == 0.1, k == 'avg2+' (k in 2,3,5), "
                    "column fraction_removed = 1 - perp_mean/full_mean (cycle - degree, mean over subjects), as %",
                    LOC_FIG, {"c": 0.1, "k": "2,3,5", "comparison": "cycle - degree", "set": "3 scales"})

    # "Nevertheless, degree-strength sets remained more concentrated than random sets (by 0.10--0.15, in all 70
    #  subjects; Fig. 4B)."
    p1 = csv(run, "concentration/q1_vs_random.csv")
    rows = {s: one(p1, family="degree_strength", scale=s, c=0.1, k="avg2+", metric="q1_perp") for s in SCALES}
    src = ("concentration/02_tables: concentration/q1_vs_random.csv | family == 'degree_strength', metric == 'q1_perp', "
           "c == 0.1, k == 'avg2+' (k in 2,3,5); column {}")
    out += rng_pair("MPerpDegreeExcessQOne", {s: r["mean"] for s, r in rows.items()}, fx,
                    src.format("mean (mean over subjects of degree q1(W_perp) minus the subject's random-set mean)"),
                    LOC_FIG, {"c": 0.1, "k": "2,3,5", "comparison": "degree vs random mean", "set": "3 scales"})
    npos = min(r["n_positive"] for r in rows.values())
    out.append(Number("MPerpDegreeExcessNPositive", float(npos), cnt(npos),
                      src.format("n_positive, min over scales (" + ", ".join(f"{s}: {int(r['n_positive'])}" for s, r in rows.items())
                                 + "); text 'in all 70 subjects'"),
                      LOC, {"c": 0.1, "k": "2,3,5", "stat": "count of 70"}))

    # ------------------------------------------------------------------ paragraph 3: decay rates and the gap
    # "At the median gap (0.43--0.53 across scales), v1 decayed approximately five to six times more slowly than v2
    #  at c=0.1, compared with approximately 1.5 times at c=1."
    gap = ss.groupby("scale").gap.median().to_dict()
    src_gap = ("mechanisms/05_second_subspace: mechanisms/second_subspace_subjects.csv | column gap = l1 - l2 "
               "of A0, median over 70 subjects per scale (identical to concentration/decay_flips.csv gap)")
    out += rng_pair("MGapMedian", gap, fx, src_gap, LOC, {"stat": "median over subjects", "set": "3 scales"})
    r01 = {s: (0.1 + g) / 0.1 for s, g in gap.items()}
    lo, hi = min(r01.values()), max(r01.values())
    src_r = src_gap + " | decay-rate ratio (c + gap_median) / c at c = {c}; per scale: {vals}"
    vals01 = ", ".join(f"{s}: {v:.2f}" for s, v in r01.items())
    out.append(Number("MDecayRatioCTenthLow", lo, WORD[round(lo)], src_r.format(c=0.1, vals=vals01) + "; min, spelled-out rounded integer",
                      LOC, {"c": 0.1, "end": "min"}))
    out.append(Number("MDecayRatioCTenthHigh", hi, WORD[round(hi)], src_r.format(c=0.1, vals=vals01) + "; max, spelled-out rounded integer",
                      LOC, {"c": 0.1, "end": "max"}))
    r1 = {s: (1.0 + g) / 1.0 for s, g in gap.items()}
    v = float(np.mean(list(r1.values())))
    out.append(Number("MDecayRatioCOne", v, fx(v, 1),
                      src_r.format(c=1, vals=", ".join(f"{s}: {x:.2f}" for s, x in r1.items()))
                      + "; text gives one approximate value: mean over scales, 1 decimal (per-scale 1.43-1.53)",
                      LOC, {"c": 1.0, "stat": "mean over scales"}))

    # ------------------------------------------------------------------ paragraph 4: damping sweep (Fig. 4C)
    # "the excess q1 of degree-strength sets over random sets decreased from 0.34--0.39 at c=0.01 to 0.21--0.25 at
    #  c=0.1 and 0.03--0.06 at c=1."
    def dr(s, c):
        return one(p1, family="degree_strength", scale=s, c=c, k="avg", metric="q1")

    src_dr = ("concentration/02_tables: concentration/q1_vs_random.csv | family == 'degree_strength', metric == 'q1', k == 'avg' "
              "(mean over k in 1,2,3,5 of criterion q1 minus the subject's random mean), c == {c}; column 'mean' (over 70 "
              "subjects). Matches decay_curves.csv.gz dr_q1 means plotted in Fig. 4C")
    for c, tag in ((0.01, "Hundredth"), (0.1, "Tenth"), (1.0, "One")):
        out += rng_pair(f"MDegExcessQOneC{tag}", {s: dr(s, c)["mean"] for s in SCALES}, fx, src_dr.format(c=c),
                        LOC, {"c": c, "k": "1,2,3,5", "comparison": "degree vs random mean", "set": "3 scales"})
    # "Degree-strength sets nevertheless exceeded the random mean in 62--70 of 70 subjects at c=1."
    out += rng_pair("MDegExcessQOneCOneNPos", {s: dr(s, 1.0)["n_positive"] for s in SCALES}, cnt,
                    src_dr.format(c=1.0).replace("column 'mean' (over 70 subjects)", "column n_positive"),
                    LOC, {"c": 1.0, "k": "1,2,3,5", "stat": "count of 70", "set": "3 scales"})

    # "The difference between degree-strength and cycle-participation sets declined similarly, with both effects about
    #  four to eight times smaller at c=1 than at c=0.1."
    p2 = csv(run, "concentration/q1_matched.csv")

    def cd(s, c, m):
        return one(p2, comparison="cycle - degree", scale=s, c=c, k="avg", metric=m)

    ratios = {}
    for s in SCALES:
        ratios[f"deg-rand {s}"] = dr(s, 0.1)["mean"] / dr(s, 1.0)["mean"]
        ratios[f"cyc-deg {s}"] = cd(s, 0.1, "q1")["mean"] / cd(s, 1.0, "q1")["mean"]
    src_ratio = ("concentration/02_tables: concentration/q1_vs_random.csv (degree - random q1) and concentration/q1_matched.csv "
                 "(comparison == 'cycle - degree', metric == 'q1'), k == 'avg', column 'mean' | ratio c=0.1 / c=1 over both "
                 "effects x 3 scales; spelled-out rounded integer")
    lo_k, hi_k = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
    det = "; ratios: " + ", ".join(f"{k}: {v:.2f}" for k, v in ratios.items())
    out.append(Number("MDampingShrinkMin", ratios[lo_k], WORD[round(ratios[lo_k])], src_ratio + f" | min ({lo_k})" + det,
                      LOC, {"end": "min"}))
    out.append(Number("MDampingShrinkMax", ratios[hi_k], WORD[round(ratios[hi_k])], src_ratio + f" | max ({hi_k})" + det,
                      LOC, {"end": "max"}))

    # ------------------------------------------------------------------ paragraph 5: gap vs excess across subjects (Fig. 4D)
    # "subjects with larger gaps showed greater excess q1 ... (Spearman rho=0.53--0.67 across scales; Fig. 4D)."
    gc = csv(run, "concentration/decay_gap_correlations.csv")
    out += rng_pair("MGapRhoDegExcess", {s: one(gc, scale=s, c=0.1, curve="dr_q1").rho_gap for s in SCALES}, fx,
                    "concentration/03_decay_collapse: concentration/decay_gap_correlations.csv | c == 0.1, curve == 'dr_q1', "
                    "column rho_gap (Spearman across 70 subjects, gap vs k-averaged degree - random q1); equals the rho "
                    "printed in Fig. 4D (fig4.py recomputes it from decay_curves.csv.gz)",
                    LOC_FIG, {"c": 0.1, "stat": "Spearman across subjects", "set": "3 scales"})
    # "These relationships remained after controlling for mean strength, edge density, and strength variability
    #  (partial rho=0.54--0.82)."
    pcr = csv(run, "concentration/decay_gap_partial_correlations.csv")
    full = pcr[pcr.controls == "strength+density+cv"]
    out += rng_pair("MGapPartialRhoDegExcess",
                    {s: one(full, scale=s, c=0.1, curve="dr_q1").partial_rho_gap for s in SCALES}, fx,
                    "concentration/03_decay_collapse: concentration/decay_gap_partial_correlations.csv | controls == "
                    "'strength+density+cv', c == 0.1, curve == 'dr_q1', column partial_rho_gap",
                    LOC, {"c": 0.1, "stat": "partial Spearman across subjects", "set": "3 scales"})
    # "Partial correlations were positive throughout c=0.03--1 (rho=0.38--0.83, all p<=0.002)."
    r = full[(full.curve == "dr_q1") & (full.c >= 10 ** -1.5 - 1e-9) & (full.c <= 1 + 1e-9)]
    assert len(r) == 39
    src_rr = ("concentration/03_decay_collapse: concentration/decay_gap_partial_correlations.csv | controls == "
              "'strength+density+cv', curve == 'dr_q1', the 13 grid values 10^-1.5 <= c <= 1 x 3 scales; {}")
    out.append(Number("MGapPartialRhoRangeMin", float(r.partial_rho_gap.min()), fx(r.partial_rho_gap.min()),
                      src_rr.format("min partial_rho_gap"), LOC, {"end": "min", "c": "0.03-1"}))
    out.append(Number("MGapPartialRhoRangeMax", float(r.partial_rho_gap.max()), fx(r.partial_rho_gap.max()),
                      src_rr.format("max partial_rho_gap"), LOC, {"end": "max", "c": "0.03-1"}))
    pmax = float(r.p.max())
    bound = math.ceil(pmax * 1000) / 1000
    assert pmax <= bound and (r.partial_rho_gap > 0).all()
    out.append(Number("MGapPartialRhoRangePMax", pmax, fx(bound, 3),
                      src_rr.format(f"max p = {pmax:.4g}; text states the bound 'all p <= 0.002' -> max p rounded up to 3 decimals"),
                      LOC, {"stat": "bound", "c": "0.03-1"}))

    # ------------------------------------------------------------------ paragraph 6: effective rank vs damping
    # "Cycle-participation sets had higher effective rank than degree-strength sets at c=0.01 and c=0.1 (differences
    #  of 0.80--1.73)"
    src_er = ("concentration/02_tables: concentration/q1_matched.csv | comparison == 'cycle - degree', metric == 'er', k == 'avg' "
              "(mean over k in 1,2,3,5), c == {c}; column {col}")
    er_lo = {f"{s}, c={c}": cd(s, c, "er")["mean"] for s in SCALES for c in (0.01, 0.1)}
    out += rng_pair("MCycMinusDegERLowDamping", er_lo, fx, src_er.format(c="0.01 and 0.1", col="mean (over 70 subjects)"),
                    LOC, {"c": "0.01, 0.1", "k": "1,2,3,5", "comparison": "cycle - degree", "set": "3 scales x 2 c"})
    # "but the difference approached zero or reversed at c=1 (-0.02 to -0.24; positive in 4--34 of 70 subjects)."
    er1 = {s: cd(s, 1.0, "er")["mean"] for s in SCALES}
    # text range runs from the value nearest zero (-0.02) to the most negative (-0.24): Max then Min
    out += rng_pair("MCycMinusDegERCOne", er1, fx, src_er.format(c=1.0, col="mean (over 70 subjects)")
                    + "; text '-0.02 to -0.24' prints the max first", LOC,
                    {"c": 1.0, "k": "1,2,3,5", "comparison": "cycle - degree", "set": "3 scales"})
    out += rng_pair("MCycMinusDegERCOneNPos", {s: cd(s, 1.0, "er")["n_positive"] for s in SCALES}, cnt,
                    src_er.format(c=1.0, col="n_positive"), LOC, {"c": 1.0, "stat": "count of 70", "set": "3 scales"})
    # "The mean difference crossed zero at approximately c=0.88, 0.69, and 0.29 for 68, 114, and 219 regions"
    sf = csv(run, "sensitivity/csweep_signflip.csv")
    for s in SCALES:
        v = float(one(sf, metric="effective_rank", scale=s).c_flip_mean)
        out.append(Number(f"MCycMinusDegERFlipC{SW[s]}", v, fx(v),
                          "sensitivity/05_summarize_csweep: sensitivity/csweep_signflip.csv | metric == 'effective_rank', "
                          "column c_flip_mean (interpolated zero crossing of the cohort-mean cycle - degree effective-rank "
                          "curve over the c sweep of sensitivity/01_csweep_perron.py)",
                          LOC, {"scale": s, "comparison": "cycle - degree", "metric": "effective rank"}))
    # "At c=1, effective rank closely tracked the mean single-node average controllability of the drivers (median
    #  within-subject rho=0.91--0.99 across random sets of each size)"
    ea = csv(run, "concentration/er_avgctrl.csv")
    medr = ea.groupby(["scale", "k"]).rho.median()
    out += rng_pair("MERAvgCtrlRho", {f"{s}, k={k}": v for (s, k), v in medr.items()}, fx,
                    "concentration/06_er_avgctrl: concentration/er_avgctrl.csv | median over subjects of rho (within-subject "
                    "Spearman across the 100 random sets, ER vs mean driver average controllability, c = 1), per scale x k "
                    "in 1,2,3,5",
                    LOC, {"c": 1.0, "stat": "median over subjects", "set": "3 scales x 4 k"})
    return out


# Tokens as printed in the manuscript.
PRINTED = {
    "MVTwoOnCorePctMin": "83",
    "MVTwoOnCorePctMax": "95",
    "MFrontBackOnVTwoCountMin": "43",
    "MFrontBackOnVTwoCountMax": "48",
    "MFrontBackOnVThreeCountMin": "17",
    "MFrontBackOnVThreeCountMax": "22",
    "MPerpFractionRemovedQOneKTwoPlusPctMin": "52",
    "MPerpFractionRemovedQOneKTwoPlusPctMax": "69",
    "MPerpDegreeExcessQOneMin": "0.10",
    "MPerpDegreeExcessQOneMax": "0.15",
    "MPerpDegreeExcessNPositive": "70",
    "MGapMedianMin": "0.43",
    "MGapMedianMax": "0.53",
    "MDecayRatioCTenthLow": "five",
    "MDecayRatioCTenthHigh": "six",
    "MDecayRatioCOne": "1.5",
    "MDegExcessQOneCHundredthMin": "0.34",
    "MDegExcessQOneCHundredthMax": "0.39",
    "MDegExcessQOneCTenthMin": "0.21",
    "MDegExcessQOneCTenthMax": "0.25",
    "MDegExcessQOneCOneMin": "0.03",
    "MDegExcessQOneCOneMax": "0.06",
    "MDegExcessQOneCOneNPosMin": "62",
    "MDegExcessQOneCOneNPosMax": "70",
    "MDampingShrinkMin": "four",
    "MDampingShrinkMax": "eight",
    "MGapRhoDegExcessMin": "0.53",
    "MGapRhoDegExcessMax": "0.67",
    "MGapPartialRhoDegExcessMin": "0.54",
    "MGapPartialRhoDegExcessMax": "0.82",
    "MGapPartialRhoRangeMin": "0.38",
    "MGapPartialRhoRangeMax": "0.83",
    "MGapPartialRhoRangePMax": "0.002",
    "MCycMinusDegERLowDampingMin": "0.80",
    "MCycMinusDegERLowDampingMax": "1.73",
    "MCycMinusDegERCOneMin": "-0.24",
    "MCycMinusDegERCOneMax": "-0.02",
    "MCycMinusDegERCOneNPosMin": "4",
    "MCycMinusDegERCOneNPosMax": "34",
    "MCycMinusDegERFlipCSixtyEight": "0.88",
    "MCycMinusDegERFlipCOneFourteen": "0.69",
    "MCycMinusDegERFlipCTwoNineteen": "0.29",
    "MERAvgCtrlRhoMin": "0.91",
    "MERAvgCtrlRhoMax": "0.99",
}
