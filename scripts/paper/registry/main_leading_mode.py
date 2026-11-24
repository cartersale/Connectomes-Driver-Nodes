"""Main text: single drivers, larger degree-strength sets and cycle participation (subsec:hub_concentration), the
leading mode (subsec:leading_mode), and the caption of Fig. 3.

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._common import scale_siblings
from .record import Number

SCALES = (68, 114, 219)
NAN = float("nan")
HUB3 = ("degree_strength", "eigenvector_centrality", "top_avg_ctrl")
_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def fx(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def pc(x: float, d: int = 0) -> str:
    return fx(100 * x, d)


def cnt(x: float) -> str:
    return str(int(round(x)))


def ceil_d(x: float, d: int) -> str:
    """Upper bound ("within x", "below x"): round up so the statement stays true."""
    f = 10**d
    return fx(math.ceil(x * f - 1e-9) / f, d)


@lru_cache(maxsize=None)
def _read(path: str, usecols: tuple | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(usecols) if usecols else None)


def one(df: pd.DataFrame, **eq) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        col = df[k]
        m &= np.isclose(col, v) if isinstance(v, float) else (col == v).to_numpy()
    x = df[m]
    assert len(x) == 1, (eq, len(x))
    return x.iloc[0]


def kavg(d: pd.DataFrame, col: str, ks) -> pd.DataFrame:
    """Mean over draws within (family, scale, subject, k), then over k, then over subjects."""
    d = d[d.k.isin(ks)]
    return (d.groupby(["family", "scale", "subject_id", "k"])[col].mean()
            .groupby(["family", "scale", "subject_id"]).mean()
            .groupby(["family", "scale"]).mean().unstack())


def rng(name, vals, fmt, source, locus, context) -> list[Number]:
    """Two Numbers (Min / Max) for a range over the values of a dict / Series."""
    v = pd.Series(vals, dtype=float)
    lo, hi = v.idxmin(), v.idxmax()
    return [Number(name + "Min", float(v[lo]), fmt(v[lo]), source + f" | min over {list(v.index)} (at {lo})", locus, context),
            Number(name + "Max", float(v[hi]), fmt(v[hi]), source + f" | max over {list(v.index)} (at {hi})", locus, context),
            *scale_siblings(name, list(vals.items()), fmt, source, locus, context)]


def abs_reach(ga: pd.DataFrame) -> pd.DataFrame:
    """Absolute reachability of each named set relative to its subject's random sets.

    Per subject, the set's value over the mean of its 100 random sets, then the median over subjects, for
    lam1 (most reachable pattern), nonlead = tr W - lam1 (all other patterns) and tr_perp = tr W - v1^T W v1
    (outside v1). Index (family, scale, k). Same computation as the summary in mechanisms/08_gramian_alignment.py.
    """
    d = ga.assign(nonlead=ga.trace - ga.lam1)
    cols = ["lam1", "nonlead", "tr_perp"]
    rand = d[d.family == "random"].groupby(["scale", "k", "subject_id"])[cols].mean()
    h = d[d.family != "random"].set_index(["family", "scale", "k", "subject_id"])[cols]
    ratio = h / rand.reindex(h.index.droplevel("family")).to_numpy()
    return ratio.groupby(["family", "scale", "k"]).median()


HC = ("subsec:hub_concentration",)
LM = ("subsec:leading_mode",)
F3 = ("fig:midline_core",)


def collect(run: Path) -> list[Number]:
    run = Path(run)
    out: list[Number] = []

    # subsec:hub_concentration, last two paragraphs
    p1 = _read(str(run / "concentration" / "q1_vs_random.csv"))
    p2 = _read(str(run / "concentration" / "q1_matched.csv"))
    P1 = "concentration/02_tables: concentration/q1_vs_random.csv | c == 0.1, metric == 'q1', family == '{f}', k == '{k}', column {col}"
    P2 = "concentration/02_tables: concentration/q1_matched.csv | comparison == '{cmp}', c == 0.1, k == '{k}', metric == '{m}', column {col}"

    def p1v(fam, k, col):
        return {s: float(one(p1, family=fam, scale=s, c=0.1, k=k, metric="q1")[col]) for s in SCALES}

    def p2v(cmp, m, col, k="avg"):
        return {s: float(one(p2, comparison=cmp, scale=s, c=0.1, k=k, metric=m)[col]) for s in SCALES}

    ctx = {"k": 1, "c": 0.1, "stat": "mean over 70 subjects", "scales": "all"}
    # "hub-oriented criteria produced less concentrated distributions than random selection (87% for degree strength
    #  compared with 95% for random regions"
    out += rng("MQOneSingleDegreePct", p1v("degree_strength", "1", "value_mean"), pc,
               P1.format(f="degree_strength", k="1", col="value_mean") + " x 100", HC, {**ctx, "criterion": "degree_strength"})
    out += rng("MQOneSingleRandomPct", p1v("degree_strength", "1", "random_mean"), pc,
               P1.format(f="degree_strength", k="1", col="random_mean (mean of each subject's 100 random single regions)") + " x 100",
               HC, {**ctx, "criterion": "random"})

    # "This difference held in every subject at each spatial scale."
    rows = [one(p1, family=f, scale=s, c=0.1, k="1", metric="q1") for f in HUB3 for s in SCALES]
    vb = min(float(r.n - r.n_positive) for r in rows)
    n_all = min(float(r.n) for r in rows)
    out.append(Number("MSingleHubBelowRandomAllN", vb, "every" if vb == n_all else cnt(vb),
                      P1.format(f="{degree_strength, eigenvector_centrality, top_avg_ctrl}", k="1", col="n - n_positive")
                      + " (subjects whose single hub q1 is not above its random mean); min over 3 criteria x 3 scales; "
                      "word 'every' when == n = 70",
                      HC, {"k": 1, "c": 0.1, "stat": "count of subjects", "comparison": "vs random mean"}))

    # "With two degree-strength drivers, q1 exceeded that of random pairs by 23--28 percentage points"
    ctx = {"k": 2, "c": 0.1, "criterion": "degree_strength", "comparison": "vs random mean"}
    out += rng("MExcessTwoDegreePts", p1v("degree_strength", "2", "mean"), pc,
               P1.format(f="degree_strength", k="2", col="mean (mean over subjects of q1 - random mean)") + " x 100 = points",
               HC, {**ctx, "stat": "mean over subjects"})

    # "increasing to 38--43 percentage points with five drivers"
    ctx = {"k": 5, "c": 0.1, "criterion": "degree_strength", "comparison": "vs random mean"}
    out += rng("MExcessFiveDegreePts", p1v("degree_strength", "5", "mean"), pc,
               P1.format(f="degree_strength", k="5", col="mean (mean over subjects of q1 - random mean)") + " x 100 = points",
               HC, {**ctx, "stat": "mean over subjects"})

    # "With two to five drivers, its driver sets had q1 values 30 percentage points lower than degree-strength sets"
    ctx = {"k": "avg2+ (2,3,5)", "c": 0.1, "criterion": "cycle_participation", "comparison": "cycle - degree"}
    out += rng("MCycleBelowDegreePts", {s: -v for s, v in p2v("cycle - degree", "q1", "mean", k="avg2+").items()}, pc,
               P2.format(cmp="cycle - degree", k="avg2+", m="q1", col="mean") + " sign flipped, x 100 = points",
               HC, {**ctx, "stat": "mean over subjects"})
    # "removing the association between cycle participation and strength produced q1 values within 0.014 of random sets"
    d = p2v("cycle_resid - random", "q1", "mean")
    vmax = max(abs(v) for v in d.values())
    out.append(Number("MCycleResidQOneWithin", vmax, ceil_d(vmax, 3),
                      P2.format(cmp="cycle_resid - random", k="avg", m="q1", col="mean")
                      + f" | max over scales of |mean| ({', '.join(f'{s}: {v:+.4f}' for s, v in d.items())}); rounded UP (bound)",
                      HC, {"k": "avg (1,2,3,5)", "c": 0.1, "criterion": "cycle_resid", "comparison": "vs random mean",
                           "stat": "mean over subjects"}))

    # v1 consistency (Fig 3A + subsec:leading_mode)
    sj = _read(str(run / "mechanisms" / "eigvec_subjects.csv"))
    SJ = "mechanisms/04_leading_eigenvectors: mechanisms/eigvec_subjects.csv | column v1_consistency (Pearson r of subject v1 with group-mean v1)"
    # "median correlations of 0.89--0.92 between each subject's v1 and the group mean"
    med = sj.groupby("scale").v1_consistency.median()
    out += rng("MVOneConsistencyMedian", med.to_dict(), fx, SJ + ", median over 70 subjects per scale (as fig3.py panel_subjects)",
               LM, {"stat": "median over subjects", "scales": "all"})

    # strength vs neighbours' loading (Fig 3B + text)
    nd = _read(str(run / "mechanisms" / "node_v1_strength.csv"))
    ND = "mechanisms/08_gramian_alignment: mechanisms/node_v1_strength.csv"
    # caption: "one isolated region is omitted" (114 regions)
    keep = (nd.strength > 0) & (nd.m > 0)
    nom = int((~keep & (nd.scale == 114)).sum())
    out.append(Number("MFigThreeOmittedRegions", float(nom), _WORDS[nom],
                      ND + " | scale == 114, rows failing strength > 0 & m > 0 (the filter of fig3.py panel_strength); printed as a word",
                      F3, {"scale": 114, "stat": "count of region-subject rows"}))
    d = nd[keep]
    rho = d.groupby(["scale", "subject_id"])[["strength", "m"]].apply(lambda g: stats.spearmanr(g.strength, g.m)[0])
    rmed = rho.groupby("scale").median()
    RS = ND + " | strength > 0 & m > 0; within-subject Spearman rho(strength, m), median over subjects (fig3.py panel_strength)"
    # text: "(median Spearman rho = 0.43--0.59; Fig. 3B)"
    out += rng("MRhoStrengthM", rmed.to_dict(), fx, RS, LM, {"stat": "median over subjects", "scales": "all"})

    # text: "Strength alone approximated v1 (cosine similarity 0.69--0.82 across scales)"
    hl = _read(str(run / "mechanisms" / "hub_loading.csv.gz"), ("null", "scale", "subject_id", "cos_s_v1"))
    cos = hl[hl.null == "real"].groupby("scale").cos_s_v1.mean()
    out += rng("MCosStrengthVOne", cos.to_dict(), fx,
               "mechanisms/01_hub_loading: mechanisms/hub_loading.csv.gz | null == 'real', column cos_s_v1, "
               "mean over 70 subjects (medians give the same rounded range)",
               LM, {"stat": "mean over subjects", "scales": "all"})

    # text: "the contribution of v1 accounted for 68% of average controllability among the highest-ranked regions at c = 0.1"
    def ac_top(g: pd.DataFrame, k: int = 5) -> pd.Series:
        top = g.nlargest(k, "avg_ctrl")
        return pd.Series({"share": float((top.avg_ctrl_v1 / top.avg_ctrl).mean())})

    act = nd.groupby(["scale", "subject_id"])[["node_index", "v1", "avg_ctrl", "avg_ctrl_v1"]].apply(ac_top)
    AC = ND + " | per subject, the 5 regions with largest avg_ctrl (c = 0.1)"
    out += rng("MAvgCtrlVOneShare", act.share.groupby("scale").median().to_dict(), pc,
               AC + "; mean of avg_ctrl_v1 / avg_ctrl over them; median over subjects; x 100",
               LM, {"stat": "median over subjects", "k": 5, "c": 0.1, "scales": "all"})

    # text: "The 20% of regions with the highest loadings accounted for 93--98% of the squared weight of v1."
    rg = _read(str(run / "mechanisms" / "eigvec_regions.csv"))

    def top_share(v: np.ndarray) -> float:
        n = int(np.ceil(0.20 * len(v)))  # CORE_FRAC of 04_leading_eigenvectors.py
        return float(np.sort(v**2)[::-1][:n].sum() / np.sum(v**2))

    sh = {s: top_share(rg[rg.scale == s].sort_values("node_index").v1_mean.to_numpy()) for s in SCALES}
    out += rng("MVOneTopTwentyShare", sh, pc,
               "mechanisms/04_leading_eigenvectors: mechanisms/eigvec_regions.csv | column v1_mean (group-mean v1); "
               "share of sum v1_mean^2 on the ceil(0.2 N) regions with largest loading (CORE_FRAC set of 04_leading_eigenvectors.py); "
               "x 100. Recomputed (no saved column). NOTE: the per-subject version (node_v1_strength.csv v1) gives "
               "median 95/98/99%, i.e. 95--99",
               LM, {"stat": "group-mean v1", "fraction": 0.2})

    # L_S and Gramian alignment
    ga = _read(str(run / "mechanisms" / "gramian_alignment.csv"))
    GA = "mechanisms/08_gramian_alignment: mechanisms/gramian_alignment.csv"

    # text: "The three hub-oriented criteria had the highest loadings (0.32--0.52 across driver-set sizes and spatial
    #        scales), compared with 0.01--0.04 for random sets"
    L = kavg(ga, "loading", [1, 2, 3, 5])
    LS = GA + " | column loading (L_S), mean over draws, then k in {1,2,3,5}, then subjects; identical to concentration/set_q1.csv.gz loading"
    ctx = {"k": "avg (1,2,3,5)", "stat": "mean over subjects"}
    out += rng("MLoadingHub", {(f, s): L.loc[f, s] for f in HUB3 for s in SCALES}, fx, LS + "; 3 hub criteria x 3 scales", LM,
               {**ctx, "criterion": "hub3"})
    out += rng("MLoadingRandom", L.loc["random"].to_dict(), fx, LS + "; family == 'random', over scales", LM,
               {**ctx, "criterion": "random"})

    # text: "With two to five drivers, the leading eigenvector of their controllability Gramian had an absolute cosine
    #        similarity of 0.97--0.99 with v1, compared with 0.17--0.34 for random sets"
    C = kavg(ga, "cos_top_v1", [2, 3, 5])
    CS = GA + " | column cos_top_v1, mean over draws, then k in {2,3,5}, then subjects"
    ctx = {"k": "avg (2,3,5)", "c": 0.1, "stat": "mean over subjects"}
    out += rng("MCosHubTwoFive", {(f, s): C.loc[f, s] for f in HUB3 for s in SCALES}, fx, CS + "; 3 hub criteria x 3 scales", LM,
               {**ctx, "criterion": "hub3"})
    out += rng("MCosRandomTwoFive", C.loc["random"].to_dict(), fx, CS + "; family == 'random', over scales", LM,
               {**ctx, "criterion": "random"})
    # "Even individual hubs were strongly aligned with v1 (0.87--0.96)"
    C1 = kavg(ga, "cos_top_v1", [1])
    out += rng("MCosHubSingle", {(f, s): C1.loc[f, s] for f in HUB3 for s in SCALES}, fx,
               GA + " | k == 1, column cos_top_v1, mean over subjects; 3 hub criteria x 3 scales", LM,
               {"k": 1, "c": 0.1, "criterion": "hub3", "stat": "mean over subjects"})
    ar = abs_reach(ga).loc[list(HUB3)]
    AR = (GA + " | columns lam1, trace, tr_perp; per subject, set / mean of its 100 random sets; median over subjects "
          "(abs_reach); 3 hub criteria, ")
    # "With two to five drivers, hub-oriented sets had 5.0--7.4 times the controllability of random sets on that
    #   pattern. Outside the leading structural mode v1, their controllability was 84--96% of that of random sets"
    #   (114 regions; all scales in C_scales)
    a2 = ar[ar.index.get_level_values("k") >= 2]
    ctx = {"k": "2,3,5", "c": 0.1, "criterion": "hub3", "stat": "median ratio over subjects"}
    out += rng("MAbsSetLead", a2.lam1.to_dict(), lambda x: fx(x, 1), AR + "3 scales, k in {2,3,5}; lam1", HC, ctx)
    out += rng("MAbsSetPerp", a2.tr_perp.to_dict(), pc, AR + "3 scales, k in {2,3,5}; tr_perp = trace - L_S/2c; x 100",
               HC, ctx)
    # "and within 20% of random sets at all three spatial resolutions"
    dev = (a2.tr_perp - 1).abs()
    out.append(Number("MAbsSetPerpWithinAllScales", float(dev.max()), ceil_d(100 * dev.max(), 0),
                      AR + f"3 scales, k in {{2,3,5}}; max |tr_perp ratio - 1| x 100 (at {dev.idxmax()}; ratios "
                      f"{a2.tr_perp.min():.4f}--{a2.tr_perp.max():.4f}); rounded UP (bound)", HC, ctx))
    return out


# Tokens as printed in the manuscript.
PRINTED = {
    "MSingleHubBelowRandomAllN": "every",
    "MCycleResidQOneWithin": "0.014",
    "MFigThreeOmittedRegions": "one",
    "MAbsSetPerpWithinAllScales": "20",
    # 114-region values quoted in the Results (ranges over scales: Supplementary Table C_scales)
    "MAbsSetLeadOneFourteenMin": "5.0", "MAbsSetLeadOneFourteenMax": "7.4",
    "MAbsSetPerpOneFourteenMin": "84", "MAbsSetPerpOneFourteenMax": "96",
    "MQOneSingleDegreePctOneFourteen": "87",
    "MQOneSingleRandomPctOneFourteen": "95",
    "MExcessTwoDegreePtsOneFourteen": "27",
    "MExcessFiveDegreePtsOneFourteen": "43",
    "MCycleBelowDegreePtsOneFourteen": "30",
    "MVOneTopTwentyShareOneFourteen": "96",
    "MVOneConsistencyMedianOneFourteen": "0.92",
    "MCosStrengthVOneOneFourteen": "0.76",
    "MRhoStrengthMOneFourteen": "0.54",
    "MLoadingHubOneFourteenMin": "0.41", "MLoadingHubOneFourteenMax": "0.45",
    "MLoadingRandomOneFourteen": "0.02",
    "MAvgCtrlVOneShareOneFourteen": "68",
    "MCosHubTwoFiveOneFourteenMin": "0.98", "MCosHubTwoFiveOneFourteenMax": "0.99",
    "MCosRandomTwoFiveOneFourteen": "0.27",
    "MCosHubSingleOneFourteenMin": "0.94", "MCosHubSingleOneFourteenMax": "0.96",
}
