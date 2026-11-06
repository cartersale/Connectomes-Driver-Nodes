"""Main text: model networks, null networks, assortativity and core cuts (subsec:nulls_results), and the
caption of Fig. 5.

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NAN = float("nan")
SCALES = (68, 114, 219)
NULLS = ("strength", "geometric", "weights")
LOC = ("subsec:nulls_results",)
FIG = ("fig:nulls",)


from .record import Number  # noqa: E402

# --------------------------------------------------------------------------- formatting

def fx(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def sg(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}"


def pc(x: float, d: int = 0) -> str:
    return fx(100 * x, d)


def cnt(x: float) -> str:
    return str(int(round(x)))


def ceil_pc(x: float) -> str:
    """'at most x%' (x a fraction): percentage rounded up so the bound stays true."""
    return str(int(math.ceil(100 * x - 1e-9)))


def N(name, value, text, source, locus=LOC, **ctx) -> Number:
    return Number(name=name, value=float(value), text=text, source=source, locus=tuple(locus), context=ctx)


def rng2(prefix, values, fmt, source, locus=LOC, fmt_lo=None, **ctx) -> list[Number]:
    """A printed range: <prefix>Min / <prefix>Max = min / max of ``values``."""
    v = np.asarray(list(values), float)
    return [N(prefix + "Min", v.min(), (fmt_lo or fmt)(v.min()), source + " | min", locus, **ctx),
            N(prefix + "Max", v.max(), fmt(v.max()), source + " | max", locus, **ctx)]


@lru_cache(maxsize=None)
def _csv(path: str, usecols: tuple | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(usecols) if usecols else None, low_memory=False)


def one(df: pd.DataFrame, **eq) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v).to_numpy()
    x = df[m]
    assert len(x) == 1, (eq, len(x))
    return x.iloc[0]


# --------------------------------------------------------------------------- blocks

def _models(run: Path) -> list[Number]:
    out: list[Number] = []
    rev = run
    # "degree-strength sets of two to five drivers exceeded random sets in q1 by 0.11--0.28 at c=0.1,
    #  compared with 0.31--0.37 in human connectomes"
    cs = _csv(str(rev / "mechanisms/canonical_summary.csv"))
    w = cs[(cs.weighting == "weighted") & np.isclose(cs.c, 0.1) & (cs.kgrp == "2+")]
    mod, conn = w[w.model != "CONN"], w[w.model == "CONN"]
    src = ("mechanisms/02_canonical_graphs: mechanisms/canonical_summary.csv | weighting == 'weighted', "
           "c == 0.1, kgrp == '2+' (mean over k in {2,3,5} and graphs); deg_minus_rand_q1")
    out += rng2("MModelExcessQOne", mod.deg_minus_rand_q1, fx, src + ", model in {ER,BA,WS,SBM,RGG} x 3 scales",
                c=0.1, k="2,3,5", comparison="degree-strength minus random mean")
    out += rng2("MConnExcessQOne", conn.deg_minus_rand_q1, fx,
                src + ", model == 'CONN' (the 69 connected human connectomes) x 3 scales",
                c=0.1, k="2,3,5", comparison="degree-strength minus random mean")

    # "the excess was correlated with the spectral gap (Spearman rho=0.63--0.79) and the loading of degree-strength
    #  sets on v1 (0.80--0.87), but less strongly with variability in node strength (0.31--0.35)"
    g = _csv(str(rev / "mechanisms/canonical_graphs.csv.gz"))
    g = g[(g.weighting == "weighted") & np.isclose(g.c, 0.1) & (g.k >= 2)].assign(ex=lambda x: x.deg_q1 - x.rand_q1)
    pg = g.groupby(["model", "scale", "graph"])[["ex", "gap", "deg_L", "strength_cv"]].mean().reset_index()
    for var, name in (("gap", "MModelRhoGap"), ("deg_L", "MModelRhoLoading"), ("strength_cv", "MModelRhoStrengthCV")):
        vals = [spearmanr(x.ex, x[var]).statistic for _, x in pg.groupby("scale")]
        out += rng2(name, vals, fx,
                    "mechanisms/02_canonical_graphs: mechanisms/canonical_graphs.csv.gz | weighting == "
                    f"'weighted', c == 0.1, k in {{2,3,5}}; per graph (all 6 models incl. CONN) mean over k of "
                    f"deg_q1 - rand_q1 and of {var}; Spearman rho across graphs per scale",
                    c=0.1, k="2,3,5", stat="Spearman across graphs, per scale")
    return out


def _gap(run: Path) -> list[Number]:
    out: list[Number] = []
    rev = run
    ni = _csv(str(rev / "null_models/network_inference.csv"))
    gap = ni[ni.stat == "gap"]
    s = "null_models/02_analyze: null_models/network_inference.csv | stat == 'gap', "
    # "Human connectomes had a spectral gap of 0.43--0.54, approximately twice the null means of 0.19--0.27"
    real = [one(gap, null="strength", scale=sc).real_mean for sc in SCALES]
    out += rng2("MGapReal", real, fx, s + "real_mean (mean over 70 subjects of l1 - l2 of A0) over scales", stat="mean over subjects")
    out += rng2("MGapNull", gap.null_mean, fx, s + "null_mean over 3 nulls x 3 scales", stat="mean over subjects of null mean")
    ratio = (gap.real_mean / gap.null_mean).to_numpy()
    out.append(N("MGapRatioWord", ratio.mean(), "twice",
                 s + f"real_mean / null_mean, mean over 3 nulls x 3 scales (range {ratio.min():.2f}-{ratio.max():.2f}); "
                 "text is the word 'twice' ('approximately twice')", stat="ratio"))
    # "exceeded the 95th percentile ... in 38--66 of 70 subjects for the strength- and geometry-preserving models
    #  and 19--60 for the weight-shuffled model"
    sgm = gap[gap.null.isin(["strength", "geometric"])].n_above95
    out += rng2("MGapAboveNinetyFiveSG", sgm, cnt, s + "null in {strength, geometric}, n_above95 over nulls x scales",
                stat="count of subjects > 95th pct of own 50 nulls")
    out += rng2("MGapAboveNinetyFiveWeights", gap[gap.null == "weights"].n_above95, cnt,
                s + "null == 'weights', n_above95 over scales", stat="count of subjects > 95th pct of own 50 nulls")
    # "l1, which was 1.44--1.89 in human connectomes compared with 1.26--1.45 in the nulls"
    gs = _csv(str(rev / "null_models/gap_split.csv"))
    l1 = gs[gs.stat == "lambda1"]
    s2 = "null_models/03_gap_mediators: null_models/gap_split.csv | stat == 'lambda1', "
    out += rng2("MLeadEigReal", [one(l1, null="strength", scale=sc).real_mean for sc in SCALES], fx,
                s2 + "real_mean over scales", stat="mean over subjects")
    out += rng2("MLeadEigNull", l1.null_mean, fx, s2 + "null_mean over 3 nulls x 3 scales", stat="mean over subjects of null mean")
    return out


@lru_cache(maxsize=None)
def _per_subject_q1(sets_path: str) -> pd.DataFrame:
    """Per subject: real degree-set q1 and the mean over its 50 nulls (per null model), k == 'avg2+', c_ref == 0.1."""
    s = _csv(sets_path, ("null", "null_index", "scale", "subject_id", "c_mode", "c_ref", "k", "degree_q1"))
    s = s[np.isclose(s.c_ref, 0.1) & (s.k.astype(str) == "avg2+")]
    real = s[s.null == "real"].groupby(["c_mode", "scale", "subject_id"]).degree_q1.mean().rename("real")
    nm = s[s.null != "real"].groupby(["null", "c_mode", "scale", "subject_id"]).degree_q1.mean().rename("null_mean").reset_index()
    return nm.join(real, on=["c_mode", "scale", "subject_id"]).assign(delta=lambda d: d.real - d.null_mean)


def _q1(run: Path) -> list[Number]:
    out: list[Number] = []
    rev = run
    inf = _csv(str(rev / "null_models/inference.csv"))
    q = inf[(inf.stat == "degree_q1") & np.isclose(inf.c_ref, 0.1) & (inf.k == "avg2+")]
    fixed, gapm = q[q.c_mode == "fixed"], q[q.c_mode == "gap"]
    s = ("null_models/02_analyze: null_models/inference.csv | stat == 'degree_q1', c_ref == 0.1, k == 'avg2+' "
         "(mean over k in {2,3,5}), ")
    ctx = dict(c=0.1, k="2,3,5", criterion="degree_strength")
    # "degree-strength sets of two to five drivers had q1=0.67--0.74 in human connectomes, compared with 0.53--0.65 in the nulls"
    out += rng2("MNullRealQOne", [one(fixed, null="strength", scale=sc).real_mean for sc in SCALES], fx,
                s + "c_mode == 'fixed', real_mean over scales", **ctx, stat="mean over subjects")
    out += rng2("MNullFixedQOne", fixed.null_mean, fx, s + "c_mode == 'fixed', null_mean over 3 nulls x 3 scales",
                **ctx, stat="mean over subjects of null mean")
    # "Empirical values exceeded the null mean in 59--70 of 70 subjects."
    ps = _per_subject_q1(str(rev / "null_models/sets.csv.gz"))
    n_pos = ps.groupby(["null", "c_mode", "scale"]).delta.apply(lambda d: int((d > 0).sum()))
    ss = ("null_models/02_analyze: null_models/sets.csv.gz | c_ref == 0.1, k == 'avg2+'; per subject real "
          "degree_q1 minus the mean over its 50 nulls (as compare() in 02_analyze.py); count of subjects with delta > 0, ")
    out += rng2("MNullFixedQOneAboveMean", n_pos.xs("fixed", level="c_mode"), cnt, ss + "c_mode == 'fixed', over 3 nulls x 3 scales",
                **ctx, stat="count of subjects")
    # "When the null networks were evaluated at damping values chosen to match the empirical ratio of the first two
    #  decay rates (Methods), their q1 increased to 0.63--0.75."
    out += rng2("MNullGapQOne", gapm.null_mean, fx, s + "c_mode == 'gap', null_mean over 3 nulls x 3 scales",
                **ctx, stat="mean over subjects of null mean", damping="gap-matched")
    # "The remaining empirical-minus-null differences were small (-0.02 to 0.04), with empirical values higher in
    #  only 18--49 subjects."
    out += rng2("MNullGapQOneDiff", gapm.delta_mean, fx, s + "c_mode == 'gap', delta_mean over 3 nulls x 3 scales",
                **ctx, stat="mean over subjects of real - null mean", damping="gap-matched")
    out += rng2("MNullGapQOneAboveMean", n_pos.xs("gap", level="c_mode"), cnt, ss + "c_mode == 'gap', over 3 nulls x 3 scales",
                **ctx, stat="count of subjects", damping="gap-matched")
    return out


def _s8_inline(run: Path) -> dict[str, float]:
    from sections.s8 import inline_values

    return inline_values(run)


def _assort(run: Path) -> list[Number]:
    out: list[Number] = []
    rev = run
    iv = _s8_inline(run)
    s8 = ("null_models/04_rayleigh_bound: null_models/rayleigh/networks.csv.gz, per_subject.csv | "
          "scripts/paper/s8.py inline_values()['{}']")
    # "including them [rich-club coefficients] as covariates accounted for at most 25%, and usually none, of the
    #  empirical-minus-null difference in q1"
    nc = _csv(str(rev / "null_models/mediation_noncircular.csv"))
    rc = nc[(nc.outcome == "degree_q1") & (nc.k == "avg") & nc.mediator.str.startswith("rc")]
    mx = rc.prop_mediated.max()
    out.append(N("MRichClubMediatedMax", mx, ceil_pc(mx),
                 "null_models/03_gap_mediators: null_models/mediation_noncircular.csv | outcome == 'degree_q1' "
                 "(fixed c = 0.1), k == 'avg' (k in {1,2,3,5}; no avg2+ rows saved), mediator in {rc_bu, rc_wu, rc_wu_p75, "
                 f"rc_wu_p85, rc_wu_p95}}; max prop_mediated over mediators x nulls x scales = {mx:.4f}; 'at most' bound "
                 "rounded up", stat="max", k="1,2,3,5"))
    # "Edge-weighted strength assortativity was 0.37--0.46 in the empirical networks, compared with 0.00--0.26 in the nulls."
    out += rng2("MStrengthAssortReal", [iv[f"rW_real_{sc}"] for sc in SCALES], fx, s8.format("rW_real_<scale>"),
                stat="mean over subjects")
    out += rng2("MStrengthAssortNull", [iv[f"rW_null_{n}_{sc}"] for n in NULLS for sc in SCALES], fx,
                s8.format("rW_null_<null>_<scale>") + " (min -0.003, strength/219, printed 0.00)",
                stat="mean over subjects of null mean")
    # "It exceeded the 95th percentile of the null distribution in 69 or 70 subjects at every scale and for every null
    #  model, except the geometry-preserving nulls at 68 regions (40 subjects)."
    above = [iv[f"rW_n_above95_{n}_{sc}"] for n in NULLS for sc in SCALES if (n, sc) != ("geometric", 68)]
    out += rng2("MStrengthAssortAbove", above, cnt, s8.format("rW_n_above95_<null>_<scale>") +
                ", over the 8 null x scale cells other than geometric/68", stat="count of subjects")
    out.append(N("MStrengthAssortAboveGeomSixtyEight", iv["rW_n_above95_geometric_68"], cnt(iv["rW_n_above95_geometric_68"]),
                 s8.format("rW_n_above95_geometric_68"), scale=68, null="geometric", stat="count of subjects"))
    # "Subjects whose connectomes exceeded their strength-preserving nulls more in assortativity also exceeded them
    #  more in spectral gap (Spearman rho = 0.31--0.47 across subjects; Fig. 5C)"
    ps = _csv(str(rev / "null_models/rayleigh/per_subject.csv"))
    d = ps[ps.null == "strength"]
    rhos = [spearmanr(g.d_r_W, g.d_gap).statistic for _, g in d.groupby("scale")]
    out += rng2("MAssortGapExcessRho", rhos, fx,
                "null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == 'strength'; Spearman "
                "rho across subjects of d_r_W vs d_gap per scale (as fig5.py panel_assort)", locus=LOC + FIG,
                null="strength", stat="Spearman across subjects, per scale")
    # "Relative to strength-preserving nulls, it accounted for a median of 36--60% of the excess in l1"
    out += rng2("MRayleighShareROne", [iv[f"share_l1_R1_strength_{sc}"] for sc in SCALES], pc,
                s8.format("share_l1_R1_strength_<scale>") + ": median over subjects of d_R1/d_lambda1", null="strength",
                stat="median over subjects")
    return out


def _cut(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mech = run / "mechanisms"
    base = _csv(str(mech / "core_cut_baseline.csv"))
    df = _csv(str(mech / "core_cut.csv.gz"))
    return base, df


def _core(run: Path) -> list[Number]:
    out: list[Number] = []
    mech = run / "mechanisms"
    base, df = _cut(run)
    sb = "mechanisms/07_core_cut: mechanisms/core_cut_baseline.csv | "
    # "Connections between these groups accounted for 5--7% of total network weight."
    share = base[base.partition == "labels"].groupby("scale").w_cross_share.median()
    out += rng2("MCoreCrossWeightPct", share, pc, sb + "partition == 'labels', median over subjects of w_cross_share, over scales",
                stat="median over subjects", partition="labels")

    # Eigenvalue changes, as 07_core_cut.summary(): per subject mean over draws (dropna dgap), dgap% = 100 dgap / gap0,
    # median over subjects.
    d = df.dropna(subset=["dgap"])
    per = d.groupby(["scale", "partition", "subject_id", "alpha", "condition"])[["dl1", "dl2", "dgap"]].mean().reset_index()
    per = per.merge(base[["scale", "partition", "subject_id", "gap"]].rename(columns={"gap": "gap0"}),
                    on=["scale", "partition", "subject_id"])
    per["red_pct"] = -100 * per.dgap / per.gap0
    med = per.groupby(["scale", "partition", "alpha", "condition"])[["dl1", "dl2", "red_pct"]].median()
    sc_ = ("mechanisms/07_core_cut: mechanisms/core_cut.csv.gz + core_cut_baseline.csv | alpha == 0.5, "
           "partition == '{p}', condition == '{c}', scale == {s}; per subject mean over draws, {what}, median over subjects "
           "(as core_cut_summary.md 'Eigenvalue changes')")
    W = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}
    pct_what = "gap reduction % = -100 dgap / baseline gap"

    def red(name, p, c, s, loc=LOC):
        v = med.loc[(s, p, 0.5, c), "red_pct"]
        return N(name, v, fx(v, 0), sc_.format(p=p, c=c, s=s, what=pct_what), loc, scale=s, partition=p, condition=c,
                 alpha=0.5, stat="median over subjects")

    # "Halving their weights reduced the spectral gap by a median of 21% at 68 regions and 32% at 219 regions"
    out += [red("MCoreCrossGapRed" + W[s], "labels", "cross", s) for s in (68, 219)]
    # "Removing the same amount of weight from within-group connections ... reduced the gap by only 6% and 14%"
    out += [red("MCoreWithinGapRed" + W[s], "labels", "within_matched", s) for s in (68, 219)]
    # "Removing it from connections outside the core reduced the gap by at most 2%."
    oth = med.xs(("labels", 0.5, "other_matched"), level=("partition", "alpha", "condition")).red_pct
    mx = float(oth.max())
    assert math.ceil(mx) <= 2, mx
    out.append(N("MCoreOtherGapRedMax", mx, str(int(math.ceil(mx))),
                 sc_.format(p="labels", c="other_matched", s="68/114/219", what=pct_what) +
                 f"; max over scales of the median reduction ({', '.join(f'{v:+.1f}' for v in oth)}); 'at most' bound rounded up",
                 partition="labels", condition="other_matched", alpha=0.5, stat="max over scales of median"))
    # "Between-group connections had the greater effect in 55 and 57 of 70 subjects at the two scales"
    comp = _csv(str(mech / "core_cut_compare.csv"))
    for s in (68, 219):
        r = one(comp, scale=s, partition="labels", alpha=0.5, control="within_matched", measure="dgap")
        out.append(N("MCoreCrossBeatsWithin" + W[s], r.frac_cross_lower * r.n, cnt(r.frac_cross_lower * r.n),
                     "mechanisms/07_core_cut: mechanisms/core_cut_compare.csv | scale == "
                     f"{s}, partition == 'labels', alpha == 0.5, control == 'within_matched', measure == 'dgap'; "
                     "frac_cross_lower * n", scale=s, alpha=0.5, stat="count of subjects"))
    # (claim "and across reductions of 25--100%": checked here, not a macro)
    all_alpha = comp[(comp.scale.isin([68, 219])) & (comp.partition == "labels") & (comp.control == "within_matched") &
                (comp.measure == "dgap")]
    assert (all_alpha.frac_cross_lower > 0.5).all(), all_alpha[["scale", "alpha", "frac_cross_lower"]]

    # "(+0.03 and -0.02, compared with -0.03 and -0.07 for matched within-group connections)" -- Delta l2
    l2w = "dl2 (change in l2)"
    for c, nm, fmt in (("cross", "MCoreCrossDeltaLTwo", sg), ("within_matched", "MCoreWithinDeltaLTwo", fx)):
        for s in (68, 219):
            v = med.loc[(s, "labels", 0.5, c), "dl2"]
            out.append(N(nm + W[s], v, fmt(v), sc_.format(p="labels", c=c, s=s, what=l2w), scale=s, partition="labels",
                         condition=c, alpha=0.5, stat="median over subjects"))
    # "At 114 regions ... Halving anterior--posterior connections reduced the gap by 9%, compared with 13% for matched
    #  within-group connections."
    out.append(red("MCoreCrossGapRedOneFourteen", "labels", "cross", 114))
    out.append(red("MCoreWithinGapRedOneFourteen", "labels", "within_matched", 114))
    # "the median absolute correlation between the anatomical split and v2 was 0.30, compared with 0.74 at 68 regions"
    vc = base[base.partition == "labels"].groupby("scale").v2_split_corr.median()
    for s in (114, 68):
        out.append(N("MCoreSplitVTwoCorr" + W[s], vc[s], fx(vc[s]),
                     sb + f"partition == 'labels', scale == {s}, median over subjects of v2_split_corr (|corr(v2, +-1 split)| on core)",
                     scale=s, stat="median over subjects"))
    # "particularly the right rostral anterior cingulate (93% of subjects) and left paracentral lobule (89%)"
    side = _csv(str(mech / "core_cut_v2_side.csv"))
    for lab, nm in (("R_rostralanteriorcingulate_1", "MCoreWrongSideRrACC"), ("L_paracentral_1", "MCoreWrongSideLParacentral")):
        v = one(side, scale=114, region_label=lab).wrong_side
        out.append(N(nm, v, pc(v), "mechanisms/07_core_cut: mechanisms/core_cut_v2_side.csv | scale == 114, "
                     f"region_label == '{lab}', wrong_side (share of subjects with v2 on the opposite side of its label)",
                     scale=114, stat="share of subjects"))
    # "When the division was instead defined by the sign of each subject's v2, halving connections between the two
    #  groups reduced the gap by 34%"  (context: 114 regions)
    out.append(red("MCoreVTwoCrossGapRedOneFourteen", "v2sign", "cross", 114))
    # "and produced a greater reduction per unit weight than weakening within-group connections in 97% of subjects"
    b = base[(base.partition == "v2sign") & (base.scale == 114)]
    win = ((b.fo_dl1_cross - b.fo_dl2_cross) < (b.fo_dl1_within - b.fo_dl2_within)).mean()
    out.append(N("MCoreVTwoPerWeightBeatsPct", win, pc(win),
                 sb + "partition == 'v2sign', scale == 114; share of subjects with first-order per-unit-weight dgap "
                 "(fo_dl1 - fo_dl2) of cross < within (= 'cross > within' column of core_cut_summary.md); the alpha = 0.5 "
                 "matched comparison (core_cut_compare.csv frac_cross_lower) is also 0.97",
                 scale=114, partition="v2sign", stat="share of subjects"))

    # ---- Fig. 5 caption (fig5.py cut_table / panel_cut)
    # "whereas the v2-defined cut closes about a third of it at every scale"
    v2c = [med.loc[(s, "v2sign", 0.5, "cross"), "red_pct"] for s in SCALES]
    out.append(N("MFigFiveVTwoCutThirdWord", float(np.mean(v2c)), "about a third",
                 sc_.format(p="v2sign", c="cross", s="68/114/219", what=pct_what) +
                 f"; per-scale medians {', '.join(f'{v:.1f}' for v in v2c)}%; text is the words 'about a third'",
                 FIG, partition="v2sign", alpha=0.5, stat="median over subjects, each scale"))
    # "nine of 840 values lie beyond the axis and are drawn at its edge"
    t = df[(df.alpha == 0.5) & df.condition.isin(["cross", "within_matched", "other_matched"])]
    t = t[(t.partition == "labels") | ((t.partition == "v2sign") & (t.condition == "cross"))]
    t = t.assign(cond=np.where(t.partition == "v2sign", "v2_cross", t.condition), closed=-100 * t.dgap / (t.gap - t.dgap))
    closed = t.groupby(["cond", "scale", "subject_id"]).closed.mean()
    lim = (-30, 80)  # fig5.py panel_cut axis limits (display setting)
    n_out = int(((closed < lim[0]) | (closed > lim[1])).sum())
    fs = ("mechanisms/07_core_cut: mechanisms/core_cut.csv.gz | as scripts/figures/fig5.py "
          "cut_table/panel_cut: alpha == 0.5, 4 conditions (labels cross/within_matched/other_matched, v2sign cross), "
          f"per-subject mean % closed; {{}} (axis {lim})")
    words = {9: "nine"}
    out.append(N("MFigFiveCutOutOfAxis", n_out, words.get(n_out, str(n_out)),
                 fs.format("count of values outside the axis; text is the word"), FIG, stat="count"))
    out.append(N("MFigFiveCutValues", len(closed), str(len(closed)), fs.format("number of plotted subject values"), FIG,
                 stat="count", note="arguably a setting (70 x 3 x 4)"))
    return out


def collect(run: Path) -> list[Number]:
    run = Path(run)
    out = _models(run) + _gap(run) + _q1(run) + _assort(run) + _core(run)
    names = [n.name for n in out]
    assert len(names) == len(set(names)), [x for x in names if names.count(x) > 1]
    return out


# --------------------------------------------------------------------------- check against the manuscript

# Tokens as printed in the manuscript.
PRINTED = {
    "MModelExcessQOneMin": "0.11", "MModelExcessQOneMax": "0.28",
    "MConnExcessQOneMin": "0.31", "MConnExcessQOneMax": "0.37",
    "MModelRhoGapMin": "0.63", "MModelRhoGapMax": "0.79",
    "MModelRhoLoadingMin": "0.80", "MModelRhoLoadingMax": "0.87",
    "MModelRhoStrengthCVMin": "0.31", "MModelRhoStrengthCVMax": "0.35",
    "MGapRealMin": "0.43", "MGapRealMax": "0.54", "MGapNullMin": "0.19", "MGapNullMax": "0.27", "MGapRatioWord": "twice",
    "MGapAboveNinetyFiveSGMin": "38", "MGapAboveNinetyFiveSGMax": "66",
    "MGapAboveNinetyFiveWeightsMin": "19", "MGapAboveNinetyFiveWeightsMax": "60",
    "MLeadEigRealMin": "1.44", "MLeadEigRealMax": "1.89", "MLeadEigNullMin": "1.26", "MLeadEigNullMax": "1.45",
    "MNullRealQOneMin": "0.67", "MNullRealQOneMax": "0.74", "MNullFixedQOneMin": "0.53", "MNullFixedQOneMax": "0.65",
    "MNullFixedQOneAboveMeanMin": "59", "MNullFixedQOneAboveMeanMax": "70",
    "MNullGapQOneMin": "0.63", "MNullGapQOneMax": "0.75",
    "MNullGapQOneDiffMin": "-0.02", "MNullGapQOneDiffMax": "0.04",
    "MNullGapQOneAboveMeanMin": "18", "MNullGapQOneAboveMeanMax": "49",
    "MRichClubMediatedMax": "25",
    "MStrengthAssortRealMin": "0.37", "MStrengthAssortRealMax": "0.46",
    "MStrengthAssortNullMin": "0.00", "MStrengthAssortNullMax": "0.26",
    "MStrengthAssortAboveMin": "69", "MStrengthAssortAboveMax": "70", "MStrengthAssortAboveGeomSixtyEight": "40",
    "MAssortGapExcessRhoMin": "0.31", "MAssortGapExcessRhoMax": "0.47",
    "MRayleighShareROneMin": "36", "MRayleighShareROneMax": "60",
    "MCoreCrossWeightPctMin": "5", "MCoreCrossWeightPctMax": "7",
    "MCoreCrossGapRedSixtyEight": "21", "MCoreCrossGapRedTwoNineteen": "32",
    "MCoreWithinGapRedSixtyEight": "6", "MCoreWithinGapRedTwoNineteen": "14",
    "MCoreOtherGapRedMax": "2",
    "MCoreCrossBeatsWithinSixtyEight": "55", "MCoreCrossBeatsWithinTwoNineteen": "57",
    "MCoreCrossDeltaLTwoSixtyEight": "+0.03", "MCoreCrossDeltaLTwoTwoNineteen": "-0.02",
    "MCoreWithinDeltaLTwoSixtyEight": "-0.03", "MCoreWithinDeltaLTwoTwoNineteen": "-0.07",
    "MCoreCrossGapRedOneFourteen": "9", "MCoreWithinGapRedOneFourteen": "13",
    "MCoreSplitVTwoCorrOneFourteen": "0.30", "MCoreSplitVTwoCorrSixtyEight": "0.74",
    "MCoreWrongSideRrACC": "93", "MCoreWrongSideLParacentral": "89",
    "MCoreVTwoCrossGapRedOneFourteen": "34", "MCoreVTwoPerWeightBeatsPct": "97",
    "MFigFiveVTwoCutThirdWord": "about a third", "MFigFiveCutOutOfAxis": "nine", "MFigFiveCutValues": "840",
}
