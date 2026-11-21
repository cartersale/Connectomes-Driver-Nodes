"""Main text: null networks, assortativity and core cuts (subsec:nulls_results), and the caption of Fig. 5.

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ._common import scale_siblings
from .record import Number

NAN = float("nan")
SCALES = (68, 114, 219)
NULLS = ("strength", "geometric", "weights")
LOC = ("subsec:nulls_results",)
FIG = ("fig:nulls",)


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
    if isinstance(values, (dict, pd.Series)):
        items = list(values.items())
    else:
        values = list(values)
        items = list(zip(SCALES, values)) if len(values) == len(SCALES) else []  # a list over SCALES, in order
    v = np.asarray([x for _, x in items] if items else values, float)
    return [N(prefix + "Min", v.min(), (fmt_lo or fmt)(v.min()), source + " | min", locus, **ctx),
            N(prefix + "Max", v.max(), fmt(v.max()), source + " | max", locus, **ctx),
            *scale_siblings(prefix, items, fmt, source, locus, ctx)]


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



def _gap(run: Path) -> list[Number]:
    out: list[Number] = []
    ni = _csv(str(run / "null_models/network_inference.csv"))
    gap = ni[ni.stat == "gap"]
    s = "null_models/02_analyze: null_models/network_inference.csv | stat == 'gap', "
    # "Human connectomes had spectral gaps of 0.43--0.54, approximately twice the null means of 0.19--0.27"
    real = [one(gap, null="strength", scale=sc).real_mean for sc in SCALES]
    out += rng2("MGapReal", real, fx, s + "real_mean (mean over 70 subjects of l1 - l2 of A0) over scales", stat="mean over subjects")
    out += rng2("MGapNull", gap.set_index(["null", "scale"]).null_mean, fx, s + "null_mean over 3 nulls x 3 scales", stat="mean over subjects of null mean")
    ratio = (gap.real_mean / gap.null_mean).to_numpy()
    out.append(N("MGapRatioWord", ratio.mean(), "twice",
                 s + f"real_mean / null_mean, mean over 3 nulls x 3 scales (range {ratio.min():.2f}-{ratio.max():.2f}); "
                 "text is the word 'twice' ('approximately twice')", stat="ratio"))
    return out


def _q1(run: Path) -> list[Number]:
    out: list[Number] = []
    inf = _csv(str(run / "null_models/inference.csv"))
    q = inf[(inf.stat == "degree_q1") & np.isclose(inf.c_ref, 0.1) & (inf.k == "avg2+")]
    fixed, gapm = q[q.c_mode == "fixed"], q[q.c_mode == "gap"]
    s = ("null_models/02_analyze: null_models/inference.csv | stat == 'degree_q1', c_ref == 0.1, k == 'avg2+' "
         "(mean over k in {2,3,5}), ")
    ctx = dict(c=0.1, k="2,3,5", criterion="degree_strength")
    # "At c=0.1, degree-strength sets of two to five drivers had q1=0.67--0.74 in human connectomes, compared with
    #  0.53--0.65 in the null networks"
    out += rng2("MNullRealQOne", [one(fixed, null="strength", scale=sc).real_mean for sc in SCALES], fx,
                s + "c_mode == 'fixed', real_mean over scales", **ctx, stat="mean over subjects")
    out += rng2("MNullFixedQOne", fixed.set_index(["null", "scale"]).null_mean, fx, s + "c_mode == 'fixed', null_mean over 3 nulls x 3 scales",
                **ctx, stat="mean over subjects of null mean")
    # "When we adjusted the damping of each null network to match the ratio of the first two decay rates in its
    #  corresponding empirical connectome, q1 increased to 0.63--0.75"
    out += rng2("MNullGapQOne", gapm.set_index(["null", "scale"]).null_mean, fx, s + "c_mode == 'gap', null_mean over 3 nulls x 3 scales",
                **ctx, stat="mean over subjects of null mean", damping="gap-matched")
    # "leaving only small differences from the empirical networks (-0.02 to 0.04)"
    out += rng2("MNullGapQOneDiff", gapm.set_index(["null", "scale"]).delta_mean, fx, s + "c_mode == 'gap', delta_mean over 3 nulls x 3 scales",
                **ctx, stat="mean over subjects of real - null mean", damping="gap-matched")
    return out


def _assort(run: Path) -> list[Number]:
    from sections.s8 import inline_values

    out: list[Number] = []
    iv = inline_values(run)
    s8 = ("null_models/04_rayleigh_bound: null_models/rayleigh/networks.csv.gz, per_subject.csv | "
          "scripts/paper/sections/s8.py inline_values()['{}']")
    # "Edge-weighted strength assortativity ... was 0.37--0.46 in empirical networks compared with 0.00--0.26 in the nulls"
    out += rng2("MStrengthAssortReal", [iv[f"rW_real_{sc}"] for sc in SCALES], fx, s8.format("rW_real_<scale>"),
                stat="mean over subjects")
    out += rng2("MStrengthAssortNull", {(n, sc): iv[f"rW_null_{n}_{sc}"] for n in NULLS for sc in SCALES}, fx,
                s8.format("rW_null_<null>_<scale>") + " (min -0.003, strength/219, printed 0.00)",
                stat="mean over subjects of null mean")
    # "Subjects with greater excess assortativity relative to strength-preserving nulls also tended to have larger
    #  excess spectral gaps (Spearman rho=0.31--0.47; Fig. 5C)"
    ps = _csv(str(run / "null_models/rayleigh/per_subject.csv"))
    d = ps[ps.null == "strength"]
    rhos = [spearmanr(g.d_r_W, g.d_gap).statistic for _, g in d.groupby("scale")]
    out += rng2("MAssortGapExcessRho", rhos, fx,
                "null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == 'strength'; Spearman "
                "rho across subjects of d_r_W vs d_gap per scale (as fig5.py panel_assort)",
                null="strength", stat="Spearman across subjects, per scale")
    return out


def _cut(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mech = run / "mechanisms"
    base = _csv(str(mech / "core_cut_baseline.csv"))
    df = _csv(str(mech / "core_cut.csv.gz"))
    return base, df


def _core(run: Path) -> list[Number]:
    """Front--back core cuts. The text and figure are in Supplementary Section S6 (sec:core); the numbers are computed
    here to reuse this module's core-cut helpers, and carry S names and the supplement section."""
    out: list[Number] = []
    S6 = ("sec:core",)
    sup = {"section": "supplement"}
    base, df = _cut(run)
    sb = "mechanisms/07_core_cut: mechanisms/core_cut_baseline.csv | "
    # "Connections between the anatomically defined anterior and posterior halves of the core carried 5--7% of total
    #  network weight."
    share = base[base.partition == "labels"].groupby("scale").w_cross_share.median()
    out += rng2("SCoreCrossWeightPct", share, pc, sb + "partition == 'labels', median over subjects of w_cross_share, over scales",
                S6, stat="median over subjects", partition="labels", **sup)

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
    pct_what = "gap reduction % = -100 dgap / baseline gap"

    def red(name, p, c, s):
        v = med.loc[(s, p, 0.5, c), "red_pct"]
        return N(name, v, fx(v, 0), sc_.format(p=p, c=c, s=s, what=pct_what), S6, scale=s, partition=p, condition=c,
                 alpha=0.5, stat="median over subjects", **sup)

    # "... and at most 2% for connections outside the core" (matched control; 21/32 and 6/14 are SCoreFrontBack*Half
    #  and SCoreWithinMatchedHalf* of supplement.py)
    oth = med.xs(("labels", 0.5, "other_matched"), level=("partition", "alpha", "condition")).red_pct
    mx = float(oth.max())
    assert math.ceil(mx) <= 2, mx
    out.append(N("SCoreOtherMatchedGapRedMax", mx, str(int(math.ceil(mx))),
                 sc_.format(p="labels", c="other_matched", s="68/114/219", what=pct_what) +
                 f"; max over scales of the median reduction ({', '.join(f'{v:+.1f}' for v in oth)}); 'at most' bound rounded up",
                 S6, partition="labels", condition="other_matched", alpha=0.5, stat="max over scales of median", **sup))
    return out


def collect(run: Path) -> list[Number]:
    run = Path(run)
    return _gap(run) + _q1(run) + _assort(run) + _core(run)


# Tokens as printed in the manuscript.
PRINTED = {
    "SCoreCrossWeightPctMin": "5", "SCoreCrossWeightPctMax": "7",
    "SCoreOtherMatchedGapRedMax": "2",
    "MGapRatioWord": "twice",
    # 114-region values quoted in the Results (ranges over scales: Supplementary Table C_scales)
    "MGapRealOneFourteen": "0.54",
    "MGapNullOneFourteenMin": "0.21", "MGapNullOneFourteenMax": "0.23",
    "MNullRealQOneOneFourteen": "0.74",
    "MNullFixedQOneOneFourteenMin": "0.57", "MNullFixedQOneOneFourteenMax": "0.59",
    "MNullGapQOneOneFourteenMin": "0.71", "MNullGapQOneOneFourteenMax": "0.73",
    "MNullGapQOneDiffOneFourteenMin": "0.00", "MNullGapQOneDiffOneFourteenMax": "0.03",
    "MStrengthAssortRealOneFourteen": "0.46",
    "MStrengthAssortNullOneFourteenMin": "0.03", "MStrengthAssortNullOneFourteenMax": "0.15",
    "MAssortGapExcessRhoOneFourteen": "0.47",
}
