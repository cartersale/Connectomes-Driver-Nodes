"""Main text: average energy and the regularization floor (subsec:energy_floor), hub concentration
(subsec:hub_concentration, first paragraphs), and the captions of Figs. 1 and 2.

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

NAN = float("nan")
SCALES = (68, 114, 219)
SCALE_WORD = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}
HUB3 = ("degree_strength", "eigenvector_centrality", "top_avg_ctrl")


from .record import Number  # noqa: E402

# --------------------------------------------------------------------------- formatting helpers

def fx(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def pc(x: float, d: int = 0) -> str:
    return fx(100 * x, d)


def cnt(x: float) -> str:
    return str(int(round(x)))


def ceil_dp(x: float, d: int) -> str:
    """Upper bound ("within x", "at most x"): round up so the statement stays true."""
    f = 10 ** d
    return fx(math.ceil(x * f - 1e-9) / f, d)


def floor_int(x: float) -> str:
    """Lower bound ("at least x")."""
    return str(int(math.floor(x + 1e-9)))


@lru_cache(maxsize=None)
def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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


def N(name, value, text, source, locus, **context) -> Number:
    if isinstance(locus, str):
        locus = (locus,)
    return Number(name=name, value=float(value), text=text, source=source, locus=tuple(locus), context=context)


def gramian_q1_factory(a0: np.ndarray, c: float):
    """Eigenbasis of A0 and the infinite-horizon Gramian kernel, exactly as hoc.analysis.null_models.Spectral.kernel and
    figures/fig1.py set_values: mu = lam - lam_max - c, kernel = -1/(mu_i + mu_j), W (in A0's eigenbasis) =
    kernel * (U_S^T U_S)."""
    lam, u = np.linalg.eigh(np.asarray(a0, float))
    mu = lam - lam[-1] - c
    kern = -1.0 / (mu[:, None] + mu[None, :])

    def spectrum(nodes):
        us = u[list(nodes), :]
        with np.errstate(all="ignore"):
            g = kern * (us.T @ us)
        return np.linalg.eigvalsh(g)  # ascending

    return lam, u, spectrum


# --------------------------------------------------------------------------- subsec:energy_floor

def energy_floor(run: Path) -> list[Number]:
    out: list[Number] = []
    loc = ("subsec:energy_floor",)
    a = csv(run, "sensitivity/energy_floor_driver_sets.csv")
    e5 = a[np.isclose(a.eps, 1e-5)]
    g = e5.groupby(["scale", "method"])[["fraction_below_eps", "log10_average_energy"]].mean()
    tsrc = ("sensitivity/04_energy_floor: sensitivity/energy_floor_driver_sets.csv | c = 0.1, eps == 1e-5, methods "
            "degree_strength, eigenvector_centrality, betweenness, cycle_participation, random_mean (mean of the 100 "
            "random sets); mean over subjects and k in {1,2,3,5} per (scale, method); ")

    # "Most eigenvalues of the controllability Gramian (86--96\%) fell below the regularization floor"
    fb = g["fraction_below_eps"]
    out.append(N("MFracBelowEpsMin", fb.min(), pc(fb.min()), tsrc + "fraction_below_eps, min over scales x methods (x100)",
                 loc, eps=1e-5, stat="min over scale x method of cohort mean"))
    out.append(N("MFracBelowEpsMax", fb.max(), pc(fb.max()), tsrc + "fraction_below_eps, max over scales x methods (x100)",
                 loc, eps=1e-5, stat="max over scale x method of cohort mean"))

    # "$\log_{10}E_{\mathrm{avg}}$ remained within 0.07 of $\log_{10}(N/\epsilon)$ across all criteria and spatial resolutions"
    dev = (np.log10(g.index.get_level_values("scale") / 1e-5) - g["log10_average_energy"]).abs()
    out.append(N("MLogEavgFloorGap", dev.max(), ceil_dp(dev.max(), 2),
                 tsrc + "max over scales x methods of |log10(N/eps) - cohort-mean log10_average_energy|; bound, rounded UP. "
                 "Holds for cohort means only: per-set max is 0.13 (68 regions)", loc, eps=1e-5, bound="within"))

    # "The smallest regularized Gramian eigenvalue was also fixed at $\epsilon$ for all 302,400 driver sets"
    t6 = csv(run, "spectra/tables/t6_condition_number_check.csv").set_index("Unnamed: 0")["0"]
    t6src = "spectra/02_analyze: spectra/tables/t6_condition_number_check.csv | "
    out.append(N("MLambdaMinEqEpsNSets", t6["n_sets"], f"{int(t6['n_sets']):,}",
                 t6src + "row n_sets (= rows of spectra/set_metrics.csv: 10 rankings + 100 random + 300 matched sets "
                 "per subject x scale x k, c = 0.1; excludes top_avg_ctrl / top_modal_* / low_perron)", loc, c=0.1, eps=1e-5))
    frac = float(t6["frac_lambda_min_eq_eps"])
    out.append(N("MLambdaMinEqEpsAll", frac, "all" if frac == 1.0 else pc(frac, 1),
                 t6src + "row frac_lambda_min_eq_eps (fraction of sets with lambda_min(W + eps I) == eps, rtol 1e-9); "
                 "word 'all' when == 1", loc, c=0.1, eps=1e-5))
    return out


# --------------------------------------------------------------------------- fig:overview (Fig. 1 caption)

def fig1(run: Path) -> list[Number]:
    """Replicates scripts/figures/fig1.py main() + set_values(): cohort_reference of
    processed/connectomes_scale-114.npz, c = 0.1, k = 3, hub = top-3 strength (hoc.rankings.strength_scores),
    1000 random 3-sets from np.random.default_rng(0); L_S = sum v1(S)^2, q1 = w_max / sum(clip(w, 0))."""
    from hoc.rankings import strength_scores

    loc = ("fig:overview",)
    a0 = np.asarray(np.load(run / "processed" / "connectomes_scale-114.npz")["cohort_reference"], float)
    n, k = len(a0), 3
    hub = list(np.argsort(-strength_scores(a0))[:k])
    rng = np.random.default_rng(0)
    rand = [rng.choice(n, k, replace=False) for _ in range(1000)]
    lam, u, spectrum = gramian_q1_factory(a0, 0.1)
    v1 = u[:, -1]

    def vals(s):
        w = spectrum(s)
        return float(np.sum(v1[list(s)] ** 2)), float(w[-1] / np.clip(w, 0, None).sum())

    hl, hq = vals(hub)
    r = np.array([vals(s) for s in rand])
    src = ("figures/fig1 (recomputed as fig1.py set_values): processed/connectomes_scale-114.npz | "
           "cohort_reference, c = 0.1, k = 3, infinite horizon; ")
    ctx = dict(scale=114, k=3, c=0.1, connectome="cohort_reference")
    out = [
        # "The degree-strength set has $L_S = 0.48$ and $q_1 = 0.78$"
        N("MFigOneHubLoading", hl, fx(hl), src + "L_S of top-3 degree-strength set", loc, **ctx),
        N("MFigOneHubQOne", hq, fx(hq), src + "q1 of top-3 degree-strength set", loc, **ctx),
        # "random sets have median $L_S$ near 0"
        N("MFigOneRandomMedianLoading", np.median(r[:, 0]), fx(np.median(r[:, 0]), 0),
          src + "median L_S over 1000 random 3-sets (text 'near 0'; printed to 0 dp)", loc, **ctx, stat="median"),
        # "and median $q_1 = 0.33$"
        N("MFigOneRandomMedianQOne", np.median(r[:, 1]), fx(np.median(r[:, 1])),
          src + "median q1 over 1000 random 3-sets", loc, **ctx, stat="median"),
    ]
    # "and none reaches the degree-strength values"
    n_reach = int(np.sum((r[:, 0] >= hl) | (r[:, 1] >= hq)))
    out.append(N("MFigOneRandomReachHub", n_reach, "none" if n_reach == 0 else cnt(n_reach),
                 src + "count of random sets with L_S >= hub L_S or q1 >= hub q1 (word 'none' when 0)", loc, **ctx))
    return out


# --------------------------------------------------------------------------- subsec:hub_concentration + paragraph after Fig. 2

P1SRC = "concentration/02_tables: concentration/q1_vs_random.csv | c == 0.1, metric == '{m}', k == '{k}'; "


def p1rows(run: Path, fams, k: str, metric: str = "q1") -> pd.DataFrame:
    p = csv(run, "concentration/q1_vs_random.csv")
    x = p[np.isclose(p.c, 0.1) & (p.k == k) & (p.metric == metric) & p.family.isin(list(fams))]
    assert len(x) == 3 * len(fams), (fams, k, metric, len(x))
    return x


def hub_concentration(run: Path) -> list[Number]:
    out: list[Number] = []
    loc = ("subsec:hub_concentration",)

    # "the three hub-oriented criteria concentrated 67--78\% of their controllability on a single pattern"
    h = p1rows(run, HUB3, "avg2+")
    s = P1SRC.format(m="q1", k="avg2+") + f"families {HUB3}, all scales; "
    ctx = dict(c=0.1, k="avg2+ (2,3,5)", criterion="hub3", stat="mean over subjects")
    out.append(N("MQOneHubTwoFiveMin", h.value_mean.min(), pc(h.value_mean.min()), s + "value_mean x100, min", loc, **ctx))
    out.append(N("MQOneHubTwoFiveMax", h.value_mean.max(), pc(h.value_mean.max()), s + "value_mean x100, max", loc, **ctx))
    # "compared with 36--39\% for random sets"
    out.append(N("MQOneRandomTwoFiveMin", h.random_mean.min(), pc(h.random_mean.min()),
                 s + "random_mean x100 (mean of each subject's 100 random sets), min over scales", loc, **ctx))
    out.append(N("MQOneRandomTwoFiveMax", h.random_mean.max(), pc(h.random_mean.max()),
                 s + "random_mean x100, max over scales", loc, **ctx))
    # "All three exceeded the random mean in every subject at every spatial scale"
    npos = h.n_positive.min()
    out.append(N("MHubAboveRandomEvery", npos, "every" if npos == h.n.min() else cnt(npos),
                 s + "min n_positive over families x scales (word 'every' when == n = 70)", loc, **ctx))
    # "ranking at the 99th to 100th percentile of random sets"
    out.append(N("MHubPctTwoFiveMin", h.pct_mean.min(), pc(h.pct_mean.min()), s + "pct_mean x100, min", loc, **ctx))
    out.append(N("MHubPctTwoFiveMax", h.pct_mean.max(), pc(h.pct_mean.max()), s + "pct_mean x100, max", loc, **ctx))
    # "with lower values than random sets in at least 65 of 70 subjects for each hub-oriented criterion and scale"
    er = pd.concat([p1rows(run, HUB3, "avg2+", "er"), p1rows(run, HUB3, "avg2+", "pr")])
    lower = (er.n - er.n_positive).min()
    out.append(N("MHubErPrLowerMinN", lower, floor_int(lower),
                 P1SRC.format(m="er|pr", k="avg2+") + f"families {HUB3}; min over metric x family x scale of n - n_positive "
                 "(subjects whose criterion ER/PR is not above the random mean; no exact ties assumed); lower bound",
                 loc, c=0.1, k="avg2+ (2,3,5)", bound="at least"))

    # ---- paragraph after Fig. 2
    loc2 = ("subsec:hub_concentration",)
    # "Betweenness and cycle criticality were intermediate, with 50--61\%"
    m = p1rows(run, ("betweenness", "cycle_criticality"), "avg2+")
    s = P1SRC.format(m="q1", k="avg2+") + "families betweenness, cycle_criticality, all scales; "
    ctx = dict(c=0.1, k="avg2+ (2,3,5)", criterion="betweenness+cycle_criticality")
    out.append(N("MQOneIntermediateMin", m.value_mean.min(), pc(m.value_mean.min()), s + "value_mean x100, min", loc2, **ctx))
    out.append(N("MQOneIntermediateMax", m.value_mean.max(), pc(m.value_mean.max()), s + "value_mean x100, max", loc2, **ctx))
    # "(79th--91st percentile of random sets)"
    out.append(N("MIntermediatePctMin", m.pct_mean.min(), pc(m.pct_mean.min()), s + "pct_mean x100, min", loc2, **ctx))
    out.append(N("MIntermediatePctMax", m.pct_mean.max(), pc(m.pct_mean.max()), s + "pct_mean x100, max", loc2, **ctx))

    # "Criteria that selected less strongly connected regions remained closer to random sets (34--49\%, within ten
    #  percentage points of random)" -- the remaining Fig. 2C rows (figures/fig3.py ROWS minus hub3, betweenness,
    #  cycle_criticality)
    rest = ("participation_coefficient", "cycle_participation", "triangle_participation", "cycle_resid", "top_modal_gu", "low_perron")
    r = p1rows(run, rest, "avg2+")
    s = P1SRC.format(m="q1", k="avg2+") + f"families {rest} (remaining Fig. 2C rows), all scales; "
    ctx = dict(c=0.1, k="avg2+ (2,3,5)", criterion="non-hub Fig. 2C rows")
    out.append(N("MQOneNonHubMin", r.value_mean.min(), pc(r.value_mean.min()), s + "value_mean x100, min", loc2, **ctx))
    out.append(N("MQOneNonHubMax", r.value_mean.max(), pc(r.value_mean.max()), s + "value_mean x100, max", loc2, **ctx))
    d = r["mean"].abs().max() * 100
    tok = "ten" if d <= 10 else f"{math.ceil(d - 1e-9)}"
    out.append(N("MNonHubWithinRandomPoints", d, tok,
                 s + "max |mean| x100 (mean over subjects of criterion q1 - random mean, percentage points); bound 'within'; "
                 "word 'ten' only if <= 10, else the rounded-up integer", loc2, **ctx, bound="within"))
    # "Modal-controllability sets were the least concentrated (34\%)"
    mg = p1rows(run, ("top_modal_gu",), "avg2+")
    vals = {int(x.scale): float(x.value_mean) for x in mg.itertuples()}
    for sc in SCALES:
        out.append(N(f"MQOneModalTwoFive{SCALE_WORD[sc]}", vals[sc], pc(vals[sc]),
                     P1SRC.format(m="q1", k="avg2+") + f"family top_modal_gu, scale {sc}; value_mean x100 "
                     "(text prints one '34%' for all scales)", loc2, c=0.1, k="avg2+ (2,3,5)", scale=sc,
                     criterion="top_modal_gu"))
    return out


# --------------------------------------------------------------------------- fig:combining (Fig. 2 caption)



def fig2(run: Path) -> list[Number]:
    out: list[Number] = []
    loc = ("fig:combining",)
    sets = csv(run, "concentration/set_q1.csv.gz")
    x = sets[(sets.scale == 114) & np.isclose(sets.c, 0.1)]

    # ---- panel A: replicates fig2.py Example + Example.spectrum
    deg3 = x[(x.family == "degree_strength") & (x.k == 3)].set_index("subject_id").q1
    sid = (deg3 - deg3.median()).abs().idxmin()
    si = int(sid.split("-")[1])
    xs = x[x.subject_id == sid]
    nodes = lambda s: [int(t) for t in str(s).split(";")]  # noqa: E731
    hub1 = nodes(xs[(xs.family == "degree_strength") & (xs.k == 1)].nodes.iloc[0])
    hubs = nodes(xs[(xs.family == "degree_strength") & (xs.k == 3)].nodes.iloc[0])
    rand = {}
    for k in (1, 3):
        rr = xs[(xs.family == "random") & (xs.k == k)]
        rand[k] = nodes(rr.iloc[(rr.q1 - rr.q1.median()).abs().argsort().iloc[0]].nodes)
    a = np.load(run / "processed" / "connectomes_scale-114.npz")["adjacency"][si]
    _, _, spectrum = gramian_q1_factory(a, 0.1)

    def norm_spec(n):
        w = np.clip(spectrum(n)[::-1], 0, None)
        return w / w.sum()

    spec = {"HubOne": norm_spec(hub1), "RandomOne": norm_spec(rand[1]), "HubThree": norm_spec(hubs),
            "RandomThree": norm_spec(rand[3])}
    asrc = (f"figures/fig2 (recomputed as Example/spectrum): concentration/set_q1.csv.gz + "
            f"processed/connectomes_scale-114.npz | 114 regions, c = 0.1; representative subject {sid} (degree q1 at k=3 "
            "closest to cohort median); random set = the subject's random set with q1 closest to its median; ")
    desc = {"HubOne": "top-1 degree region", "RandomOne": "representative random region (k=1)",
            "HubThree": "top-3 degree regions", "RandomThree": "representative random 3-set"}
    # "A single hub spreads ... ($q_1 = 0.90$ against 0.99), but three hubs keep one dominant pattern ($q_1 = 0.75$),
    #  whereas three random regions ... ($q_1 = 0.34$)"
    for key, w in spec.items():
        out.append(N(f"MFigTwoQOne{key}", w[0], fx(w[0]), asrc + f"q1 = first normalized eigenvalue, {desc[key]}", loc,
                     scale=114, c=0.1, subject=sid))
    # "Eigenvalues beyond the sixth hold at most 0.2\% of the total"
    rest = max(float(w[6:].sum()) for w in spec.values())
    out.append(N("MFigTwoTailMaxPct", rest, ceil_dp(100 * rest, 1),
                 asrc + "max over the four spectra of sum of normalized eigenvalues 7..N, x100; bound rounded up", loc,
                 scale=114, c=0.1, subject=sid, bound="at most"))

    # ---- panel B: replicates fig2.py panel_k (mean over subjects of per-subject mean q1)
    def bmean(fam, k):
        return float(x[(x.family == fam) & (x.k == k)].groupby("subject_id").q1.mean().mean())

    bsrc = ("figures/fig2 (recomputed as panel_k): concentration/set_q1.csv.gz | scale 114, c = 0.1, "
            "family == '{f}', k == {k}; mean over subjects of the per-subject mean q1")
    # "Degree-strength sets fall below random sets with one driver (0.87 against 0.95)"
    # "and lie above them for every $k \geq 2$ (0.69 against 0.25 at $k = 5$)"
    for fam, nm in (("degree_strength", "Degree"), ("random", "Random")):
        for k, kn in ((1, "One"), (5, "Five")):
            v = bmean(fam, k)
            out.append(N(f"MFigTwoQOne{nm}K{kn}", v, fx(v), bsrc.format(f=fam, k=k), loc, scale=114, c=0.1, k=k,
                         criterion=fam, stat="mean over subjects"))
    # (claim "lie above them for every k >= 2" checked in the report, not a number)

    # ---- panel C: replicates fig2.py panel_criteria band
    al = csv(run, "mechanisms/gramian_alignment.csv")
    aa = al[(al.scale == 114) & (al.k == 3)]
    rm = aa[aa.family == "random"].groupby("subject_id").q1.mean().to_numpy()
    lo, hi = np.percentile(rm, [5, 95])
    csrc = ("figures/fig2 (recomputed as panel_criteria): mechanisms/gramian_alignment.csv "
            "(written by mechanisms/08_gramian_alignment) | scale 114, k == 3, family == 'random'; per-subject mean "
            "q1, then np.percentile {}")
    # "the band spans the 5th to 95th percentile of these per-subject averages (0.35--0.38)"
    out.append(N("MFigTwoRandomBandLo", lo, fx(lo), csrc.format("5"), loc, scale=114, k=3, c=0.1))
    out.append(N("MFigTwoRandomBandHi", hi, fx(hi), csrc.format("95"), loc, scale=114, k=3, c=0.1))
    return out


# --------------------------------------------------------------------------- entry point

def collect(run: Path) -> list[Number]:
    run = Path(run)
    out = energy_floor(run) + fig1(run) + hub_concentration(run) + fig2(run)
    names = [n.name for n in out]
    assert len(names) == len(set(names)), "duplicate macro names"
    assert all(n.isalpha() for n in names)
    return out


# Tokens as printed in the manuscript.
PRINTED = {
    "MFracBelowEpsMin": "86", "MFracBelowEpsMax": "96", "MLogEavgFloorGap": "0.07", "MLambdaMinEqEpsNSets": "302,400",
    "MLambdaMinEqEpsAll": "all",
    "MFigOneHubLoading": "0.48", "MFigOneHubQOne": "0.78", "MFigOneRandomMedianLoading": "0",
    "MFigOneRandomMedianQOne": "0.33", "MFigOneRandomReachHub": "none",
    "MQOneHubTwoFiveMin": "67", "MQOneHubTwoFiveMax": "78", "MQOneRandomTwoFiveMin": "36", "MQOneRandomTwoFiveMax": "39",
    "MHubAboveRandomEvery": "every", "MHubPctTwoFiveMin": "99", "MHubPctTwoFiveMax": "100", "MHubErPrLowerMinN": "65",
    "MQOneIntermediateMin": "50", "MQOneIntermediateMax": "61", "MIntermediatePctMin": "79", "MIntermediatePctMax": "91",
    "MQOneNonHubMin": "34", "MQOneNonHubMax": "49", "MNonHubWithinRandomPoints": "ten",
    "MQOneModalTwoFiveSixtyEight": "34", "MQOneModalTwoFiveOneFourteen": "34", "MQOneModalTwoFiveTwoNineteen": "34",
    "MFigTwoQOneHubOne": "0.90", "MFigTwoQOneRandomOne": "0.99", "MFigTwoQOneHubThree": "0.75",
    "MFigTwoQOneRandomThree": "0.34", "MFigTwoTailMaxPct": "0.2",
    "MFigTwoQOneDegreeKOne": "0.87", "MFigTwoQOneRandomKOne": "0.95", "MFigTwoQOneDegreeKFive": "0.69",
    "MFigTwoQOneRandomKFive": "0.25", "MFigTwoRandomBandLo": "0.35", "MFigTwoRandomBandHi": "0.38",
}
