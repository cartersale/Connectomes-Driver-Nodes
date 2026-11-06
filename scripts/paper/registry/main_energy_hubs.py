"""Main text: average energy and the regularization floor (subsec:energy_floor) and hub concentration with two to
five drivers (subsec:hub_concentration).

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

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


def floor_int(x: float) -> str:
    """Lower bound ("at least x")."""
    return str(int(math.floor(x + 1e-9)))


@lru_cache(maxsize=None)
def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def csv(run: Path, rel: str) -> pd.DataFrame:
    return _csv(str((run / rel).resolve()))


def N(name, value, text, source, locus, **context) -> Number:
    if isinstance(locus, str):
        locus = (locus,)
    return Number(name=name, value=float(value), text=text, source=source, locus=tuple(locus), context=context)


# --------------------------------------------------------------------------- subsec:energy_floor

def energy_floor(run: Path) -> list[Number]:
    out: list[Number] = []
    loc = ("subsec:energy_floor",)
    a = csv(run, "sensitivity/energy_floor_driver_sets.csv")
    e5 = a[np.isclose(a.eps, 1e-5)]
    g = e5.groupby(["scale", "method"])[["fraction_below_eps"]].mean()
    tsrc = ("sensitivity/04_energy_floor: sensitivity/energy_floor_driver_sets.csv | c = 0.1, eps == 1e-5, methods "
            "degree_strength, eigenvector_centrality, betweenness, cycle_participation, random_mean (mean of the 100 "
            "random sets); mean over subjects and k in {1,2,3,5} per (scale, method); ")

    # "Most eigenvalues of the controllability Gramian (86--96\%) fell below the regularization threshold"
    fb = g["fraction_below_eps"]
    out.append(N("MFracBelowEpsMin", fb.min(), pc(fb.min()), tsrc + "fraction_below_eps, min over scales x methods (x100)",
                 loc, eps=1e-5, stat="min over scale x method of cohort mean"))
    out.append(N("MFracBelowEpsMax", fb.max(), pc(fb.max()), tsrc + "fraction_below_eps, max over scales x methods (x100)",
                 loc, eps=1e-5, stat="max over scale x method of cohort mean"))
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

    # "With two to five drivers, sets selected by degree strength, eigenvector centrality, or average controllability
    #  concentrated 67--78\% of their controllability on a single pattern"
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
    # "Effective rank and participation ratio showed the same pattern, with lower values than random sets ... in at
    #  least 65 of 70 subjects for each criterion and scale"
    er = pd.concat([p1rows(run, HUB3, "avg2+", "er"), p1rows(run, HUB3, "avg2+", "pr")])
    lower = (er.n - er.n_positive).min()
    out.append(N("MHubErPrLowerMinN", lower, floor_int(lower),
                 P1SRC.format(m="er|pr", k="avg2+") + f"families {HUB3}; min over metric x family x scale of n - n_positive "
                 "(subjects whose criterion ER/PR is not above the random mean; no exact ties assumed); lower bound",
                 loc, c=0.1, k="avg2+ (2,3,5)", bound="at least"))

    # ---- paragraph after Fig. 2
    loc2 = ("subsec:hub_concentration",)
    # "Betweenness and cycle criticality showed intermediate concentration (50--61\%)"
    m = p1rows(run, ("betweenness", "cycle_criticality"), "avg2+")
    s = P1SRC.format(m="q1", k="avg2+") + "families betweenness, cycle_criticality, all scales; "
    ctx = dict(c=0.1, k="avg2+ (2,3,5)", criterion="betweenness+cycle_criticality")
    out.append(N("MQOneIntermediateMin", m.value_mean.min(), pc(m.value_mean.min()), s + "value_mean x100, min", loc2, **ctx))
    out.append(N("MQOneIntermediateMax", m.value_mean.max(), pc(m.value_mean.max()), s + "value_mean x100, max", loc2, **ctx))
    # "participation coefficient, triangle participation, cycle participation, degree-residualized cycle participation,
    #  modal controllability, and lowest eigenvector centrality remained closer to random selection (34--49\%)"
    rest = ("participation_coefficient", "cycle_participation", "triangle_participation", "cycle_resid", "top_modal_gu", "low_perron")
    r = p1rows(run, rest, "avg2+")
    s = P1SRC.format(m="q1", k="avg2+") + f"families {rest} (remaining Fig. 2C rows), all scales; "
    ctx = dict(c=0.1, k="avg2+ (2,3,5)", criterion="non-hub Fig. 2C rows")
    out.append(N("MQOneNonHubMin", r.value_mean.min(), pc(r.value_mean.min()), s + "value_mean x100, min", loc2, **ctx))
    out.append(N("MQOneNonHubMax", r.value_mean.max(), pc(r.value_mean.max()), s + "value_mean x100, max", loc2, **ctx))
    return out


# --------------------------------------------------------------------------- entry point

def collect(run: Path) -> list[Number]:
    run = Path(run)
    out = energy_floor(run) + hub_concentration(run)
    names = [n.name for n in out]
    assert len(names) == len(set(names)), "duplicate macro names"
    assert all(n.isalpha() for n in names)
    return out


# Tokens as printed in the manuscript.
PRINTED = {
    "MFracBelowEpsMin": "86", "MFracBelowEpsMax": "96",
    "MQOneHubTwoFiveMin": "67", "MQOneHubTwoFiveMax": "78", "MQOneRandomTwoFiveMin": "36", "MQOneRandomTwoFiveMax": "39",
    "MHubAboveRandomEvery": "every", "MHubPctTwoFiveMin": "99", "MHubPctTwoFiveMax": "100", "MHubErPrLowerMinN": "65",
    "MQOneIntermediateMin": "50", "MQOneIntermediateMax": "61",
    "MQOneNonHubMin": "34", "MQOneNonHubMax": "49",
}
