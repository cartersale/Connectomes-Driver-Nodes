"""Numbers quoted in the prose and captions of the Supporting Information. Table bodies are built by
scripts/paper/tables.py.

See the package docstring for the formatting conventions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SC = (68, 114, 219)
SCN = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}
NULLS = ("strength", "geometric", "weights")
YEO7 = ("net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default")
_FRACTION_WORDS = {2: "one half", 3: "one third", 4: "one quarter", 5: "one fifth", 6: "one sixth"}
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


from .record import Number  # noqa: E402

# ----------------------------------------------------------------------------- formatting

def _f(x, d=2):
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _pct(x):
    """Fraction -> integer percent text."""
    return f"{100 * x:.0f}"


def _ceil(x, d):
    f = 10.0 ** d
    return math.ceil(round(x * f, 9)) / f


def _sig1_ceil(x: float) -> str:
    """One significant figure rounded UP (bound), LaTeX exponent form: 4.1e-15 -> '5\\times10^{-15}'."""
    e = math.floor(math.log10(abs(x)))
    m = math.ceil(round(x / 10**e, 9))
    if m == 10:
        m, e = 1, e + 1
    return f"{m}\\times10^{{{e}}}"


# ----------------------------------------------------------------------------- I/O + record

class _Ctx:
    def __init__(self, run: Path):
        self.run = Path(run)
        self.out: list[Number] = []

    def path(self, rel: str) -> Path:
        p = self.run / rel
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    def csv(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.path(rel), **kw)

    def add(self, name, value, computed, printed, source, locus, context=None, bound=None):
        """``computed``: text from the saved output; ``printed``: the token in supp_mat.tex.
        ``bound``: (holds: bool, description) for statements of a bound ("at most", "within")."""
        ctx = dict(context or {})
        if bound is not None:
            ctx["bound"] = bound[1]
            ctx["bound_holds"] = bool(bound[0])
        ctx["printed"] = printed
        if isinstance(locus, str):
            locus = (locus,)
        self.out.append(Number(name, float(value), computed, source, tuple(locus), ctx))



def collect(run: Path) -> list[Number]:
    c = _Ctx(run)
    for fn in (_primer_eps, _cycle_spin, _damping, _setlevel, _nulls, _stability, _targets, _lesions):
        fn(c)
    names = [n.name for n in c.out]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, dup
    return c.out


# ============================================================================ sec:primer / sec:eps
def _primer_eps(c: _Ctx) -> None:
    # sec:primer & sec:eps: "86--87\%, 92\%, and 95--96\% of the eigenvalues of W were below eps";
    # sec:eps: "zero to numerical precision (70--71\%, 81--82\%, and 90\% at eps = 10^{-5})"
    a = c.csv("sensitivity/energy_floor_driver_sets.csv")
    asub = a.groupby(["scale", "eps", "method", "subject_id"])[["fraction_at_floor", "fraction_below_eps"]].mean().reset_index()
    aagg = asub.groupby(["scale", "eps", "method"]).mean(numeric_only=True).reset_index()
    src_frac = ("sensitivity/04_energy_floor: sensitivity/energy_floor_driver_sets.csv | eps == 1e-5; 5 set types "
                "(degree, eigenvector, cycle, betweenness, random_mean); per-subject k-mean, subject mean; min/max "
                "across set types, rounded to 2 dp as in S4_2, x100; column {}")
    ms = {("fraction_at_floor", 68): ("70", "71"), ("fraction_at_floor", 114): ("81", "82"),
          ("fraction_at_floor", 219): ("90",),
          ("fraction_below_eps", 68): ("86", "87"), ("fraction_below_eps", 114): ("92",),
          ("fraction_below_eps", 219): ("95", "96")}
    for col, tag, locus in (("fraction_at_floor", "NumericallyZero", ("sec:eps",)),
                            ("fraction_below_eps", "BelowEps", ("sec:primer", "sec:eps"))):
        for s in SC:
            g = aagg[(aagg.scale == s) & np.isclose(aagg.eps, 1e-5)][col]
            lo, hi = round(100 * float(f"{g.min():.2f}")), round(100 * float(f"{g.max():.2f}"))
            base = f"SEpsFrac{tag}{SCN[s]}"
            ctx = {"scale": s, "eps": 1e-5, "c": 0.1, "stat": "range across 5 set types of cohort means"}
            m = ms[(col, s)]
            if len(m) == 2:
                c.add(base + "Lo", 100 * g.min(), str(lo), m[0], src_frac.format(col), locus, ctx)
                c.add(base + "Hi", 100 * g.max(), str(hi), m[1], src_frac.format(col), locus, ctx)
            else:
                txt = str(lo) if lo == hi else f"{lo}--{hi}"
                c.add(base, 100 * g.mean(), txt, m[0], src_frac.format(col) + " (min and max round to the same value)",
                      locus, ctx)

    # sec:eps: "mean log10 E_avg lay within about 0.1 of log10(N/eps)"
    tidy = c.csv("sensitivity/horizon_eps_largek_tidy.csv.gz")
    t = tidy[(tidy.c == 0.1) & np.isinf(tidy["T"]) & tidy.k.isin([1, 2, 3, 5])
             & tidy.method.isin(["degree_strength", "cycle_participation"])]
    sub = t.groupby(["scale", "eps", "method", "subject_id"]).log10_average_energy.mean().reset_index()
    agg = sub.groupby(["scale", "eps", "method"]).log10_average_energy.mean().reset_index()
    gapmax = float((np.log10(agg.scale / agg.eps) - agg.log10_average_energy).abs().max())
    c.add("SEpsEnergyWithinFloorBound", gapmax, f"{gapmax:.1f}", "0.1",
          "sensitivity/03_horizon_eps_largek: sensitivity/horizon_eps_largek_tidy.csv.gz | c == 0.1, T == inf, "
          "k in 1,2,3,5, degree/cycle; max over scale x eps x method of |log10(N/eps) - mean log10 E_avg| (S4_1 cols 4-6)",
          "sec:eps", {"c": 0.1, "stat": "max over scale x eps x method", "note": "'about' -> nearest rounding"})

    # sec:eps: "and changed by one unit per tenfold change in eps"
    slopes = {}
    for (s, m), g in agg.groupby(["scale", "method"]):
        slopes[(s, m)] = float(-np.polyfit(np.log10(g.eps), g.log10_average_energy, 1)[0])
    sv = np.array(list(slopes.values()))
    c.add("SEpsEnergyTenfoldSlope", sv.mean(), "one" if np.all(np.round(sv) == 1) else _f(sv.mean()), "one",
          "sensitivity/03_horizon_eps_largek: sensitivity/horizon_eps_largek_tidy.csv.gz | same cohort means; "
          "least-squares slope of mean log10 E_avg on -log10(eps) per scale x method "
          f"(range {sv.min():.3f} to {sv.max():.3f}); word if every slope rounds to 1",
          "sec:eps", {"c": 0.1, "stat": "slope, mean over scale x method"})

    # sec:eps: "at most 2.4e-3 in magnitude ... (-2.3e-3 at eps = 1e-8, +2.4e-3 at eps = 1e-2)"
    s1 = c.csv("sensitivity/epsilon_summary.csv")
    e = s1[(s1.c == 0.1) & (s1.metric == "log10_average_energy")]
    src_e = ("sensitivity/03_horizon_eps_largek: sensitivity/epsilon_summary.csv | c == 0.1, "
             "metric == log10_average_energy; mean_diff (cycle - degree, k-averaged; S4_1 column 7)")
    mx = float(e.mean_diff.abs().max())
    c.add("SEpsEnergyCycleDegreeMaxAbs", mx, f"{_ceil(mx * 1e3, 1):.1f}\\times10^{{-3}}", "2.4\\times10^{-3}",
          src_e + "; max |mean_diff| over scales x eps, rounded up", "sec:eps",
          {"c": 0.1, "comparison": "cycle - degree", "stat": "max |cohort mean| over 3 scales x 7 eps"},
          bound=(mx <= 2.4e-3, "<="))
    for eps, nm, sign, ms_ in ((1e-8, "EpsMinusEight", "", "-2.3\\times10^{-3}"),
                               (1e-2, "EpsMinusTwo", "+", "+2.4\\times10^{-3}")):
        v = float(e[(e.scale == 68) & np.isclose(e.eps, eps)].mean_diff.iloc[0])
        c.add(f"SEpsEnergyCycleDegreeSixtyEight{nm}", v, f"{sign}{v * 1e3:.1f}\\times10^{{-3}}", ms_,
              src_e + f"; scale 68, eps {eps:g}", "sec:eps", {"scale": 68, "eps": eps, "c": 0.1})

    # sec:eps: "interquartile range 6.52, 3.97, and 2.59 x 10^{-3} log units" / "0.36 to 0.38 log units"
    L = c.csv("sensitivity/energy_floor_landscape.csv")
    Lagg = L.groupby(["scale", "eps"])[["iqr_log10E", "iqr_log10E_above_eps"]].mean().reset_index()
    L5 = Lagg[np.isclose(Lagg.eps, 1e-5)].set_index("scale")
    src_l = ("sensitivity/04_energy_floor: sensitivity/energy_floor_landscape.csv | eps == 1e-5; mean over subjects x k "
             "of the per-subject IQR across candidate sets; ")
    for s, ms_ in zip(SC, ("6.52", "3.97", "2.59")):
        v = float(L5.loc[s, "iqr_log10E"])
        c.add(f"SEpsLandscapeIQR{SCN[s]}", v * 1e3, f"{v * 1e3:.2f}", ms_,
              src_l + "iqr_log10E in units of 1e-3 (S4_2 'All (x1e-3)')", "sec:eps", {"scale": s, "eps": 1e-5})
    ab = L5.iqr_log10E_above_eps
    c.add("SEpsLandscapeIQRAboveEpsLo", ab.min(), f"{ab.min():.2f}", "0.36", src_l + "iqr_log10E_above_eps, min over scales",
          "sec:eps", {"eps": 1e-5})
    c.add("SEpsLandscapeIQRAboveEpsHi", ab.max(), f"{ab.max():.2f}", "0.38", src_l + "iqr_log10E_above_eps, max over scales",
          "sec:eps", {"eps": 1e-5})


# ============================================================================ sec:cycle
def _cycle_spin(c: _Ctx) -> None:
    spin = c.csv("target_states/spin_tests.csv")
    # "strongly correlated with degree strength (Spearman rho = 0.75, 0.71, and 0.61)"
    for s, ms_ in zip(SC, ("0.75", "0.71", "0.61")):
        r = spin[(spin.scale == s) & (spin.x == "degree_strength") & (spin.y == "cycle_participation")].iloc[0]
        c.add(f"SCycleDegreeRho{SCN[s]}", r.spearman_rho, _f(r.spearman_rho), ms_,
              "target_states/05_spin_tests: target_states/spin_tests.csv | x == degree_strength, "
              "y == cycle_participation; spearman_rho (as S10_9)", "sec:cycle", {"scale": s, "stat": "Spearman across parcels"})
    # "11, 4, and 5 of the 15 highest-ranked regions ... also among the 15 highest-ranked by degree strength"
    top = c.csv("target_states/top15_network_composition.csv")
    for s, ms_ in zip(SC, ("11", "4", "5")):
        n = top[(top.scale == s) & (top.category == "both")].n.iloc[0]
        c.add(f"STopFifteenShared{SCN[s]}", n, str(int(n)), ms_,
              "target_states/06_network_enrichment: target_states/top15_network_composition.csv | category == both; n. "
              "Ranks the subject-mean node scores, not scores of the cohort-average connectome ('subject-averaged connectome')",
              "sec:cycle", {"scale": s, "k": 15})
    # "rank divergence was weakly correlated with the gradient (rho = -0.28, -0.26, and -0.16) ...
    #       (p_spin = 0.16, 0.12, and 0.23)"
    for s, mr, mp in zip(SC, ("-0.28", "-0.26", "-0.16"), ("0.16", "0.12", "0.23")):
        r = spin[(spin.scale == s) & (spin.x == "rank_divergence") & (spin.y == "G1_registered")].iloc[0]
        src = "target_states/05_spin_tests: target_states/spin_tests.csv | x == rank_divergence, y == G1_registered; "
        c.add(f"SRankDivGOneRho{SCN[s]}", r.spearman_rho, _f(r.spearman_rho), mr, src + "spearman_rho", "sec:cycle",
              {"scale": s})
        c.add(f"SRankDivGOnePspin{SCN[s]}", r.p_spin_vasa, _f(r.p_spin_vasa), mp, src + "p_spin_vasa (2 dp)", "sec:cycle",
              {"scale": s, "test": "Vasa spin, 10,000 rotations"})
    # "The only nominal effects were a degree priority in the ventral attention network at 68 and 114
    #       regions and a cycle priority in the dorsal attention network at 219 regions (uncorrected p <= 0.05)"
    en = c.csv("target_states/rank_divergence_network_enrichment.csv")
    cells = en[((en.network == "VentAttn") & en.scale.isin([68, 114])) | ((en.network == "DorsAttn") & (en.scale == 219))]
    others = en.drop(cells.index)
    pmax = float(cells.p_spin_vasa.max())
    signs_ok = bool((cells[cells.network == "VentAttn"].mean_rank_divergence < 0).all()
                    and (cells[cells.network == "DorsAttn"].mean_rank_divergence > 0).all())
    only = bool((others.p_spin_vasa > 0.05).all())
    c.add("SRankDivNominalPBound", pmax, _f(_ceil(pmax, 2)), "0.05",
          "target_states/06_network_enrichment: target_states/rank_divergence_network_enrichment.csv | "
          "max p_spin_vasa over VentAttn@68, VentAttn@114, DorsAttn@219 "
          f"(values {dict(zip(cells.network + '@' + cells.scale.astype(str), cells.p_spin_vasa.round(4)))}); "
          f"signs as stated: {signs_ok}; all other cells p > 0.05: {only} (min other {others.p_spin_vasa.min():.3f}); "
          "bound rounded up to 2 dp", "sec:cycle",
          {"stat": "uncorrected Vasa spin p", "comparison": "network mean rank divergence vs spin null"},
          bound=(pmax <= 0.05 and signs_ok and only, "<="))


# ============================================================================ sec:damping
def _damping(c: _Ctx) -> None:
    # "the cycle-minus-degree difference changed sign in 8, 4, and 20 subjects"
    fl = c.csv("concentration/decay_flip_summary.csv")
    r = fl[(fl.curve == "cd_q1") & (fl.axis == "flip_c")].iloc[0]
    for s, ms_ in zip(SC, ("8", "4", "20")):
        v = int(r[f"n_flip_{s}"])
        c.add(f"SCsweepQOneSignChangeSubjects{SCN[s]}", v, str(v), ms_,
              "concentration/03_decay_collapse: concentration/decay_flip_summary.csv | curve == cd_q1, axis == flip_c; "
              f"n_flip_{s} (S5_7)", "sec:damping", {"scale": s, "stat": "count of subjects", "grid": "25 c values"})
    # "the effective-rank difference changed sign at c ~ 0.88, 0.69, and 0.29"
    sf = c.csv("sensitivity/csweep_signflip.csv")
    for s, ms_ in zip(SC, ("0.88", "0.69", "0.29")):
        v = float(sf[(sf.metric == "effective_rank") & (sf.scale == s)].c_flip_mean.iloc[0])
        c.add(f"SCsweepERFlipC{SCN[s]}", v, f"{v:.2f}", ms_,
              "sensitivity/05_summarize_csweep: sensitivity/csweep_signflip.csv | metric == effective_rank; "
              "c_flip_mean (S5_5 'Mean curve')", "sec:damping", {"scale": s, "grid": "13 c values"})

    # decay-ratio collapse
    col = c.csv("concentration/decay_collapse.csv")
    src_c = "concentration/03_decay_collapse: concentration/decay_collapse.csv | column D (S5_8 matched-window rows); "
    red = []
    for curve, nm, mc, md in (("cd_q1", "CycleDegreeQOne", "0.022", "0.015"), ("cd_er", "CycleDegreeER", "0.037", "0.026"),
                              ("dr_q1", "DegreeRandomQOne", "0.030", "0.021")):
        dc = float(col[(col.curve == curve) & (col.axis == "c (matched window)")].D.iloc[0])
        dd = float(col[(col.curve == curve) & (col.axis == "decay_ratio (matched window)")].D.iloc[0])
        red.append(1 - dd / dc)
        c.add(f"SCollapseD{nm}C", dc, f"{dc:.3f}", mc, src_c + f"curve == {curve}, axis == 'c (matched window)'",
              "sec:damping", {"curve": curve, "axis": "c"})
        c.add(f"SCollapseD{nm}Decay", dd, f"{dd:.3f}", md, src_c + f"curve == {curve}, axis == 'decay_ratio (matched window)'",
              "sec:damping", {"curve": curve, "axis": "decay ratio"})
    pct = 100 * float(np.mean(red))
    c.add("SCollapseDispersionDropPct", pct, f"{round(pct, -1):.0f}", "30",
          src_c + f"mean over the 3 curves of 1 - D(d)/D(c) (per curve {[round(100 * x, 1) for x in red]}), x100, "
          "rounded to 10 ('about 30%')", "sec:damping", {"stat": "mean relative drop"})
    # "ratio of the largest to the smallest scale median 1.4, against 2.0 on the c axis"
    src_f = "concentration/03_decay_collapse: concentration/decay_flip_summary.csv | curve == cd_er; max_over_min (S5_7); "
    for axis, nm, ms_ in (("flip_decay", "Decay", "1.4"), ("flip_c", "C", "2.0")):
        v = float(fl[(fl.curve == "cd_er") & (fl.axis == axis)].max_over_min.iloc[0])
        c.add(f"SCollapseERFlipSpread{nm}", v, f"{v:.1f}", ms_, src_f + f"axis == {axis}", "sec:damping", {"axis": axis})
    # "(Kruskal--Wallis p < 10^{-3})"
    kp = float(fl[(fl.curve == "cd_er") & (fl.axis == "flip_decay")].kruskal_p.iloc[0])
    c.add("SCollapseERFlipKruskalBound", kp, f"10^{{{math.ceil(math.log10(kp))}}}" if kp >= 1e-3 else "10^{-3}", "10^{-3}",
          src_f + f"axis == flip_decay; kruskal_p ({kp:.2g}), reported as the bound p < 1e-3", "sec:damping",
          {"test": "Kruskal-Wallis across scales"}, bound=(kp < 1e-3, "<"))
    # "the mean gap differs little between scales (0.43, 0.54, and 0.44)"
    dcv = c.csv("concentration/decay_curves.csv.gz", usecols=["subject_id", "scale", "gap"])
    gaps = dcv.drop_duplicates(["scale", "subject_id"]).groupby("scale").gap.mean()
    for s, ms_ in zip(SC, ("0.43", "0.54", "0.44")):
        v = float(gaps.loc[s])
        c.add(f"SCollapseMeanGap{SCN[s]}", v, f"{v:.2f}", ms_,
              "concentration/03_decay_collapse: concentration/decay_curves.csv.gz | gap (l1 - l2 of A0) per subject, mean over "
              "70 subjects", "sec:damping", {"scale": s, "stat": "mean over subjects"})

    # "In 72 checked combinations ... differed by the factor l1 + 1 to within 4e-15 ... with eps rescaled ...
    #        agreed to within 3e-14"
    det = c.csv("sensitivity/nctpy_equivalence_detail.csv")
    src_n = "sensitivity/02_nctpy_equivalence: sensitivity/nctpy_equivalence_detail.csv | "
    c.add("SNctpyChecks", len(det), str(len(det)), "72", src_n + "number of rows (subject x criterion x k checks)",
          "sec:damping", {"stat": "count of checks"})
    v = float(det.max_abs_W_diff.max())
    c.add("SNctpyMaxGramianDiff", v, _sig1_ceil(v), "4\\times10^{-15}",
          src_n + f"max of max_abs_W_diff = max |W_nctpy - (l1 + 1) W_ours| = {v:.3g}; bound, 1 sig. fig. rounded up "
          "(rounding to nearest also gives 4e-15)", "sec:damping", {"stat": "max over 72 checks"},
          bound=(v <= 4e-15, "<="))
    v = float(det.d_er_eps_rescaled.abs().max())
    c.add("SNctpyMaxRescaledDiff", v, _sig1_ceil(v), "3\\times10^{-14}",
          src_n + f"max |d_er_eps_rescaled| = {v:.3g} (ER only: no rescaled-eps PR column is saved; prose says ER and PR); "
          "bound, 1 sig. fig. rounded up", "sec:damping", {"stat": "max over 72 checks"}, bound=(v <= 3e-14, "<="))


# ============================================================================ sec:setlevel
def _setlevel(c: _Ctx) -> None:
    # "The upper bound was close (median gap 0.01 to 0.04 at c = 0.1)"
    g = c.csv("modal_controllability/driver_gram_proxy.csv")
    gap = g[g.c == 0.1].median_q1G_minus_q1raw
    src = ("modal_controllability/02_analyze_modal: modal_controllability/driver_gram_proxy.csv | c == 0.1; k 2,3,5 x 3 scales; {} of "
           "median_q1G_minus_q1raw (S7_3 'upper'), 2 dp")
    c.add("SDriverGramUpperGapLo", gap.min(), _f(gap.min()), "0.01", src.format("min"), "sec:setlevel", {"c": 0.1})
    c.add("SDriverGramUpperGapHi", gap.max(), _f(gap.max()), "0.04", src.format("max"), "sec:setlevel", {"c": 0.1})

    # "They shared almost no regions (one region in one subject across all scales and sizes)"
    s = c.csv("modal_controllability/set_level_modal.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "sample", "nodes"])
    s = s[(s.c == 0.1) & (s["sample"] == -1) & s.family.isin(["top_modal_gu", "cycle_participation"])]
    w = s.pivot_table(index=["subject_id", "scale", "k"], columns="family", values="nodes", aggfunc="first").dropna()
    shared = w.apply(lambda r: len(set(r.top_modal_gu.split()) & set(r.cycle_participation.split())), axis=1)
    n_subj = shared[shared > 0].index.get_level_values("subject_id").nunique()
    n_reg = int(shared.sum())
    src = ("modal_controllability/01_compute_modal: modal_controllability/set_level_modal.csv.gz | c == 0.1, sample == -1, top_modal_gu vs "
           "cycle_participation, all scales x k 2,3,5; {}; printed as a word")
    c.add("SModalCycleSharedSubjects", n_subj, _COUNT_WORDS.get(n_subj, str(n_subj)), "one",
          src.format("subjects with any shared region"), "sec:setlevel", {"c": 0.1, "criterion": "top_modal_gu"})
    c.add("SModalCycleSharedRegions", n_reg, _COUNT_WORDS.get(n_reg, str(n_reg)), "one",
          src.format("total shared regions summed over subject x scale x k cells"), "sec:setlevel",
          {"c": 0.1, "criterion": "top_modal_gu"})

    # "at k = 3 cycle-participation sets had higher effective rank (median differences 0.44, 0.48, and 0.74;
    #        positive in 55, 65, and 69 subjects)"
    cf = c.csv("modal_controllability/cycle_vs_counterfactual.csv")
    cf = cf[(cf.c == 0.1) & (cf.k == 3) & (cf.contrast == "cycle - top_modal_gu") & (cf.target == "effective_rank")]
    for sc, mm, mn in zip(SC, ("0.44", "0.48", "0.74"), ("55", "65", "69")):
        r = cf[cf.scale == sc].iloc[0]
        src = ("modal_controllability/02_analyze_modal: modal_controllability/cycle_vs_counterfactual.csv | c == 0.1, k == 3, contrast == "
               "'cycle - top_modal_gu', target == effective_rank; {} (S7_5)")
        ctx = {"scale": sc, "k": 3, "c": 0.1, "comparison": "cycle - top-k modal (Gu)"}
        c.add(f"SModalCycleERMedianDiff{SCN[sc]}", r["median"], _f(r["median"]), mm, src.format("median, 2 dp"),
              "sec:setlevel", ctx)
        c.add(f"SModalCycleERNPos{SCN[sc]}", r.n_pos, str(int(r.n_pos)), mn, src.format("n_pos"), "sec:setlevel", ctx)


# ============================================================================ sec:nulls
def _nulls(c: _Ctx) -> None:
    fid = c.csv("null_models/null_fidelity.csv")
    st = fid[fid.null == "strength"]
    src = "null_models/02_analyze: null_models/null_fidelity.csv | null == strength; {col} over scales (S8_1, {d} dp)"
    # "(correlation 0.998 to 0.999; largest relative error 0.16 to 0.27)"
    ctx = {"null": "strength", "stat": "mean over subjects x 50 nulls"}
    c.add("SNullStrengthCorrLo", st.strength_r.min(), _f(st.strength_r.min(), 3), "0.998", src.format(col="min strength_r", d=3),
          "sec:nulls", ctx)
    c.add("SNullStrengthCorrHi", st.strength_r.max(), _f(st.strength_r.max(), 3), "0.999", src.format(col="max strength_r", d=3),
          "sec:nulls", ctx)
    c.add("SNullStrengthMaxRelErrLo", st.strength_max_rel_err.min(), _f(st.strength_max_rel_err.min()), "0.16",
          src.format(col="min strength_max_rel_err", d=2), "sec:nulls", ctx)
    c.add("SNullStrengthMaxRelErrHi", st.strength_max_rel_err.max(), _f(st.strength_max_rel_err.max()), "0.27",
          src.format(col="max strength_max_rel_err", d=2), "sec:nulls", ctx)
    # "kept binary degree and the mean connection length (within 1\%)"
    real = fid[fid.null == "real"].set_index("scale").mean_edge_length
    geo = fid[fid.null == "geometric"].set_index("scale").mean_edge_length
    dev = float((100 * (geo - real).abs() / real).max())
    c.add("SNullGeomEdgeLengthWithinPct", dev, _f(_ceil(dev, 0), 0), "1",
          "null_models/02_analyze: null_models/null_fidelity.csv | max over scales of |geometric - real| / real "
          f"mean_edge_length, % ({dev:.2f}%)", "sec:nulls", {"null": "geometric"}, bound=(dev < 1, "<"))

    # "slightly above it at 114 regions (median z 0.6 to 0.9)"
    gs = c.csv("null_models/gap_split.csv")
    z = gs[(gs.stat == "lambda2") & (gs.scale == 114) & gs.null.isin(NULLS)].z_median
    src = "null_models/03_gap_mediators: null_models/gap_split.csv | stat == lambda2, scale == 114, 3 nulls; {} z_median"
    c.add("SNullLambdaTwoZOneFourteenLo", z.min(), f"{z.min():.1f}", "0.6", src.format("min"), "sec:nulls",
          {"scale": 114, "stat": "median z over subjects, range over 3 nulls"})
    c.add("SNullLambdaTwoZOneFourteenHi", z.max(), f"{z.max():.1f}", "0.9", src.format("max"), "sec:nulls",
          {"scale": 114, "stat": "median z over subjects, range over 3 nulls"})

    # "the two were almost perfectly correlated (median Pearson r = 0.998)"
    ps = c.csv("null_models/rayleigh/per_subject.csv")
    ps["sl_R1"] = ps["d_R1"] / ps["d_lambda1"]
    pss = ps[ps.null == "strength"]
    r = pss.pearson_rW_sTWs.median()
    c.add("SRayleighPearsonRWsAs", r, _f(r, 3), "0.998",
          "null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == strength; median over all "
          "subjects x scales of pearson_rW_sTWs", "sec:nulls", {"null": "strength", "stat": "median over subjects x scales"})
    # "R1 guaranteed a median of 60\%, 43\%, and 36\% of the excess in l1"
    med = pss.groupby("scale").median(numeric_only=True)
    for sc, ms_ in zip(SC, ("60", "43", "36")):
        v = med.loc[sc, "sl_R1"]
        c.add(f"SRayleighShareROne{SCN[sc]}", 100 * v, _pct(v), ms_,
              f"null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == strength, scale == {sc}; "
              "median of d_R1/d_lambda1 (S8_6), %", "sec:nulls", {"scale": sc, "null": "strength"})
    # "r_W tracked l1 more closely (median within-subject rho 0.64 to 0.71) than the toolbox version (0.17 to 0.29)"
    for col_, nm, lo_, hi_ in (("rho_rW_l1", "RW", "0.64", "0.71"), ("rho_rbct_l1", "Bct", "0.17", "0.29")):
        v = med[col_]
        src = (f"null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == strength; per-scale "
               f"median of {col_} (S8_6); {{}} over scales, 2 dp")
        c.add(f"SRayleighRho{nm}LambdaLo", v.min(), _f(v.min()), lo_, src.format("min"), "sec:nulls", {"null": "strength"})
        c.add(f"SRayleighRho{nm}LambdaHi", v.max(), _f(v.max()), hi_, src.format("max"), "sec:nulls", {"null": "strength"})

    # "The hub strength fraction ... removed 62\% to 93\% of the difference in q1"
    mn = c.csv("null_models/mediation_noncircular.csv")
    mn["k"] = mn["k"].astype(str)
    hub = mn[(mn.outcome == "degree_q1") & (mn.k == "avg") & (mn.mediator == "hub_strength_frac") & mn.null.isin(NULLS)]
    src = ("null_models/03_gap_mediators: null_models/mediation_noncircular.csv | outcome == degree_q1, k == avg, "
           "mediator == hub_strength_frac, 3 nulls x 3 scales; {} prop_mediated (S8_8), %")
    hp = hub.prop_mediated
    c.add("SMediationHubFracLo", 100 * hp.min(), _pct(hp.min()), "62", src.format("min"), "sec:nulls", {"c": 0.1})
    c.add("SMediationHubFracHi", 100 * hp.max(), _pct(hp.max()), "93", src.format("max"), "sec:nulls", {"c": 0.1})
    # "Rich-club coefficients removed at most 25\% and usually none"
    rc = mn[mn.mediator.isin(["rc_bu", "rc_wu", "rc_wu_p75", "rc_wu_p85", "rc_wu_p95"])
            & (((mn.outcome == "degree_q1") & (mn.k == "avg")) | ((mn.outcome == "gap") & (mn.k == "-")))
            & mn.null.isin(NULLS)].prop_mediated
    rmax = float(rc.max())
    c.add("SMediationRichClubMax", 100 * rmax, f"{_ceil(100 * rmax, 0):.0f}", "25",
          "null_models/03_gap_mediators: null_models/mediation_noncircular.csv | mediator in rc_bu/rc_wu/rc_wu_p75/"
          f"p85/p95; outcome degree_q1 (k avg) or gap; 3 nulls x 3 scales; max prop_mediated = {100 * rmax:.2f}%, rounded up "
          "(S8_8)", "sec:nulls", {"c": 0.1, "stat": "max"}, bound=(100 * rmax <= 25, "<="))


# ============================================================================ sec:stability
def _stability(c: _Ctx) -> None:
    tb = "topology/tables/"
    # "within-subject rho with degree 0.74 to 0.78, against 0.38 to 0.48 for Ripser cocycles"
    ag = c.csv(tb + "rep_agreement.csv")
    for var, nm, lo_, hi_ in (("geodesic", "Geodesic", "0.74", "0.78"), ("ripser", "Ripser", "0.38", "0.48")):
        v = ag[ag.variant == var].rho_degree_median
        src = (f"topology/summarize_representatives: {tb}rep_agreement.csv | variant == {var}; {{}} over scales of "
               "rho_degree_median (S9_1), 2 dp")
        c.add(f"SRepRho{nm}DegreeLo", v.min(), _f(v.min()), lo_, src.format("min"), "sec:stability", {"convention": var})
        c.add(f"SRepRho{nm}DegreeHi", v.max(), _f(v.max()), hi_, src.format("max"), "sec:stability", {"convention": var})
    # "the overlap with the degree set explained 25\% to 72\% of the variance ... for k >= 2"
    dr = c.csv(tb + "rep_distance_regressions.csv")
    r2 = dr[dr.k >= 2].jaccard_degree_partial_r2
    src = (f"topology/summarize_representatives: {tb}rep_distance_regressions.csv | k >= 2 (2,3,5,15), all scales; {{}} "
           "jaccard_degree_partial_r2 (S9_5 R^2_J), %")
    c.add("SRepOverlapRsqLo", 100 * r2.min(), _pct(r2.min()), "25", src.format("min"), "sec:stability", {"k": ">=2"})
    c.add("SRepOverlapRsqHi", 100 * r2.max(), _pct(r2.max()), "72", src.format("max"), "sec:stability", {"k": ">=2"})
    # "no convention had an effective rank higher by more than 0.12 at any scale or k"
    nc = c.csv(tb + "rep_null_comparisons.csv")
    x = float(nc[(nc.variant != "degree") & nc.k.isin([1, 2, 3, 5])].strength_mean.max())
    c.add("SRepStrengthNullMaxER", x, _f(_ceil(x, 2)), "0.12",
          f"topology/summarize_representatives: {tb}rep_null_comparisons.csv | 4 conventions, k in 1,2,3,5, all scales; "
          f"max strength_mean = {x:.4f} (S9_4 strength-matched), rounded up", "sec:stability",
          {"comparison": "vs strength-rank-matched random sets"}, bound=(x <= 0.12, "<="))
    # "(median rank correlation ... 0.94 to 0.95 at sigma = 0.05) ... (0.56 to 0.63 at p = 0.20)"
    rs = c.csv(tb + "perturb_rank_stability.csv")
    for fam, lvl, nm, lo_, hi_ in (("noise", 0.05, "NoiseFive", "0.94", "0.95"), ("dropout", 0.2, "RemovalTwenty", "0.56", "0.63")):
        v = rs[(rs.ranking == "ripser") & (rs.family == fam) & np.isclose(rs.level, lvl)].rho_median
        src = (f"topology/summarize_perturbation: {tb}perturb_rank_stability.csv | ranking == ripser, family == {fam}, "
               f"level == {lvl}; {{}} over scales of rho_median (S9_6), 2 dp")
        c.add(f"SPerturbRho{nm}Lo", v.min(), _f(v.min()), lo_, src.format("min"), "sec:stability", {"perturbation": f"{fam} {lvl}"})
        c.add(f"SPerturbRho{nm}Hi", v.max(), _f(v.max()), hi_, src.format("max"), "sec:stability", {"perturbation": f"{fam} {lvl}"})
    # "In 51\% to 65\% of the noise perturbations, the largest weight exceeded 1 and the matrix was renormalized"
    rf = c.csv(tb + "perturb_rescaled_fraction.csv")
    v = rf[rf.family == "noise"].rescaled
    src = f"topology/summarize_perturbation: {tb}perturb_rescaled_fraction.csv | family == noise, all scales x levels; {{}} rescaled, %"
    c.add("SPerturbRescaledLo", 100 * v.min(), _pct(v.min()), "51", src.format("min"), "sec:stability", {"perturbation": "noise"})
    c.add("SPerturbRescaledHi", 100 * v.max(), _pct(v.max()), "65", src.format("max"), "sec:stability", {"perturbation": "noise"})


# ============================================================================ sec:targets
def _targets(c: _Ctx) -> None:
    T = "target_states/"
    # "excluding the medial wall (median purity 0.62, 0.68, and 0.80)" / "(2, 4, and 10 parcels)"
    pa = c.csv(T + "parcel_network_assignment.csv")
    pur = pa.groupby("scale").yeo7_purity.median()
    both = pa[pa.legacy_occipital & pa.legacy_association].groupby("scale").size().reindex(list(SC), fill_value=0)
    for s, ms_ in zip(SC, ("0.62", "0.68", "0.80")):
        c.add(f"SYeoPurityMedian{SCN[s]}", pur[s], _f(pur[s]), ms_,
              f"target_states/01_build_targets: {T}parcel_network_assignment.csv | median yeo7_purity per scale", "sec:targets",
              {"scale": s})
    for s, ms_ in zip(SC, ("2", "4", "10")):
        c.add(f"SLegacyPrecuneusParcels{SCN[s]}", both[s], str(int(both[s])), ms_,
              f"target_states/01_build_targets: {T}parcel_network_assignment.csv | count parcels with legacy_occipital & "
              "legacy_association (legacy masks = processed/targets_scale-<N>.npz written by 00_prepare_data)",
              "sec:targets", {"scale": s})

    # occipital (-8.2, -8.5, -1.2), visual (-7.2, -8.9, -1.8), visual drivers excluded (-5.7, -8.9, -1.7)
    nts = c.csv(T + "named_target_summary.csv").set_index(["variant", "target_name", "scale"])
    for variant, tgt, stem, mss in (("full", "kw_occipital_module", "STargetOccipitalFull", ("-8.2", "-8.5", "-1.2")),
                                    ("full", "net_Visual", "STargetVisualFull", ("-7.2", "-8.9", "-1.8")),
                                    ("drivers_excluded", "net_Visual", "STargetVisualExcl", ("-5.7", "-8.9", "-1.7"))):
        for s, ms_ in zip(SC, mss):
            v = nts.loc[(variant, tgt, s)].median_pct
            c.add(f"{stem}{SCN[s]}", v, _f(v, 1), ms_,
                  f"target_states/03_target_stats: {T}named_target_summary.csv | variant == {variant}, target == {tgt}; median_pct "
                  "(S10_1/S10_2), 1 dp", "sec:targets",
                  {"scale": s, "target": tgt, "variant": variant, "c": 0.1, "eps": 1e-5, "stat": "median over subjects"})

    # "Cycle-participation drivers fell inside the visual network in 16\%, 22\%, and 15\% of cases, against 3\%,
    #        1\%, and 1\% for degree-strength drivers"
    ov = c.csv(T + "driver_target_overlap.csv").set_index(["target_name", "scale"])
    for col_, stem, mss in (("frac_any_driver_in_cyc", "SVisualOverlapCycle", ("16", "22", "15")),
                            ("frac_any_driver_in_deg", "SVisualOverlapDegree", ("3", "1", "1"))):
        for s, ms_ in zip(SC, mss):
            v = ov.loc[("net_Visual", s)][col_]
            c.add(f"{stem}{SCN[s]}", 100 * v, str(round(100 * float(f"{v:.2f}"))), ms_,
                  f"target_states/03_target_stats: {T}driver_target_overlap.csv | target == net_Visual; {col_}, 2 dp as S10_3, x100",
                  "sec:targets", {"scale": s, "target": "net_Visual", "stat": "fraction of subject x k cases"})

    # "With two exceptions (the somatomotor network at 68 regions and the default network at 219 regions),
    #        targets in which cycle-participation drivers lay more often favored cycle-participation sets, and vice versa."
    #  Selection inferred: the 7 Yeo networks x 3 scales with Holm-significant full-target energy differences.
    nfull = nts.loc["full"]
    rows = []
    for tgt in YEO7:
        for s in SC:
            d_ov = ov.loc[(tgt, s)].frac_any_driver_in_cyc - ov.loc[(tgt, s)].frac_any_driver_in_deg
            r = nfull.loc[(tgt, s)]
            rows.append((tgt, s, d_ov, r.median_pct, r.p_holm))
    dd = pd.DataFrame(rows, columns=["target", "scale", "d_overlap", "median_pct", "p_holm"])
    dd["agree"] = np.sign(dd.d_overlap) == np.sign(-dd.median_pct)
    sig = dd[dd.p_holm < 0.05]
    exc = sig[~sig.agree]
    exc_all = dd[~dd.agree]
    n_exc = len(exc)
    named = sorted(zip(exc.target, exc.scale)) == sorted([("net_SomMot", 68), ("net_Default", 219)])
    c.add("STargetOverlapExceptions", n_exc, _COUNT_WORDS.get(n_exc, str(n_exc)) if named else f"{n_exc} (other cells)", "two",
          f"target_states/03_target_stats: {T}driver_target_overlap.csv + {T}named_target_summary.csv | 7 Yeo networks x 3 scales, "
          "variant == full; sign(frac_any_driver_in_cyc - frac_any_driver_in_deg) vs sign(-median_pct); counted among cells "
          f"with p_holm < 0.05 ({len(sig)} cells): exceptions {list(zip(exc.target, exc.scale))}. Without the significance "
          f"filter there are {len(exc_all)} disagreeing cells {list(zip(exc_all.target, exc_all.scale))}",
          "sec:targets", {"selection": "Yeo-7 x 3 scales, Holm p < 0.05 (inferred; prose does not state the filter)"})

    # "(for the visual network, rho = -0.77, -0.84, and -0.82)"
    for s, ms_ in zip(SC, ("-0.77", "-0.84", "-0.82")):
        v = nts.loc[("full", "net_Visual", s)].rho_pct_vs_dcoupling
        c.add(f"SVisualCouplingRho{SCN[s]}", v, _f(v), ms_,
              f"target_states/03_target_stats: {T}named_target_summary.csv | variant == full, target == net_Visual; "
              "rho_pct_vs_dcoupling (S10_4)", "sec:targets", {"scale": s, "stat": "Spearman across subjects"})

    # "At equal distance, the visual effect was not significant at 68 regions (-1.1\%) and smaller at 114 and 219
    #        regions (-6.9\% and -2.8\%)"
    pc = c.csv(T + "proximity_controls.csv").set_index(["target_name", "scale"])
    for s, ms_ in zip(SC, ("-1.1", "-6.9", "-2.8")):
        v = pc.loc[("net_Visual", s)].eqdist_intercept
        c.add(f"SVisualEqualDist{SCN[s]}", v, _f(v, 1), ms_,
              f"target_states/07_proximity_controls: {T}proximity_controls.csv | target == net_Visual; eqdist_intercept (median "
              "regression intercept, S10_4)", "sec:targets", {"scale": s, "target": "net_Visual"})

    # "(rho between -0.67 and -0.79; between -0.90 and -0.94 with drivers excluded)"
    sp = c.csv(T + "single_parcel_summary.csv")
    for variant, stem, mw, ms_ in (("full", "SSingleParcelCouplingRho", "-0.67", "-0.79"),
                                   ("drivers_excluded", "SSingleParcelCouplingRhoExcl", "-0.90", "-0.94")):
        rho = sp[sp.variant == variant].rho_parcels_pct_vs_dcoupling
        src = f"target_states/03_target_stats: {T}single_parcel_summary.csv | variant == {variant}; {{}} over scales of rho_parcels_pct_vs_dcoupling"
        c.add(f"{stem}Weak", rho.max(), _f(rho.max()), mw, src.format("max (weakest)"), "sec:targets", {"variant": variant})
        c.add(f"{stem}Strong", rho.min(), _f(rho.min()), ms_, src.format("min (strongest)"), "sec:targets", {"variant": variant})

    # "the median share of target energy carried by directions held at the regularization floor ranged from 86\%
    #        to nearly 100\% across targets"
    sl = c.csv(T + "sensitivity_long.csv.gz")
    sl = sl[(sl.c == 0.1) & (sl.eps == 1e-5)]
    med = sl.groupby(["scale", "target_name"]).floor_frac.median()
    c.add("STargetFloorShareLo", 100 * med.min(), f"{100 * med.min():.0f}", "86",
          f"target_states/04_sensitivity: {T}sensitivity_long.csv.gz | c == 0.1, eps == 1e-5; min over scale x target of median "
          "floor_frac, x100 (S10_8)", "sec:targets", {"c": 0.1, "eps": 1e-5})
    c.add("STargetFloorShareHi", 100 * med.max(), f"{100 * med.max():.0f}", "100",
          f"target_states/04_sensitivity: {T}sensitivity_long.csv.gz | same; max over scale x target, x100 (prose 'nearly 100')",
          "sec:targets", {"c": 0.1, "eps": 1e-5})


# ============================================================================ sec:lesions
def _lesions(c: _Ctx) -> None:
    LE = "lesions/"
    rows = c.csv(LE + "lesion_rows.csv.gz")
    # "random sets of one to five regions have effective rank of about 3.2 at every scale"
    rr = rows[(rows.protocol == "recompute") & (rows.strategy == "degree_strength") & (rows.lesion_type == "degree")]
    er = rr.groupby(["scale", "subject", "k"]).ref_intact_er.mean().groupby(["scale", "k"]).mean().groupby("scale").mean()
    txt = {_f(v, 1) for v in er}
    c.add("SLesionRandomSetER", er.mean(), txt.pop() if len(txt) == 1 else "/".join(sorted(txt)), "3.2",
          f"lesions/01_run_reciprocal_lesion: {LE}lesion_rows.csv.gz | protocol == recompute (random sets on the intact network); "
          f"ref_intact_er, mean over subjects and k = 1,2,3,5 per scale ({', '.join(f'{v:.2f}' for v in er)}), 1 dp",
          "sec:lesions", {"stat": "mean over subjects and k, every scale"})

    # "(difference in loss -0.35; 66 of 70 subjects; Table S11_2)"
    cs = c.csv(LE + "crossover_stats.csv")
    pooled = cs[(cs.protocol == "static") & (cs.scale == "pooled")].set_index(["metric", "lesion_type"])
    r = pooled.loc[("dER", "degree")]
    src = f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | static, pooled, metric dER, degree lesion; "
    c.add("SLesionDegreeDiffER", r["mean"], _f(r["mean"]), "-0.35", src + "mean D (S11_2)", "sec:lesions",
          {"protocol": "static", "lesion": "degree", "comparison": "cycle loss - degree loss"})
    n_neg = int(r.n) - int(r.n_pos)
    c.add("SLesionDegreeDiffERNNeg", n_neg, str(n_neg), "66", src + "n - n_pos (subjects with D < 0)", "sec:lesions",
          {"protocol": "static", "lesion": "degree"})
    # "cycle-participation sets lost 3.6 more percentile points in effective rank and 5.2 more in participation ratio"
    pct = {}
    for m, nm, ms_ in (("dPct_ER", "ER", "3.6"), ("dPct_PR", "PR", "5.2")):
        v = 100 * pooled.loc[(m, "cycle")]["mean"]
        pct[nm] = v
        c.add(f"SLesionCyclePctDiff{nm}", v, f"{v:.1f}", ms_,
              f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | static, pooled, metric {m}, cycle lesion; "
              "100 x mean D, percentile points (S11_2)", "sec:lesions", {"protocol": "static", "lesion": "cycle"})
    # "about one fifth (effective rank) to one third (participation ratio) of the effect of degree lesions"
    for m, nm, ms_ in (("dPct_ER", "ER", "one fifth"), ("dPct_PR", "PR", "one third")):
        deg = 100 * pooled.loc[(m, "degree")]["mean"]
        ratio = abs(pct[nm] / deg)
        word = _FRACTION_WORDS.get(round(1 / ratio), f"{ratio:.2f}")
        c.add(f"SLesionReciprocalFraction{nm}", ratio, word, ms_,
              f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | static, pooled, metric {m}: |mean D (cycle lesion)"
              f" / mean D (degree lesion)| = {pct[nm]:.2f}/{abs(deg):.2f} = {ratio:.3f}; word = nearest 1/n",
              "sec:lesions", {"protocol": "static", "comparison": "cycle-lesion vs degree-lesion effect, percentile units"})

    # "(3.64, against 2.40 for the top-ranked degree set and 3.36 for cycle-participation sets)"
    d = c.csv(LE + "per_subject_degradation.csv")

    def intact(protocol, lesion, strategy):
        p = d[(d.protocol == protocol) & (d.lesion_type == lesion)]
        p = p.pivot_table(index=["scale", "h", "subject"], columns="strategy", values="intact_effective_rank")
        return float(p[strategy].mean())

    src = f"lesions/02_analyze_reciprocal_lesion: {LE}per_subject_degradation.csv | "
    v = intact("static", "degree", "degree_strength")
    c.add("SLesionNextRankedDegreeER", v, f"{v:.2f}", "3.64",
          src + "static, degree lesion; mean intact_effective_rank of the degree-strength (next-ranked surviving) sets, pooled "
          "over scales and h (S11_5 'Intact ER' deg)", "sec:lesions", {"protocol": "static", "lesion": "degree"})
    v = intact("recompute", "degree", "degree_strength")
    c.add("SLesionTopDegreeER", v, f"{v:.2f}", "2.40",
          src + "recompute, degree lesion; mean intact_effective_rank of the top-ranked degree set on the intact network",
          "sec:lesions", {"protocol": "recompute", "lesion": "degree"})
    v = intact("recompute", "degree", "cycle_participation")
    alt = intact("static", "degree", "cycle_participation")
    c.add("SLesionTopCycleER", v, f"{v:.2f}", "3.36",
          src + "recompute, degree lesion; mean intact_effective_rank of the top-ranked cycle-participation set on the intact "
          f"network (parallel to the 2.40 selection). The static degree-lesion cycle row gives {alt:.2f} (cycle sets that "
          "avoid the removed degree hubs)", "sec:lesions",
          {"protocol": "recompute", "lesion": "degree", "alt_static_degree_lesion": round(alt, 4)})

    # "(r = 0.989 with log10[(N - h)/N])"
    st = rows[rows.protocol == "static"]
    rsz = float(np.corrcoef(st.lesioned_log10_average_energy - st.intact_log10_average_energy,
                            np.log10(st.n_lesioned / st.n_intact))[0, 1])
    c.add("SLesionEnergySizeCorr", rsz, f"{rsz:.3f}", "0.989",
          f"lesions/01_run_reciprocal_lesion: {LE}lesion_rows.csv.gz | protocol == static, all rows; Pearson r of lesioned - "
          "intact log10 E_avg with log10(n_lesioned/n_intact) (as 02_analyze 'energy_corr')", "sec:lesions",
          {"protocol": "static"})
    # "the two strategies differed by at most 0.005 log units"
    cells = cs[(cs.metric == "dlog10E") & (cs.protocol == "static") & (cs.scale != "pooled")
               & cs.lesion_type.isin(["degree", "cycle", "random"])]
    v = float(cells["mean"].abs().max())
    c.add("SLesionEnergyStrategyDiffMax", v, f"{_ceil(v, 3):.3f}", "0.005",
          f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | metric dlog10E, static, scale x h cells, 3 lesion "
          f"types; max |mean D| = {v:.4f}, rounded up", "sec:lesions", {"protocol": "static"}, bound=(v <= 0.005, "<="))
