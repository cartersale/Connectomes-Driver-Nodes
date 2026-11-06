"""Numbers quoted in the prose of the Supporting Information (supp_mat.tex), section by section.

Table bodies are built by scripts/paper/tables.py and scripts/paper/condense.py. See the package docstring for
the formatting conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from hoc.analysis.null_models import load_stack

from ._common import NULLS, SC, SCN, Collector, ceil_to, f2, pct, sig1_ceil
from .record import Number


def collect(run: Path) -> list[Number]:
    c = Collector(run)
    for fn in (_robustness, _criteria, _modes, _nulls, _core):
        fn(c)
    return c.done()


# ============================================================================ S2 sec:robustness
def _robustness(c: Collector) -> None:
    L = "sec:robustness"
    # "At every scale and eps, 80% to 98% of the eigenvalues of W lay below eps"
    a = c.csv("sensitivity/energy_floor_driver_sets.csv")
    a = a[np.isclose(a.eps, 1e-8) | np.isclose(a.eps, 1e-5) | np.isclose(a.eps, 1e-2)]
    sub = a.groupby(["scale", "eps", "method", "subject_id"]).fraction_below_eps.mean()
    agg = sub.groupby(["scale", "eps", "method"]).mean().map(lambda x: float(f"{x:.2f}"))
    src = ("sensitivity/04_energy_floor: sensitivity/energy_floor_driver_sets.csv | eps in 1e-8, 1e-5, 1e-2; 5 set types; "
           "per-subject k-mean, subject mean, 2 dp as in Table S1; {} over scale x eps x set type, x100")
    c.add("SEpsBelowFloorMin", 100 * agg.min(), pct(agg.min()), "80", src.format("min"), L, {"c": 0.1})
    c.add("SEpsBelowFloorMax", 100 * agg.max(), pct(agg.max()), "98", src.format("max"), L, {"c": 0.1})

    # "mean log10 E_avg lay within 0.1 of log10(N/eps), changing by one unit per tenfold change in eps"
    tidy = c.csv("sensitivity/horizon_eps_largek_tidy.csv.gz")
    t = tidy[(tidy.c == 0.1) & np.isinf(tidy["T"]) & tidy.k.isin([1, 2, 3, 5])
             & tidy.method.isin(["degree_strength", "cycle_participation"])]
    agg = (t.groupby(["scale", "eps", "method", "subject_id"]).log10_average_energy.mean()
           .groupby(["scale", "eps", "method"]).mean().reset_index())
    gapmax = float((np.log10(agg.scale / agg.eps) - agg.log10_average_energy).abs().max())
    c.add("SEpsEnergyWithinFloorAbout", gapmax, f"{gapmax:.1f}", "0.1",
          "sensitivity/03_horizon_eps_largek: sensitivity/horizon_eps_largek_tidy.csv.gz | c == 0.1, T == inf, k in 1,2,3,5, "
          "degree/cycle; max over scale x eps x method of |log10(N/eps) - mean log10 E_avg| ('within about')", L,
          {"c": 0.1, "stat": "max over scale x eps x method"})
    slopes = np.array([-np.polyfit(np.log10(g.eps), g.log10_average_energy, 1)[0] for _, g in agg.groupby(["scale", "method"])])
    c.add("SEpsEnergyTenfoldSlope", slopes.mean(), "one" if np.all(np.round(slopes) == 1) else f2(slopes.mean()), "one",
          "sensitivity/03_horizon_eps_largek: sensitivity/horizon_eps_largek_tidy.csv.gz | same cohort means; least-squares "
          f"slope of mean log10 E_avg on -log10(eps) per scale x method (range {slopes.min():.3f} to {slopes.max():.3f})", L,
          {"c": 0.1})

    # "The cycle-minus-degree difference in log10 E_avg was at most 2.4e-3"
    e = c.csv("sensitivity/epsilon_summary.csv")
    e = e[(e.c == 0.1) & (e.metric == "log10_average_energy")]
    mx = float(e.mean_diff.abs().max())
    c.add("SEpsEnergyCycleDegreeMaxAbs", mx, f"{ceil_to(mx * 1e3, 1):.1f}\\times10^{{-3}}", "2.4\\times10^{-3}",
          "sensitivity/03_horizon_eps_largek: sensitivity/epsilon_summary.csv | c == 0.1, metric == log10_average_energy; "
          "max |mean_diff| (cycle - degree, k-averaged) over 3 scales x 7 eps, rounded up", L,
          {"c": 0.1}, bound=(mx <= 2.4e-3, "<="))

    # Damping: "changed sign in 8, 4, and 20 subjects"; "changed sign at c ~ 0.88, 0.69, and 0.29"
    fl = c.csv("concentration/decay_flip_summary.csv")
    r = fl[(fl.curve == "cd_q1") & (fl.axis == "flip_c")].iloc[0]
    for s, ms_ in zip(SC, ("8", "4", "20")):
        v = int(r[f"n_flip_{s}"])
        c.add(f"SCsweepQOneSignChangeSubjects{SCN[s]}", v, str(v), ms_,
              "concentration/03_decay_collapse: concentration/decay_flip_summary.csv | curve == cd_q1, axis == flip_c; "
              f"n_flip_{s}", L, {"scale": s, "grid": "25 c values"})
    sf = c.csv("sensitivity/csweep_signflip.csv")
    for s, ms_ in zip(SC, ("0.88", "0.69", "0.29")):
        v = float(sf[(sf.metric == "effective_rank") & (sf.scale == s)].c_flip_mean.iloc[0])
        c.add(f"SCsweepERFlipC{SCN[s]}", v, f"{v:.2f}", ms_,
              "sensitivity/05_summarize_csweep: sensitivity/csweep_signflip.csv | metric == effective_rank; c_flip_mean", L,
              {"scale": s, "grid": "13 c values"})

    # nctpy: "In 72 checked combinations ... to within 4e-15 ... agreed to within 3e-14"
    det = c.csv("sensitivity/nctpy_equivalence_detail.csv")
    src = "sensitivity/02_nctpy_equivalence: sensitivity/nctpy_equivalence_detail.csv | "
    c.add("SNctpyChecks", len(det), str(len(det)), "72", src + "number of rows (subject x criterion x k checks)", L)
    v = float(det.max_abs_W_diff.max())
    c.add("SNctpyMaxGramianDiff", v, sig1_ceil(v), "5\\times10^{-15}",
          src + f"max of max_abs_W_diff = {v:.3g}; 1 significant figure rounded up", L, bound=(v <= 5e-15, "<="))
    v = float(det.d_er_eps_rescaled.abs().max())
    c.add("SNctpyMaxRescaledDiff", v, sig1_ceil(v), "3\\times10^{-14}",
          src + f"max |d_er_eps_rescaled| = {v:.3g}; 1 significant figure rounded up", L, bound=(v <= 3e-14, "<="))


# ============================================================================ S3 sec:criteria
def _criteria(c: Collector) -> None:
    L = "sec:criteria"
    spin = c.csv("target_states/spin_tests.csv")
    cd = spin[(spin.x == "degree_strength") & (spin.y == "cycle_participation")].set_index("scale")
    for s, ms_ in zip(SC, ("0.75", "0.71", "0.61")):
        v = cd.loc[s, "spearman_rho"]
        c.add(f"SCycleDegreeRho{SCN[s]}", v, f2(v), ms_,
              "target_states/05_spin_tests: target_states/spin_tests.csv | x == degree_strength, y == cycle_participation; "
              "spearman_rho", L, {"scale": s, "stat": "Spearman across parcels"})
    p = float(cd.p_spin_vasa.max())
    c.add("SCycleDegreePspinMax", p, f"{ceil_to(p, 4):.4f}", "0.0002",
          "target_states/05_spin_tests: target_states/spin_tests.csv | same rows; max over scales of p_spin_vasa, rounded up",
          L, {"test": "Vasa spin, 10,000 rotations"}, bound=(p <= 0.0002, "<="))
    top = c.csv("target_states/top15_network_composition.csv")
    for s, ms_ in zip(SC, ("11", "4", "5")):
        n = top[(top.scale == s) & (top.category == "both")].n.iloc[0]
        c.add(f"STopFifteenShared{SCN[s]}", n, str(int(n)), ms_,
              "target_states/06_network_enrichment: target_states/top15_network_composition.csv | category == both; n", L,
              {"scale": s, "k": 15})

    tb = "topology/tables/"
    ag = c.csv(tb + "rep_agreement.csv")
    src = f"topology/summarize_representatives: {tb}rep_agreement.csv | {{sel}}; {{stat}} over scales of {{col}}, 2 dp"
    other = ag[ag.variant != "ripser"].rho_ripser_median
    c.add("SRepRhoRipserMin", other.min(), f2(other.min()), "0.40",
          src.format(sel="variant != ripser", stat="min", col="rho_ripser_median"), L)
    c.add("SRepRhoRipserMax", other.max(), f2(other.max()), "0.65",
          src.format(sel="variant != ripser", stat="max", col="rho_ripser_median"), L)
    for sel, nm, lo_, hi_ in ((["geodesic"], "Geodesic", "0.74", "0.78"),
                              (["ripser", "gudhi", "birth_edge"], "Others", "0.37", "0.60")):
        v = ag[ag.variant.isin(sel)].rho_degree_median
        c.add(f"SRepRho{nm}DegreeMin", v.min(), f2(v.min()), lo_,
              src.format(sel=f"variant in {sel}", stat="min", col="rho_degree_median"), L)
        c.add(f"SRepRho{nm}DegreeMax", v.max(), f2(v.max()), hi_,
              src.format(sel=f"variant in {sel}", stat="max", col="rho_degree_median"), L)
    ps = pd.concat([c.csv(f"topology/representatives/per_subject_scale-{s}.csv") for s in SC])
    kav = (ps[ps.k.isin([1, 2, 3, 5])].groupby(["scale", "variant", "subject_id"]).er_diff_degree.mean()
           .groupby(["scale", "variant"]).mean())
    srcd = ("topology/run_representatives_all_scales: topology/representatives/per_subject_scale-<N>.csv | per-subject mean "
            "of er_diff_degree over k in 1,2,3,5, mean over subjects; {} over 4 conventions x 3 scales")
    c.add("SRepERDiffMin", kav.min(), f"{kav.min():+.2f}", "+0.23", srcd.format("min"), L)
    c.add("SRepERDiffMax", kav.max(), f"{kav.max():+.2f}", "+1.05", srcd.format("max"), L)
    dr = c.csv(tb + "rep_distance_regressions.csv")
    # "the overlap with the degree-strength set explained 25% to 71% of the variance in this difference for k in {2, 3, 5}"
    r2 = dr[dr.k.isin([2, 3, 5])].jaccard_degree_partial_r2
    src = (f"topology/summarize_representatives: {tb}rep_distance_regressions.csv | k in 2,3,5, all scales; {{}} "
           "jaccard_degree_partial_r2, %")
    c.add("SRepOverlapRsqMin", 100 * r2.min(), pct(r2.min()), "25", src.format("min"), L, {"k": "2,3,5"})
    c.add("SRepOverlapRsqMax", 100 * r2.max(), pct(r2.max()), "71", src.format("max"), L, {"k": "2,3,5"})


# ============================================================================ S4 sec:modes
def _projector_eigenvalues(stack: np.ndarray, cols: list[int]) -> np.ndarray:
    """Eigenvalues (descending) of the mean projector onto each subject's span of the given eigenvectors
    (1 = v1, 2 = v2, ...), as in scripts/analysis/mechanisms/05_second_subspace.py."""
    P = 0
    with np.errstate(all="ignore"):  # spurious BLAS warnings on macOS Accelerate; the result is checked below
        for A in stack:
            _, U = np.linalg.eigh(A)
            V = U[:, [-i for i in cols]]
            P = P + V @ V.T
    if not np.isfinite(P).all():
        raise FloatingPointError("non-finite mean projector")
    return np.linalg.eigvalsh(P / len(stack))[::-1]


def _modes(c: Collector) -> None:
    L = "sec:modes"
    src = ("processed/connectomes_scale-<N>.npz (as mechanisms/05_second_subspace): eigenvalues of the mean projector onto "
           "each subject's {}")
    printed = {68: ("0.72", "0.40", "0.48", None), 114: ("0.67", "0.37", "0.44", "0.38"), 219: ("0.60", "0.34", "0.40", "0.35")}
    for s in SC:
        stack = load_stack(s, c.run)
        w23 = _projector_eigenvalues(stack, [2, 3])
        w2 = _projector_eigenvalues(stack, [2])
        pa, pb, p2, p3 = printed[s]
        c.add(f"SModesPlaneAxisA{SCN[s]}", w23[0], f2(w23[0]), pa, src.format("v2-v3 plane; first"), L, {"scale": s})
        c.add(f"SModesPlaneAxisB{SCN[s]}", w23[1], f2(w23[1]), pb, src.format("v2-v3 plane; second"), L, {"scale": s})
        c.add(f"SModesVTwoAlone{SCN[s]}", w2[0], f2(w2[0]), p2, src.format("v2; largest"), L, {"scale": s})
        if p3 is not None:
            w234 = _projector_eigenvalues(stack, [2, 3, 4])
            c.add(f"SModesThirdAxis{SCN[s]}", w234[2], f2(w234[2]), p3, src.format("v2-v4 span; third"), L, {"scale": s})

    # hemispheric division of the core: |corr(axis, hemisphere)| across the 20% of regions with the highest mean v1
    ax = c.csv("mechanisms/second_subspace_axes.csv")
    srch = ("mechanisms/05_second_subspace: mechanisms/second_subspace_axes.csv | core = top 20% of regions by v1_mean; "
            "|Pearson r| across core regions between axis {} and hemisphere (R = 1)")
    for s, col, nm, ms_ in ((68, "b", "SecondAxis", "0.79"), (114, "b", "SecondAxis", "0.26"), (219, "b", "SecondAxis", "0.13"),
                            (114, "c", "ThirdAxis", "0.69"), (219, "c", "ThirdAxis", "0.61")):
        g = ax[ax.scale == s]
        core = g.nlargest(int(round(0.2 * len(g))), "v1_mean")
        r = abs(float(np.corrcoef(core[col], (core.hemi == "R").astype(float))[0, 1]))
        c.add(f"SModesHemi{nm}{SCN[s]}", r, f2(r), ms_, srch.format(col), L, {"scale": s})

    # model networks: "For k >= 2, the excess was smaller than in the connectomes (0.11 to 0.28, against 0.31 to 0.37),
    # as was the spectral gap (0.14 to 0.32, against 0.43 to 0.54)."
    cs = c.csv("mechanisms/canonical_summary.csv")
    w = cs[(cs.weighting == "weighted") & np.isclose(cs.c, 0.1) & (cs.kgrp == "2+")]
    srce = ("mechanisms/02_canonical_graphs: mechanisms/canonical_summary.csv | weighting == weighted, c == 0.1, kgrp == 2+ "
            "(mean over k in 2,3,5 and graphs); {} deg_minus_rand_q1")
    for sel, nm, lo_, hi_, over in ((w.model != "CONN", "Model", "0.11", "0.28", "models ER, BA, WS, SBM, RGG x scales"),
                                    (w.model == "CONN", "Conn", "0.31", "0.37", "the connectomes (model CONN) x scales")):
        x = w[sel].deg_minus_rand_q1
        ctx = {"c": 0.1, "k": "2,3,5", "comparison": "degree-strength minus random mean"}
        c.add(f"S{nm}ExcessQOneMin", x.min(), f2(x.min()), lo_, srce.format(f"min over {over} of"), L, ctx)
        c.add(f"S{nm}ExcessQOneMax", x.max(), f2(x.max()), hi_, srce.format(f"max over {over} of"), L, ctx)
    srcm = "mechanisms/02_canonical_graphs: mechanisms/canonical_summary.csv | weighting == weighted, c == 0.1, kgrp == 2+; {} gap"
    for sel, nm, lo_, hi_ in ((w.model != "CONN", "Model", "0.14", "0.32"), (w.model == "CONN", "Conn", "0.43", "0.54")):
        g = w[sel].gap
        c.add(f"S{nm}GapMin", g.min(), f2(g.min()), lo_, srcm.format("min over models x scales" if nm == "Model" else "min over scales"), L)
        c.add(f"S{nm}GapMax", g.max(), f2(g.max()), hi_, srcm.format("max over models x scales" if nm == "Model" else "max over scales"), L)


# ============================================================================ S5 sec:nulls
def _nulls(c: Collector) -> None:
    L = "sec:nulls"
    fid = c.csv("null_models/null_fidelity.csv")
    st = fid[fid.null == "strength"]
    src = "null_models/02_analyze: null_models/null_fidelity.csv | null == strength; {} over scales"
    c.add("SNullStrengthCorrMin", st.strength_r.min(), f2(st.strength_r.min(), 3), "0.998", src.format("min strength_r"), L)
    c.add("SNullStrengthCorrMax", st.strength_r.max(), f2(st.strength_r.max(), 3), "0.999", src.format("max strength_r"), L)
    c.add("SNullStrengthMaxRelErrMin", st.strength_max_rel_err.min(), f2(st.strength_max_rel_err.min()), "0.16",
          src.format("min strength_max_rel_err"), L)
    c.add("SNullStrengthMaxRelErrMax", st.strength_max_rel_err.max(), f2(st.strength_max_rel_err.max()), "0.27",
          src.format("max strength_max_rel_err"), L)
    real = fid[fid.null == "real"].set_index("scale").mean_edge_length
    geo = fid[fid.null == "geometric"].set_index("scale").mean_edge_length
    dev = float((100 * (geo - real).abs() / real).max())
    c.add("SNullGeomEdgeLengthWithinPct", dev, f"{ceil_to(dev, 0):.0f}", "1",
          f"null_models/02_analyze: null_models/null_fidelity.csv | max over scales of |geometric - real| / real mean_edge_length, % "
          f"({dev:.2f}%)", L, bound=(dev < 1, "<"))

    gs = c.csv("null_models/gap_split.csv")
    z = gs[(gs.stat == "lambda2") & (gs.scale == 114) & gs.null.isin(NULLS)].z_median
    src = "null_models/03_gap_mediators: null_models/gap_split.csv | stat == lambda2, scale == 114, 3 nulls; {} z_median"
    c.add("SNullLambdaTwoZOneFourteenMin", z.min(), f"{z.min():.1f}", "0.6", src.format("min"), L)
    c.add("SNullLambdaTwoZOneFourteenMax", z.max(), f"{z.max():.1f}", "0.9", src.format("max"), L)

    ps = c.csv("null_models/rayleigh/per_subject.csv")
    ps = ps[ps.null == "strength"].assign(sh1=lambda x: x.d_R1 / x.d_lambda1, sh3=lambda x: x.d_R3 / x.d_lambda1)
    med = ps.groupby("scale").median(numeric_only=True)
    src = "null_models/04_rayleigh_bound: null_models/rayleigh/per_subject.csv | null == strength; "
    for s, ms_ in zip(SC, ("60", "43", "36")):
        v = med.loc[s, "sh1"]
        c.add(f"SRayleighShareROne{SCN[s]}", 100 * v, pct(v), ms_, src + f"scale == {s}; median of d_R1/d_lambda1, %", L,
              {"scale": s})
    sh3 = med.sh3
    c.add("SRayleighShareRThreeMin", 100 * sh3.min(), pct(sh3.min()), "95", src + "per-scale median of d_R3/d_lambda1; min, %", L)
    c.add("SRayleighShareRThreeMax", 100 * sh3.max(), pct(sh3.max()), "112", src + "per-scale median of d_R3/d_lambda1; max, %", L)
    for col, nm, lo_, hi_ in (("rho_rW_l1", "RW", "0.64", "0.71"), ("rho_rbct_l1", "Bct", "0.17", "0.29")):
        v = med[col]
        c.add(f"SRayleighRho{nm}LambdaMin", v.min(), f2(v.min()), lo_, src + f"per-scale median of {col}; min", L)
        c.add(f"SRayleighRho{nm}LambdaMax", v.max(), f2(v.max()), hi_, src + f"per-scale median of {col}; max", L)


# ============================================================================ S6 sec:core
def _core(c: Collector) -> None:
    L = "sec:core"
    fo = c.csv("mechanisms/core_cut_first_order.csv")
    src = ("mechanisms/07_core_cut: mechanisms/core_cut_first_order.csv | partition == {p}, condition == {cond}, {sel}; "
           "-median_dgap_pct (gap reduction, % of the unperturbed gap)")

    def red(scale, part, cond, alpha):
        r = fo[(fo.scale == scale) & (fo.partition == part) & (fo.condition == cond) & np.isclose(fo.alpha, alpha)]
        assert len(r) == 1, (scale, part, cond, alpha)
        return -float(r.median_dgap_pct.iloc[0])

    alphas = (0.25, 0.5, 0.75, 1.0)
    words = {0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "Full"}
    for s, printed in ((68, ("11", "21", "29", "35")), (219, ("17", "32", "42", "47"))):
        for a, ms_ in zip(alphas, printed):
            v = red(s, "labels", "cross", a)
            c.add(f"SCoreFrontBack{SCN[s]}{words[a]}", v, f"{v:.0f}", ms_,
                  src.format(p="labels", cond="cross", sel=f"scale == {s}, alpha == {a}"), L, {"scale": s, "alpha": a})
    for s, cond, nm, lo_, hi_ in ((68, "within_matched", "WithinMatched", "3", "13"), (219, "within_matched", "WithinMatched", "7", "26"),
                                  (114, "cross", "FrontBack", "5", "12"), (114, "within_matched", "WithinMatched", "7", "25")):
        v = [red(s, "labels", cond, a) for a in alphas]
        sel = f"scale == {s}, alpha in {alphas}"
        c.add(f"SCore{nm}{SCN[s]}Min", min(v), f"{min(v):.0f}", lo_, src.format(p="labels", cond=cond, sel=sel) + "; min", L)
        c.add(f"SCore{nm}{SCN[s]}Max", max(v), f"{max(v):.0f}", hi_, src.format(p="labels", cond=cond, sel=sel) + "; max", L)
    for cond, nm, printed in (("within_uniform", "WithinUniform", ("7", "22", "16")), ("within_matched", "WithinMatchedHalf", ("6", "13", "14"))):
        for s, ms_ in zip(SC, printed):
            v = red(s, "labels", cond, 0.5)
            c.add(f"SCore{nm}{SCN[s]}", v, f"{v:.0f}", ms_, src.format(p="labels", cond=cond, sel=f"scale == {s}, alpha == 0.5"), L)
    ou = max(abs(red(s, "labels", "other_uniform", 0.5)) for s in SC)
    c.add("SCoreOutsideUniformMax", ou, f"{ceil_to(ou, 0):.0f}", "2",
          src.format(p="labels", cond="other_uniform", sel="alpha == 0.5") + f"; max |change| over scales ({ou:.2f}), rounded up",
          L, bound=(ou < 2, "<"))

    # median-coordinate split
    for s, cond, nm, ms_ in ((68, "cross", "FrontBack", "26"), (68, "within_matched", "WithinMatched", "10"),
                             (114, "cross", "FrontBack", "12"), (114, "within_matched", "WithinMatched", "17")):
        v = red(s, "ycut", cond, 0.5)
        c.add(f"SCoreMedianSplit{nm}{SCN[s]}", v, f"{v:.0f}", ms_, src.format(p="ycut", cond=cond, sel=f"scale == {s}, alpha == 0.5"), L)
    comp = c.csv("mechanisms/core_cut_compare.csv")
    r = comp[(comp.scale == 68) & (comp.partition == "ycut") & np.isclose(comp.alpha, 0.5) & (comp.control == "within_matched")
             & (comp.measure == "dgap")].iloc[0]
    n = r.frac_cross_lower * r.n
    c.add("SCoreMedianSplitBeatsWithinSixtyEight", n, f"{n:.0f}", "57",
          "mechanisms/07_core_cut: mechanisms/core_cut_compare.csv | scale == 68, partition == ycut, alpha == 0.5, control == "
          "within_matched, measure == dgap; frac_cross_lower * n", L)
    base = c.csv("mechanisms/core_cut_baseline.csv")
    vc = base.groupby(["partition", "scale"]).v2_split_corr.median()
    for part, s, nm, ms_ in (("ycut", 68, "MedianSplit", "0.67"), ("labels", 68, "Labels", "0.74"),
                             ("ycut", 114, "MedianSplit", "0.32"), ("labels", 114, "Labels", "0.30")):
        v = vc.loc[(part, s)]
        c.add(f"SCoreVTwoCorr{nm}{SCN[s]}", v, f2(v), ms_,
              f"mechanisms/07_core_cut: mechanisms/core_cut_baseline.csv | partition == {part}, scale == {s}; median v2_split_corr",
              L)

    # first-order prediction (the bound and the correlations are also quoted in the main text, subsec:nulls_results)
    LM = (L, "subsec:nulls_results")
    sel = fo[fo.partition.isin(["labels", "v2sign"]) & (fo.alpha <= 0.5)]
    err = float((sel.median_pred_dgap_pct - sel.median_dgap_pct).abs().max())
    c.add("SCoreFirstOrderWithinPP", err, f"{ceil_to(err, 0):.0f}", "5",
          "mechanisms/07_core_cut: mechanisms/core_cut_first_order.csv | partition in labels, v2sign; alpha <= 0.5; all "
          f"conditions and scales; max |median_pred_dgap_pct - median_dgap_pct| = {err:.2f}, rounded up", LM,
          bound=(err <= 5, "<="))
    rho = fo[np.isclose(fo.alpha, 0.5) & fo.condition.isin(["cross", "within_matched", "other_matched"])].rho
    srcr = ("mechanisms/07_core_cut: mechanisms/core_cut_first_order.csv | alpha == 0.5, condition in cross, within_matched, "
            "other_matched (all partitions); {} rho")
    c.add("SCoreFirstOrderRhoMin", rho.min(), f2(rho.min()), "0.84", srcr.format("min"), LM)
    c.add("SCoreFirstOrderRhoMax", rho.max(), f2(rho.max()), "0.99", srcr.format("max"), LM)
    d = c.csv("mechanisms/core_cut.csv.gz", usecols=["subject_id", "scale", "partition", "alpha", "condition", "dgap", "pred_dgap"])
    d = d[(d.partition == "labels") & np.isclose(d.alpha, 0.5) & d.condition.isin(["cross", "within_matched"])]
    p = d.groupby(["scale", "subject_id", "condition"])[["dgap", "pred_dgap"]].mean().unstack("condition")
    agree = (np.sign(p[("dgap", "cross")] - p[("dgap", "within_matched")])
             == np.sign(p[("pred_dgap", "cross")] - p[("pred_dgap", "within_matched")])).groupby("scale").sum()
    for s, ms_ in zip(SC, ("67", "62", "64")):
        c.add(f"SCoreFirstOrderPicksWinner{SCN[s]}", agree[s], str(int(agree[s])), ms_,
              "mechanisms/07_core_cut: mechanisms/core_cut.csv.gz | partition == labels, alpha == 0.5; per subject mean over "
              "draws of dgap and pred_dgap for cross and within_matched; subjects in which the predicted and observed signs of "
              "(cross - within_matched) agree", L, {"scale": s})
    r = fo[(fo.scale == 219) & (fo.partition == "labels") & (fo.condition == "cross") & np.isclose(fo.alpha, 1.0)].iloc[0]
    gap = float(r.median_dgap_pct - r.median_pred_dgap_pct)
    c.add("SCoreFirstOrderOverestimate", gap, f"{ceil_to(gap, 0):.0f}", "26",
          "mechanisms/07_core_cut: mechanisms/core_cut_first_order.csv | scale == 219, partition == labels, condition == cross, "
          f"alpha == 1; median_dgap_pct - median_pred_dgap_pct = {gap:.2f} percentage points, rounded up", L)
    for s, ms_ in zip(SC, ("10", "16", "14")):
        v = float(fo[(fo.scale == s) & (fo.partition == "labels") & (fo.condition == "cross") & np.isclose(fo.alpha, 1.0)]
                  .v2_swap.iloc[0])
        c.add(f"SCoreVTwoRotated{SCN[s]}", 100 * v, pct(v), ms_,
              "mechanisms/07_core_cut: mechanisms/core_cut_first_order.csv | partition == labels, condition == cross, alpha == 1; "
              "v2_swap (share of subjects with |cos(v2, v2 perturbed)| < 0.5), %", L, {"scale": s})

    # share of front-back weight on opposite sides of v2 (v2 signed to agree with the anatomical split)
    reg = c.csv("mechanisms/core_cut_regions.csv")
    for s, ms_ in zip(SC, ("56", "18", "23")):
        lab = reg[reg.scale == s].sort_values("node_index")["labels"].to_numpy()
        g = np.where(lab == "anterior", 1.0, np.where(lab == "posterior", -1.0, 0.0))
        shares = []
        for A in load_stack(s, c.run):
            n = A.shape[0]
            v2 = linalg.eigh(A, subset_by_index=[n - 3, n - 1])[1][:, ::-1][:, 1]
            v2 = v2 * (1.0 if v2 @ g >= 0 else -1.0)
            iu, ju = np.triu_indices(n, 1)
            w = A[iu, ju]
            cross = (g[iu] * g[ju] < 0) & (w > 0)
            shares.append(w[cross & (np.sign(v2[iu]) != np.sign(v2[ju]))].sum() / w[cross].sum())
        v = float(np.median(shares))
        c.add(f"SCoreFrontBackOppositeVTwo{SCN[s]}", 100 * v, pct(v), ms_,
              "processed/connectomes_scale-<N>.npz + mechanisms/core_cut_regions.csv (labels): per subject, share of front-back "
              "edge weight joining core regions on opposite sides of v2 (v2 signed as in 07_core_cut.oriented_v2); median, %", L,
              {"scale": s})
    r = fo[(fo.scale == 114) & (fo.partition == "labels") & (fo.condition == "cross") & np.isclose(fo.alpha, 0.5)].iloc[0]
    ratios = (r.median_dl2 / r.median_dl1, r.median_pred_dl2 / r.median_pred_dl1)
    c.add("SCoreHalfRatioOneFourteen", float(np.mean(ratios)), "half" if all(0.4 <= x <= 0.6 for x in ratios) else f2(np.mean(ratios)),
          "half", "mechanisms/07_core_cut: mechanisms/core_cut_first_order.csv | scale == 114, labels, cross, alpha == 0.5; "
          f"median_dl2 / median_dl1 observed {ratios[0]:.2f}, predicted {ratios[1]:.2f}; the word 'half'", L)
