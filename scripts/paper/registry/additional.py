"""Numbers quoted in supplementary/additional_analyses.md (analyses published with the code, not in the paper).

See the package docstring for the formatting conventions. Names start with ``A``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._common import COUNT_WORDS, NULLS, SC, SCN, Collector, ceil_to, f2, pct
from .record import Number

L = "additional"


def collect(run: Path) -> list[Number]:
    c = Collector(run)
    for fn in (_decay, _gradients, _setlevel, _perturbation, _mediation, _targets, _lesions):
        fn(c)
    return c.done()


def _decay(c: Collector) -> None:
    col = c.csv("concentration/decay_collapse.csv")
    src = "concentration/03_decay_collapse: concentration/decay_collapse.csv | column D (matched windows); "
    red = []
    for curve, nm, mc, md in (("cd_q1", "CycleDegreeQOne", "0.022", "0.015"), ("cd_er", "CycleDegreeER", "0.037", "0.026"),
                              ("dr_q1", "DegreeRandomQOne", "0.030", "0.021")):
        dc = float(col[(col.curve == curve) & (col.axis == "c (matched window)")].D.iloc[0])
        dd = float(col[(col.curve == curve) & (col.axis == "decay_ratio (matched window)")].D.iloc[0])
        red.append(1 - dd / dc)
        c.add(f"ACollapseD{nm}C", dc, f"{dc:.3f}", mc, src + f"curve == {curve}, axis == 'c (matched window)'", L)
        c.add(f"ACollapseD{nm}Decay", dd, f"{dd:.3f}", md, src + f"curve == {curve}, axis == 'decay_ratio (matched window)'", L)
    v = 100 * float(np.mean(red))
    c.add("ACollapseDispersionDropPct", v, f"{round(v, -1):.0f}", "30",
          src + "mean over the 3 curves of 1 - D(d)/D(c), x100, rounded to 10 ('about 30%')", L)
    fl = c.csv("concentration/decay_flip_summary.csv")
    srcf = "concentration/03_decay_collapse: concentration/decay_flip_summary.csv | curve == cd_er; "
    for axis, nm, ms_ in (("flip_decay", "Decay", "1.4"), ("flip_c", "C", "2.0")):
        v = float(fl[(fl.curve == "cd_er") & (fl.axis == axis)].max_over_min.iloc[0])
        c.add(f"ACollapseERFlipSpread{nm}", v, f"{v:.1f}", ms_, srcf + f"axis == {axis}; max_over_min", L)
    kp = float(fl[(fl.curve == "cd_er") & (fl.axis == "flip_decay")].kruskal_p.iloc[0])
    c.add("ACollapseERFlipKruskalBound", kp, f"10^{{{math.ceil(math.log10(kp))}}}" if kp >= 1e-3 else "10^{-3}", "10^{-3}",
          srcf + f"axis == flip_decay; kruskal_p ({kp:.2g}), reported as p < 1e-3", L, bound=(kp < 1e-3, "<"))
    dcv = c.csv("concentration/decay_curves.csv.gz", usecols=["subject_id", "scale", "gap"])
    gaps = dcv.drop_duplicates(["scale", "subject_id"]).groupby("scale").gap.mean()
    for s, ms_ in zip(SC, ("0.43", "0.54", "0.44")):
        c.add(f"ACollapseMeanGap{SCN[s]}", gaps[s], f2(gaps[s]), ms_,
              "concentration/03_decay_collapse: concentration/decay_curves.csv.gz | gap per subject, mean over 70 subjects", L)


def _gradients(c: Collector) -> None:
    spin = c.csv("target_states/spin_tests.csv")
    src = "target_states/05_spin_tests: target_states/spin_tests.csv | x == rank_divergence, y == G1; "
    for s, mr, mp in zip(SC, ("-0.28", "-0.26", "-0.16"), ("0.16", "0.12", "0.23")):
        r = spin[(spin.scale == s) & (spin.x == "rank_divergence") & (spin.y == "G1")].iloc[0]
        c.add(f"ARankDivGOneRho{SCN[s]}", r.spearman_rho, f2(r.spearman_rho), mr, src + "spearman_rho", L, {"scale": s})
        c.add(f"ARankDivGOnePspin{SCN[s]}", r.p_spin_vasa, f2(r.p_spin_vasa), mp, src + "p_spin_vasa", L, {"scale": s})


def _setlevel(c: Collector) -> None:
    g = c.csv("modal_controllability/driver_gram_proxy.csv")
    gap = g[g.c == 0.1].median_q1G_minus_q1raw
    src = "modal_controllability/02_analyze_modal: modal_controllability/driver_gram_proxy.csv | c == 0.1; {} median_q1G_minus_q1raw"
    c.add("ADriverGramUpperGapMin", gap.min(), f2(gap.min()), "0.01", src.format("min"), L)
    c.add("ADriverGramUpperGapMax", gap.max(), f2(gap.max()), "0.04", src.format("max"), L)

    s = c.csv("modal_controllability/set_level_modal.csv.gz", usecols=["subject_id", "scale", "c", "k", "family", "sample", "nodes"])
    s = s[(s.c == 0.1) & (s["sample"] == -1) & s.family.isin(["top_modal_gu", "cycle_participation"])]
    w = s.pivot_table(index=["subject_id", "scale", "k"], columns="family", values="nodes", aggfunc="first").dropna()
    shared = w.apply(lambda r: len(set(r.top_modal_gu.split()) & set(r.cycle_participation.split())), axis=1)
    n_subj = shared[shared > 0].index.get_level_values("subject_id").nunique()
    n_reg = int(shared.sum())
    src = ("modal_controllability/01_compute_modal: modal_controllability/set_level_modal.csv.gz | c == 0.1, sample == -1, "
           "top_modal_gu vs cycle_participation, all scales and k; {}; printed as a word")
    c.add("AModalCycleSharedSubjects", n_subj, COUNT_WORDS.get(n_subj, str(n_subj)), "one", src.format("subjects with any shared region"), L)
    c.add("AModalCycleSharedRegions", n_reg, COUNT_WORDS.get(n_reg, str(n_reg)), "one", src.format("total shared regions"), L)

    cf = c.csv("modal_controllability/cycle_vs_counterfactual.csv")
    cf = cf[(cf.c == 0.1) & (cf.k == 3) & (cf.contrast == "cycle - top_modal_gu") & (cf.target == "effective_rank")]
    for sc, mm, mn in zip(SC, ("0.44", "0.48", "0.74"), ("55", "65", "69")):
        r = cf[cf.scale == sc].iloc[0]
        src = ("modal_controllability/02_analyze_modal: modal_controllability/cycle_vs_counterfactual.csv | c == 0.1, k == 3, "
               "contrast == 'cycle - top_modal_gu', target == effective_rank; {}")
        c.add(f"AModalCycleERMedianDiff{SCN[sc]}", r["median"], f2(r["median"]), mm, src.format("median"), L, {"scale": sc})
        c.add(f"AModalCycleERNPos{SCN[sc]}", r.n_pos, str(int(r.n_pos)), mn, src.format("n_pos"), L, {"scale": sc})


def _perturbation(c: Collector) -> None:
    tb = "topology/tables/"
    nc = c.csv(tb + "rep_null_comparisons.csv")
    x = float(nc[(nc.variant != "degree") & nc.k.isin([1, 2, 3, 5])].strength_mean.max())
    c.add("ARepStrengthNullMaxER", x, f2(ceil_to(x, 2)), "0.12",
          f"topology/summarize_representatives: {tb}rep_null_comparisons.csv | 4 conventions, k in 1,2,3,5, all scales; max "
          f"strength_mean = {x:.4f}, rounded up", L, bound=(x <= 0.12, "<="))
    rs = c.csv(tb + "perturb_rank_stability.csv")
    for fam, lvl, nm, lo_, hi_ in (("noise", 0.05, "NoiseFive", "0.94", "0.95"), ("dropout", 0.2, "RemovalTwenty", "0.56", "0.63")):
        v = rs[(rs.ranking == "ripser") & (rs.family == fam) & np.isclose(rs.level, lvl)].rho_median
        src = (f"topology/summarize_perturbation: {tb}perturb_rank_stability.csv | ranking == ripser, family == {fam}, "
               f"level == {lvl}; {{}} over scales of rho_median")
        c.add(f"APerturbRho{nm}Min", v.min(), f2(v.min()), lo_, src.format("min"), L)
        c.add(f"APerturbRho{nm}Max", v.max(), f2(v.max()), hi_, src.format("max"), L)
    rf = c.csv(tb + "perturb_rescaled_fraction.csv")
    v = rf[rf.family == "noise"].rescaled
    src = f"topology/summarize_perturbation: {tb}perturb_rescaled_fraction.csv | family == noise, all scales and levels; {{}} rescaled, %"
    c.add("APerturbRescaledMin", 100 * v.min(), pct(v.min()), "51", src.format("min"), L)
    c.add("APerturbRescaledMax", 100 * v.max(), pct(v.max()), "65", src.format("max"), L)


def _mediation(c: Collector) -> None:
    mn = c.csv("null_models/mediation_noncircular.csv")
    mn["k"] = mn["k"].astype(str)
    hub = mn[(mn.outcome == "degree_q1") & (mn.k == "avg") & (mn.mediator == "hub_strength_frac") & mn.null.isin(NULLS)]
    src = ("null_models/03_gap_mediators: null_models/mediation_noncircular.csv | outcome == degree_q1, k == avg, "
           "mediator == hub_strength_frac, 3 nulls x 3 scales; {} prop_mediated, %")
    hp = hub.prop_mediated
    c.add("AMediationHubFracMin", 100 * hp.min(), pct(hp.min()), "62", src.format("min"), L)
    c.add("AMediationHubFracMax", 100 * hp.max(), pct(hp.max()), "93", src.format("max"), L)
    rc = mn[mn.mediator.isin(["rc_bu", "rc_wu", "rc_wu_p75", "rc_wu_p85", "rc_wu_p95"])
            & (((mn.outcome == "degree_q1") & (mn.k == "avg")) | ((mn.outcome == "gap") & (mn.k == "-")))
            & mn.null.isin(NULLS)].prop_mediated
    rmax = float(rc.max())
    c.add("AMediationRichClubMax", 100 * rmax, f"{ceil_to(100 * rmax, 0):.0f}", "26",
          "null_models/03_gap_mediators: null_models/mediation_noncircular.csv | rich-club mediators; outcome degree_q1 (k avg) "
          f"or gap; 3 nulls x 3 scales; max prop_mediated = {100 * rmax:.2f}%, rounded up", L, bound=(100 * rmax <= 26, "<="))


def _targets(c: Collector) -> None:
    T = "target_states/"
    pa = c.csv(T + "parcel_network_assignment.csv")
    pur = pa.groupby("scale").yeo7_purity.median()
    for s, ms_ in zip(SC, ("0.62", "0.68", "0.80")):
        c.add(f"AYeoPurityMedian{SCN[s]}", pur[s], f2(pur[s]), ms_,
              f"target_states/01_build_targets: {T}parcel_network_assignment.csv | median yeo7_purity", L, {"scale": s})
    nts = c.csv(T + "named_target_summary.csv").set_index(["variant", "target_name", "scale"])
    for variant, tgt, stem, mss in (("full", "kw_occipital_module", "ATargetOccipitalFull", ("-8.2", "-8.5", "-1.2")),
                                    ("full", "net_Visual", "ATargetVisualFull", ("-7.2", "-8.9", "-1.8")),
                                    ("drivers_excluded", "net_Visual", "ATargetVisualExcl", ("-5.7", "-8.9", "-1.7"))):
        for s, ms_ in zip(SC, mss):
            v = nts.loc[(variant, tgt, s)].median_pct
            c.add(f"{stem}{SCN[s]}", v, f2(v, 1), ms_,
                  f"target_states/03_target_stats: {T}named_target_summary.csv | variant == {variant}, target == {tgt}; median_pct",
                  L, {"scale": s})
    ov = c.csv(T + "driver_target_overlap.csv").set_index(["target_name", "scale"])
    for col, stem, mss in (("frac_any_driver_in_cyc", "AVisualOverlapCycle", ("16", "22", "15")),
                           ("frac_any_driver_in_deg", "AVisualOverlapDegree", ("3", "1", "1"))):
        for s, ms_ in zip(SC, mss):
            v = ov.loc[("net_Visual", s)][col]
            c.add(f"{stem}{SCN[s]}", 100 * v, str(round(100 * float(f"{v:.2f}"))), ms_,
                  f"target_states/03_target_stats: {T}driver_target_overlap.csv | target == net_Visual; {col}, x100", L)
    for s, ms_ in zip(SC, ("-0.77", "-0.84", "-0.82")):
        v = nts.loc[("full", "net_Visual", s)].rho_pct_vs_dcoupling
        c.add(f"AVisualCouplingRho{SCN[s]}", v, f2(v), ms_,
              f"target_states/03_target_stats: {T}named_target_summary.csv | full, net_Visual; rho_pct_vs_dcoupling", L)
    pc = c.csv(T + "proximity_controls.csv").set_index(["target_name", "scale"])
    for s, ms_ in zip(SC, ("-1.1", "-6.9", "-2.8")):
        v = pc.loc[("net_Visual", s)].eqdist_intercept
        c.add(f"AVisualEqualDist{SCN[s]}", v, f2(v, 1), ms_,
              f"target_states/07_proximity_controls: {T}proximity_controls.csv | net_Visual; eqdist_intercept", L)
    sp = c.csv(T + "single_parcel_summary.csv")
    for variant, stem, mw, ms_ in (("full", "ASingleParcelCouplingRho", "-0.67", "-0.79"),
                                   ("drivers_excluded", "ASingleParcelCouplingRhoExcl", "-0.90", "-0.94")):
        rho = sp[sp.variant == variant].rho_parcels_pct_vs_dcoupling
        src = f"target_states/03_target_stats: {T}single_parcel_summary.csv | variant == {variant}; {{}} rho_parcels_pct_vs_dcoupling"
        c.add(f"{stem}Weak", rho.max(), f2(rho.max()), mw, src.format("max (weakest)"), L)
        c.add(f"{stem}Strong", rho.min(), f2(rho.min()), ms_, src.format("min (strongest)"), L)
    sl = c.csv(T + "sensitivity_long.csv.gz")
    sl = sl[(sl.c == 0.1) & (sl.eps == 1e-5)]
    med = sl.groupby(["scale", "target_name"]).floor_frac.median()
    src = f"target_states/04_sensitivity: {T}sensitivity_long.csv.gz | c == 0.1, eps == 1e-5; {{}} over scale x target of median floor_frac, %"
    c.add("ATargetFloorShareMin", 100 * med.min(), f"{100 * med.min():.0f}", "86", src.format("min"), L)
    c.add("ATargetFloorShareMax", 100 * med.max(), f"{100 * med.max():.0f}", "100", src.format("max ('nearly 100')"), L)


def _lesions(c: Collector) -> None:
    LE = "lesions/"
    cs = c.csv(LE + "crossover_stats.csv")
    pooled = cs[(cs.protocol == "static") & (cs.scale == "pooled")].set_index(["metric", "lesion_type"])
    r = pooled.loc[("dER", "degree")]
    src = f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | static, pooled, metric dER, degree lesion; "
    c.add("ALesionDegreeDiffER", r["mean"], f2(r["mean"]), "-0.35", src + "mean D", L)
    n_neg = int(r.n) - int(r.n_pos)
    c.add("ALesionDegreeDiffERNNeg", n_neg, str(n_neg), "66", src + "n - n_pos", L)
    for m, nm, ms_ in (("dPct_ER", "ER", "3.6"), ("dPct_PR", "PR", "5.2")):
        v = 100 * pooled.loc[(m, "cycle")]["mean"]
        c.add(f"ALesionCyclePctDiff{nm}", v, f"{v:.1f}", ms_,
              f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | static, pooled, metric {m}, cycle lesion; "
              "100 x mean D", L)
    d = c.csv(LE + "per_subject_degradation.csv")

    def intact(protocol, lesion, strategy):
        p = d[(d.protocol == protocol) & (d.lesion_type == lesion)]
        return float(p.pivot_table(index=["scale", "h", "subject"], columns="strategy", values="intact_effective_rank")[strategy].mean())

    src = f"lesions/02_analyze_reciprocal_lesion: {LE}per_subject_degradation.csv | "
    for (protocol, strategy, nm, ms_) in (("static", "degree_strength", "ALesionNextRankedDegreeER", "3.64"),
                                          ("recompute", "degree_strength", "ALesionTopDegreeER", "2.40"),
                                          ("recompute", "cycle_participation", "ALesionTopCycleER", "3.27")):
        v = intact(protocol, "degree", strategy)
        c.add(nm, v, f"{v:.2f}", ms_, src + f"{protocol}, degree lesion; mean intact_effective_rank of {strategy} sets", L)
    rows = c.csv(LE + "lesion_rows.csv.gz")
    st = rows[rows.protocol == "static"]
    rsz = float(np.corrcoef(st.lesioned_log10_average_energy - st.intact_log10_average_energy,
                            np.log10(st.n_lesioned / st.n_intact))[0, 1])
    c.add("ALesionEnergySizeCorr", rsz, f"{rsz:.3f}", "0.989",
          f"lesions/01_run_reciprocal_lesion: {LE}lesion_rows.csv.gz | static; Pearson r of the change in log10 E_avg with "
          "log10(n_lesioned/n_intact)", L)
    cells = cs[(cs.metric == "dlog10E") & (cs.protocol == "static") & (cs.scale != "pooled")
               & cs.lesion_type.isin(["degree", "cycle", "random"])]
    v = float(cells["mean"].abs().max())
    c.add("ALesionEnergyStrategyDiffMax", v, f"{ceil_to(v, 3):.3f}", "0.006",
          f"lesions/02_analyze_reciprocal_lesion: {LE}crossover_stats.csv | dlog10E, static, scale x h cells; max |mean D|, "
          "rounded up", L, bound=(v <= 0.006, "<="))
