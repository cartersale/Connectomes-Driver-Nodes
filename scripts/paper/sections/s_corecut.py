"""Full table S_corecut: first-order prediction of the core-cut eigenvalue changes.

Built from <run>/mechanisms/core_cut_first_order.csv (written by
scripts/analysis/mechanisms/07_core_cut.py).
"""

from __future__ import annotations

from pathlib import Path

from common import SCALES, Table, num, pd, sgn, tabular

PARTITIONS = [("labels", "Anatomical"), ("ycut", "Median coordinate"), ("v2sign", "$v_2$ sign")]
CONDITIONS = [("cross", "Front--back"), ("within_matched", "Within halves"), ("other_matched", "Outside core"),
              ("within_uniform", "Within halves, uniform"), ("other_uniform", "Outside core, uniform")]
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def build(res: Path) -> list[Table]:
    fo = pd.read_csv(res / "mechanisms" / "core_cut_first_order.csv")
    rows = []
    for scale in SCALES:
        for p_key, p_label in PARTITIONS:
            for c_key, c_label in CONDITIONS:
                sub = fo[(fo.scale == scale) & (fo.partition == p_key) & (fo.condition == c_key)].set_index("alpha")
                assert set(ALPHAS) <= set(sub.index), (scale, p_key, c_key)
                r = [str(scale) if (p_key, c_key) == ("labels", "cross") else "",
                     p_label if c_key == "cross" else "", c_label]
                for a in ALPHAS:
                    x = sub.loc[a]
                    r.append(f"{sgn(x.median_dgap_pct, 1)} / {sgn(x.median_pred_dgap_pct, 1)}")
                h = sub.loc[0.5]
                r += [f"{sgn(h.median_dl1, 3)} / {sgn(h.median_pred_dl1, 3)}",
                      f"{sgn(h.median_dl2, 3)} / {sgn(h.median_pred_dl2, 3)}",
                      num(h.rho, 2), num(sub.loc[1.0].v2_swap, 2)]
                rows.append(r)
        if scale != SCALES[-1]:
            rows.append("MIDRULE")
    tab = tabular("lll" + "c" * 8,
                  [" & & & \\multicolumn{4}{c}{Change in gap (\\%)} & \\multicolumn{2}{c}{$\\alpha = 0.5$} & & \\\\",
                   "\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}",
                   "Scale & Partition & Edges weakened & $\\alpha=0.25$ & $\\alpha=0.5$ & $\\alpha=0.75$ & $\\alpha=1$ & $\\Delta\\ell_1$ & $\\Delta\\ell_2$ "
                   "& $\\rho$ & $v_2$ swap \\\\"], rows)
    note = """S_corecut. Observed vs first-order predicted changes in l1, l2 and the gap when weight is removed from each edge class.
Source: <run>/mechanisms/core_cut_first_order.csv (07_core_cut.py, function first_order). Each cell is "observed / predicted",
median over 70 subjects; matched controls averaged over draws within subject first. Prediction per condition and draw:
d l_k = -2 sum_e removed_e v_k(i) v_k(j) with the unperturbed eigenvectors, d gap = d l1 - d l2. Gap change in % of the subject's
unperturbed gap. Front-back = every cross edge scaled by (1 - alpha); within halves / outside core = strength-matched controls removing
the same total weight (columns median_dgap_pct, median_pred_dgap_pct, median_dl1, median_pred_dl1, median_dl2, median_pred_dl2).
rho = Spearman over subjects of predicted vs observed d gap at alpha = 0.5. v2 swap = share of subjects (mean over draws for matched
controls) whose perturbed v2 has |cos| < 0.5 with the unperturbed v2, at alpha = 1. Median coordinate = core split at the median anterior-posterior centroid coordinate (ycut).
Uniform = the same total weight removed uniformly from every edge of the class (unmatched)."""
    return [Table("S_corecut", tab, note)]
