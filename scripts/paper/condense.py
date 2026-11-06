#!/usr/bin/env python
"""Build the short tables printed in the supplement from the full tables.

    python scripts/paper/condense.py --manuscript PATH

Reads the full tables in PATH/full_tables/ (written by scripts/paper/tables.py, copied by sync.py) and writes
the tables the supplement inputs to PATH/supp_tables/: C_<name>.tex, and S_canonical.tex copied unchanged. Every cell is copied from a full table, never recomputed, so the supplement and the full
tables cannot disagree. Each short table keeps the primary settings (c = 0.1, epsilon = 1e-5, k-averaged) and
the three scales side by side; the full tables keep every setting.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SCALES = ("68", "114", "219")
RULES = ("\\toprule", "\\midrule", "\\bottomrule", "\\addlinespace", "\\cmidrule", "\\begin{", "\\end{", "\\caption",
         "\\endfirsthead", "\\endhead", "\\endfoot", "\\endlastfoot", "\\hline")


# ----------------------------------------------------------------------------- parsing
@dataclass
class Row:
    group: str
    cells: list[str]


@dataclass
class Parsed:
    rows: list[Row] = field(default_factory=list)

    def find(self, first: str | None = None, group: str | None = None, **match: str) -> Row:
        hits = self.select(first, group, **match)
        assert len(hits) == 1, (first, group, match, len(hits))
        return hits[0]

    def select(self, first: str | None = None, group: str | None = None, **match: str) -> list[Row]:
        out = []
        for r in self.rows:
            if first is not None and r.cells[0] != first:
                continue
            if group is not None and r.group != group:
                continue
            if any(r.cells[int(k[1:])] != v for k, v in match.items()):
                continue
            out.append(r)
        return out


def _group_label(line: str) -> str | None:
    m = re.match(r"\\multicolumn\{\d+\}\{l\}\{\\(?:textit|emph)\{(.*)\}\}\s*\\\\\s*$", line)
    return m.group(1) if m else None


def parse(path: Path) -> Parsed:
    """Body rows of a booktabs tabular/longtable; blank leading cells are filled down from the row above."""
    text = path.read_text()
    body = text.split("\\midrule", 1)[1] if "\\midrule" in text else text
    out, group, prev = Parsed(), "", None
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(RULES):
            continue
        g = _group_label(line)
        if g is not None:
            group, prev = g, None
            continue
        if "&" not in line or not line.endswith("\\\\"):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        if prev is not None:
            for i, c in enumerate(cells):
                if c:
                    break
                cells[i] = prev[i]
        out.rows.append(Row(group, cells))
        prev = cells
    return out


# ----------------------------------------------------------------------------- formatting helpers
def count(n_of: str) -> str:
    """'64/70' -> '64'."""
    return n_of.split("/")[0].strip()


def with_n(value: str, n_of: str) -> str:
    return f"{value} ({count(n_of)})"


def strip_ci(cell: str) -> str:
    """'0.95 [0.92, 0.96]' -> '0.95'; '+0.53 ($3.0\\times10^{-6}$)' -> '+0.53'."""
    return re.split(r"\s+[\[(]", cell, maxsplit=1)[0]


def tabular(colspec: str, header: list[str], rows: list) -> str:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header, "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append("\\midrule")
        elif r == "ADDLINE":
            lines.append("\\addlinespace")
        elif isinstance(r, str):
            lines.append(r)
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def scale_header(lead: int, per: int, sub: list[str], lead_names: list[str]) -> list[str]:
    top = " & " * lead + " & ".join(f"\\multicolumn{{{per}}}{{c}}{{{s}}}" for s in SCALES) + " \\\\"
    rules = "".join(f"\\cmidrule(lr){{{lead + 1 + i * per}-{lead + (i + 1) * per}}}" for i in range(3))
    return [top, rules, " & ".join(lead_names + sub * 3) + " \\\\"]


def metric_header(lead_names: list[str], groups: list[str]) -> list[str]:
    """Column groups (one per metric), each with the three scales."""
    lead = len(lead_names)
    top = " & " * lead + " & ".join(f"\\multicolumn{{3}}{{c}}{{{g}}}" for g in groups) + " \\\\"
    rules = "".join(f"\\cmidrule(lr){{{lead + 1 + 3 * i}-{lead + 3 * (i + 1)}}}" for i in range(len(groups)))
    return [top, rules, " & ".join(lead_names + list(SCALES) * len(groups)) + " \\\\"]


# ----------------------------------------------------------------------------- short tables


def c_eps(t: Path) -> str:
    """S2: the regularization floor and average energy at three values of epsilon (from S4_2, S4_3)."""
    floor, er = parse(t / "S4_2.tex"), parse(t / "S4_3.tex")
    rows = []
    for s, sc in enumerate(SCALES):
        for i, hit in enumerate(floor.select(group=f"{sc} regions")):
            eps = hit.cells[0]
            d_er = er.find(eps, group="c = 0.1").cells[1 + s]
            rows.append([sc if i == 0 else "", eps, hit.cells[2], hit.cells[3], hit.cells[4],
                         hit.cells[5].replace("/70)", ")"), d_er, hit.cells[7], hit.cells[8]])
        if sc != SCALES[-1]:
            rows.append("MIDRULE")
    header = [" & & & & & \\multicolumn{2}{c}{Cycle $-$ degree} & \\multicolumn{2}{c}{IQR of $\\log_{10}E_{\\mathrm{avg}}$} \\\\",
              "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
              "Scale & $\\epsilon$ & Frac.\\ $\\lambda(W) < \\epsilon$ & $\\log_{10}E_{\\mathrm{avg}}$ & $\\log_{10}(N/\\epsilon)$ & "
              "$\\Delta\\log_{10}E_{\\mathrm{avg}}$ ($\\times10^{-3}$) & $\\Delta$ ER & All ($\\times10^{-3}$) & Above $\\epsilon$ \\\\"]
    return tabular("ll" + "c" * 7, header, rows)


def c_horizon(t: Path) -> str:
    """S2: finite horizons (from S5_1), epsilon = 1e-5, effective rank."""
    p = parse(t / "S5_1.tex")
    rows = []
    for c in ("0.01", "0.1", "1"):
        for i, hit in enumerate(p.select("$10^{-5}$", group=f"c = {c}")):
            rows.append([c if i == 0 else "", hit.cells[1], *hit.cells[2:5]])
        if c != "1":
            rows.append("ADDLINE")
    return tabular("llccc", ["$c$ & $T$ & 68 & 114 & 219 \\\\"], rows)


def c_gapcorr(t: Path) -> str:
    """S4: spectral gap vs degree-minus-random q1 across subjects (from S5_9)."""
    p = parse(t / "S5_9.tex")
    rows = []
    for hit in p.select(group="Degree $-$ random $q_1$"):
        rows.append([hit.cells[0], *[strip_ci(c) for c in hit.cells[1:4]], *hit.cells[7:10]])
    return tabular("l" + "c" * 6, metric_header(["$c$"], ["Spearman $\\rho$", "Partial $\\rho$"]), rows)


CRITERIA = ["Degree strength", "Eigenvector centrality", "Top-$k$ average controllability", "Betweenness",
            "Participation coefficient", "Triangle participation", "Tetrahedron participation", "Cycle participation",
            "Cycle part.\\ (degree-residualized)", "Degree-normalized cycle part.", "Cycle criticality",
            "Top-$k$ modal controllability (Gu)", "Top-$k$ modal controllability (CT)", "Lowest Perron loading"]


def c_criteria(t: Path) -> str:
    """S3: every criterion against random sets (from S6_1, S6_2): q1 at k = 1 and k >= 2, ER at k >= 2."""
    q1, er = parse(t / "S6_1.tex"), parse(t / "S6_2.tex")
    rows = []
    for crit in CRITERIA:
        r = [crit]
        for tab, k in ((q1, "1"), (q1, "avg $k\\geq2$"), (er, "avg $k\\geq2$")):
            hit = tab.find(k, group=crit)
            r += [with_n(hit.cells[1 + 3 * i], hit.cells[3 + 3 * i]) for i in range(3)]
        rows.append(r)
    return tabular("l" + "c" * 9, metric_header(["Criterion"], ["$q_1$, $k = 1$", "$q_1$, $k \\geq 2$",
                                                               "Effective rank, $k \\geq 2$"]), rows)


MATCHED = [("Cycle $-$ degree", "Cycle $-$ degree"), ("Cycle $-$ random", "Cycle $-$ random"),
           ("Cycle $-$ degree-matched", "Cycle $-$ strength-matched"),
           ("Cycle $-$ spatially matched", "Cycle $-$ spatially matched"),
           ("Cycle $-$ degree+spatial matched", "Cycle $-$ strength- and spatially matched"),
           ("Degree-residualized cycle $-$ random", "Residualized cycle $-$ random"),
           ("Degree-residualized cycle $-$ degree", "Residualized cycle $-$ degree")]


def c_matched(t: Path) -> str:
    """S3: matched and residualized controls (from S6_4, S6_5), k >= 2."""
    q1, er = parse(t / "S6_4.tex"), parse(t / "S6_5.tex")
    rows = []
    for group, label in MATCHED:
        r = [label]
        for tab in (q1, er):
            hit = [x for x in tab.select(group=group) if x.cells[0].startswith("avg $k\\geq2$")]
            assert len(hit) == 1, (group, len(hit))
            r += [with_n(hit[0].cells[1 + 3 * i], hit[0].cells[2 + 3 * i]) for i in range(3)]
        rows.append(r)
    return tabular("l" + "c" * 6, metric_header(["Comparison"], ["$q_1$", "Effective rank"]), rows)


def c_nullspec(t: Path) -> str:
    """S5: leading eigenvalues and gap in empirical and null networks (from S8_2)."""
    p = parse(t / "S8_2.tex")
    rows = []
    for stat, label in (("$\\lambda_1$", "$\\ell_1$"), ("$\\lambda_2$", "$\\ell_2$"), ("$\\lambda_1-\\lambda_2$", "Gap")):
        hits = p.select(stat)
        assert len(hits) == 3, stat
        for i, hit in enumerate(hits):
            r = [label if i == 0 else "", hit.cells[1]]
            for s in range(3):
                real, null, z, cnt = hit.cells[2 + 4 * s: 6 + 4 * s]
                r += [real if i == 0 else "", null, z]
            rows.append(r)
        if stat != "$\\lambda_1-\\lambda_2$":
            rows.append("ADDLINE")
    return tabular("ll" + "c" * 9, scale_header(2, 3, ["Real", "Null", "$\\tilde z$"], ["", "Null model"]), rows)


def c_nullq1(t: Path) -> str:
    """S5: degree-set q1 at c = 0.1 (k = 2, 3, 5) in empirical and null networks (from S8_3)."""
    p = parse(t / "S8_3.tex")
    hits = [r for r in p.rows if r.cells[0] == "0.1"]
    assert len(hits) == 7, len(hits)
    rows = []
    for i, hit in enumerate(hits):
        mode = {"--": "", "fixed": "Same $c$", "gap-matched": "Gap-matched $c$"}[hit.cells[1]]
        label = hit.cells[2] if i == 0 else f"{hit.cells[2]}"
        r = [mode if hit.cells[1] != (hits[i - 1].cells[1] if i else None) else "", label]
        for s in range(3):
            mean, z, above, _ = hit.cells[3 + 4 * s: 7 + 4 * s]
            r += [mean, z, count(above) if above else ""]
        rows.append(r)
        if i in (0, 3):
            rows.append("ADDLINE")
    return tabular("ll" + "c" * 9, scale_header(2, 3, ["Mean", "$\\tilde z$", "$>$95th"], ["Damping", "Network"]), rows)


CORECUT_KEEP = [("Anatomical", "Front--back", "Front--back"),
                ("Anatomical", "Within halves", "Within halves (matched)"),
                ("Anatomical", "Within halves, uniform", "Within halves (uniform)"),
                ("Anatomical", "Outside core", "Outside core (matched)"),
                ("Anatomical", "Outside core, uniform", "Outside core (uniform)"),
                ("Median coordinate", "Front--back", "Median split: front--back"),
                ("Median coordinate", "Within halves", "Median split: within halves (matched)"),
                ("$v_2$ sign", "Front--back", "$v_2$ sign: front--back")]


def c_corecut(t: Path) -> str:
    """S6: weakening core links, observed / first-order predicted (from S_corecut). The median-coordinate split
    equals the anatomical one at 219 regions, so its rows are shown at 68 and 114 regions only."""
    p = parse(t / "S_corecut.tex")
    rows = []
    for sc in SCALES:
        keep = [k for k in CORECUT_KEEP if not (sc == "219" and k[0] == "Median coordinate")]
        for j, (part, edges, label) in enumerate(keep):
            hit = p.find(sc, c1=part, c2=edges)
            # columns: 3..6 gap at alpha .25 .5 .75 1; 7 dl1; 8 dl2; 9 rho; 10 v2 swap
            rows.append([sc if j == 0 else "", label, hit.cells[4], hit.cells[6], hit.cells[7], hit.cells[8],
                         hit.cells[9], hit.cells[10]])
        if sc != SCALES[-1]:
            rows.append("MIDRULE")
    header = [" & & \\multicolumn{2}{c}{Change in gap (\\%)} & \\multicolumn{3}{c}{$\\alpha = 0.5$} & \\\\",
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
              "Scale & Edges weakened & $\\alpha=0.5$ & $\\alpha=1$ & $\\Delta\\ell_1$ & $\\Delta\\ell_2$ & $\\rho$ & "
              "$v_2$ rotated \\\\"]
    return tabular("llcccccc", header, rows)


BUILDERS = {"C_eps": c_eps, "C_horizon": c_horizon, "C_criteria": c_criteria, "C_matched": c_matched,
            "C_gapcorr": c_gapcorr, "C_nullspec": c_nullspec, "C_nullq1": c_nullq1, "C_corecut": c_corecut}


COPIED = ("S_canonical",)  # printed in the supplement as is
SUPPLEMENT_TABLES = (*BUILDERS, *COPIED)  # every table supp_mat.tex inputs


def build_all(full: Path, out: Path) -> list[Path]:
    """Write the supplement's tables to ``out`` from the full tables in ``full`` (the two may be the same folder)."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in BUILDERS.items():
        (out / f"{name}.tex").write_text(build(full))
        written.append(out / f"{name}.tex")
    for name in COPIED:
        if full.resolve() != out.resolve():
            shutil.copy2(full / f"{name}.tex", out / f"{name}.tex")
        written.append(out / f"{name}.tex")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript", type=Path, required=True, help="Manuscript folder (holds supp_mat.tex).")
    args = parser.parse_args()
    for path in build_all(args.manuscript / "full_tables", args.manuscript / "supp_tables"):
        print(f"wrote {path.relative_to(args.manuscript)}")


if __name__ == "__main__":
    main()
