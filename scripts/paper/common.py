"""Shared helpers for the supplement-table builders in sections/.

A builder module defines ``build(res: Path) -> list[Table]``, where ``res`` is the
run directory (runs/<name>/). ``tables.py`` runs every builder and writes the files.
Formatting is fixed: the tables must stay byte-identical to the release reference
(reference/tables.sha256, checked by validate.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np  # noqa: F401 - re-exported for builders
import pandas as pd  # noqa: F401 - re-exported for builders

SCALES = (68, 114, 219)

LABEL = {
    "degree_strength": "Degree strength", "degree": "Degree strength",
    "eigenvector_centrality": "Eigenvector centrality",
    "betweenness": "Betweenness",
    "participation_coefficient": "Participation coefficient",
    "triangle_participation": "Triangle participation",
    "tetrahedron_participation": "Tetrahedron participation",
    "cycle_participation": "Cycle participation",
    "cycle_resid": "Cycle part.\\ (degree-residualized)",
    "degree_normalized_cycle_participation": "Degree-normalized cycle part.",
    "cycle_criticality": "Cycle criticality",
    "top_avg_ctrl": "Top-$k$ average controllability",
    "top_modal_gu": "Top-$k$ modal controllability (Gu)",
    "top_modal_ct": "Top-$k$ modal controllability (CT)",
    "low_perron": "Lowest Perron loading",
    "random": "Random (100/subject)", "random_mean": "Random (mean of 100)",
    "degree_matched": "Degree-matched random",
    "spatial_matched": "Spatially matched random",
    "degree_spatial_matched": "Degree+spatially matched random",
    "ripser": "Ripser cocycle", "gudhi": "GUDHI pairing", "geodesic": "Length-minimized",
    "birth_edge": "Birth edge",
}


@dataclass
class Table:
    """One supplement table, written to ``<id>.tex`` (and ``<id>_long.tex``).

    ``tex`` is the tabular environment only (no comments). ``long`` also writes a
    longtable copy for tables that break across pages; ``long_colspec`` replaces
    the column spec in that copy when it differs. ``notes`` (source files,
    filters, definitions) go to ``notes/<id>.txt``.
    """

    id: str
    tex: str
    notes: str = ""
    long: bool = False
    long_colspec: str | None = None


def is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def sgn(x, d=2):
    if is_nan(x):
        return "--"
    return f"{x:+.{d}f}"


def num(x, d=2):
    if is_nan(x):
        return "--"
    return f"{x:.{d}f}"


def ptex(p):
    if is_nan(p):
        return "--"
    if p >= 0.995:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def npos(n, tot=70):
    return f"{int(round(n))}/{tot}"


def tabular(colspec, header_lines, rows):
    """rows: list of lists of strings, or the string 'MIDRULE' / ('MULTI', text, ncols)."""
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out += header_lines
    out.append("\\midrule")
    for r in rows:
        if r == "MIDRULE":
            out.append("\\midrule")
        elif isinstance(r, tuple) and r[0] == "MULTI":
            out.append(f"\\multicolumn{{{r[2]}}}{{l}}{{\\textit{{{r[1]}}}}} \\\\")
        else:
            out.append(" & ".join(r) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def to_longtable(tex: str, colspec: str | None = None) -> str:
    """The longtable copy used by \\ltab in supp_mat.tex."""
    lines = tex.splitlines()
    first = lines[0]
    assert first.startswith("\\begin{tabular}{"), first
    spec = first[len("\\begin{tabular}{"):-1] if colspec is None else colspec
    body = "\n".join(lines[1:]).replace("\\end{tabular}", "\\end{longtable}")
    return f"\\begin{{longtable}}{{{spec}}}\n\\caption{{\\ltcap}}\\label{{\\ltlab}}\\\\\n{body}"
