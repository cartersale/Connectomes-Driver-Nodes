#!/usr/bin/env python
"""Supplementary table C_scales: every main-text Results value at all three scales.

The Results quote the 114-region value of each number; this table lists the same quantity at 68, 114 and 219
regions. Values come from the registry (scripts/paper/registry): each quoted range registers its per-scale values
(``by_scale_text``) on the 114-region number, formatted as in the text. A cell holds a range when several values
share a scale (criteria, driver-set sizes or null models; the row says which).

    python scripts/paper/scale_table.py --run paper                    # writes <run>/paper/tables/C_scales.tex
    python scripts/paper/scale_table.py --run paper --manuscript DIR   # also copies it to DIR/supp_tables/

scripts/paper/tables.py writes it with the other supplement tables.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoc.runs import resolve_run_dir

# (Results subsection, quantity, registry prefix of the quoted range). Order follows the Results.
ROWS: list[tuple[str, str, str]] = [
    ("Energy floor", "Gramian eigenvalues below $\\epsilon$ (\\%; range over criteria)", "MFracBelowEps"),
    ("Combining hubs", "$q_1$, single degree-strength driver (\\%)", "MQOneSingleDegreePct"),
    ("Combining hubs", "$q_1$, single random driver (\\%)", "MQOneSingleRandomPct"),
    ("Combining hubs", "$q_1$, hub-oriented sets, $k = 2$--5 (\\%; range over criteria)", "MQOneHubTwoFive"),
    ("Combining hubs", "$q_1$, random sets, $k = 2$--5 (\\%)", "MQOneRandomTwoFive"),
    ("Combining hubs", "Percentile of hub-oriented sets among random sets (range over criteria)", "MHubPctTwoFive"),
    ("Combining hubs", "Excess $q_1$, two degree-strength drivers (percentage points)", "MExcessTwoDegreePts"),
    ("Combining hubs", "Excess $q_1$, five degree-strength drivers (percentage points)", "MExcessFiveDegreePts"),
    ("Combining hubs", "$\\lambda_{\\max}$, hub-oriented over random sets, $k = 2$--5 (median ratio; range over criteria)",
     "MAbsSetLead"),
    ("Combining hubs", "Controllability outside $v_1$, hub-oriented over random sets, $k = 2$--5 (\\%; range over criteria)",
     "MAbsSetPerp"),
    ("Combining hubs", "$q_1$, betweenness and cycle criticality (\\%)", "MQOneIntermediate"),
    ("Combining hubs", "$q_1$, remaining criteria (\\%)", "MQOneNonHub"),
    ("Combining hubs", "Cycle participation below degree strength in $q_1$, $k = 2$--5 (percentage points)", "MCycleBelowDegreePts"),
    ("Leading mode", "Squared weight of $v_1$ on the top 20\\% of regions (\\%)", "MVOneTopTwentyShare"),
    ("Leading mode", "Median correlation of subject $v_1$ with the group mean", "MVOneConsistencyMedian"),
    ("Leading mode", "Cosine similarity of strength and $v_1$", "MCosStrengthVOne"),
    ("Leading mode", "Median Spearman $\\rho$, strength vs neighbours' loading", "MRhoStrengthM"),
    ("Leading mode", "Loading $L_S$, hub-oriented sets (range over criteria)", "MLoadingHub"),
    ("Leading mode", "Loading $L_S$, random sets", "MLoadingRandom"),
    ("Leading mode", "Share of average controllability from $v_1$, top five regions (\\%)", "MAvgCtrlVOneShare"),
    ("Leading mode", "Cosine of Gramian top eigenvector with $v_1$, hub-oriented, $k = 2$--5", "MCosHubTwoFive"),
    ("Leading mode", "Cosine of Gramian top eigenvector with $v_1$, random, $k = 2$--5", "MCosRandomTwoFive"),
    ("Leading mode", "Cosine of Gramian top eigenvector with $v_1$, single hubs", "MCosHubSingle"),
    ("Spectral gap", "Squared weight of $v_2$ on the core (\\%)", "MVTwoOnCorePct"),
    ("Spectral gap", "Degree minus cycle $q_1$ difference removed with $v_1$ (\\%)", "MPerpFractionRemovedQOneKTwoPlusPct"),
    ("Spectral gap", "Excess $q_1$ of degree-strength sets, $c = 10^{-5}$", "MDegExcessQOneCTiny"),
    ("Spectral gap", "Excess $q_1$ of degree-strength sets, $c = 0.01$", "MDegExcessQOneCHundredth"),
    ("Spectral gap", "Excess $q_1$ of degree-strength sets, $c = 0.1$", "MDegExcessQOneCTenth"),
    ("Spectral gap", "Excess $q_1$ of degree-strength sets, $c = 1$", "MDegExcessQOneCOne"),
    ("Spectral gap", "Excess $q_1$, single degree-strength driver, $c = 0.1$", "MDegExcessQOneSingleCTenth"),
    ("Spectral gap", "Spearman $\\rho$, gap vs excess $q_1$", "MGapRhoDegExcess"),
    ("Spectral gap", "Partial Spearman $\\rho$, gap vs excess $q_1$", "MGapPartialRhoDegExcess"),
    ("Null networks", "Spectral gap, human", "MGapReal"),
    ("Null networks", "Spectral gap, nulls (range over models)", "MGapNull"),
    ("Null networks", "$q_1$ of degree-strength sets, human", "MNullRealQOne"),
    ("Null networks", "$q_1$ of degree-strength sets, nulls, same $c$ (range over models)", "MNullFixedQOne"),
    ("Null networks", "$q_1$ of degree-strength sets, nulls, gap-matched $c$ (range over models)", "MNullGapQOne"),
    ("Null networks", "Human minus null $q_1$, gap-matched $c$ (range over models)", "MNullGapQOneDiff"),
    ("Null networks", "Strength assortativity, human", "MStrengthAssortReal"),
    ("Null networks", "Strength assortativity, nulls (range over models)", "MStrengthAssortNull"),
    ("Null networks", "Spearman $\\rho$, excess assortativity vs excess gap", "MAssortGapExcessRho"),
]
SCALES = ("68", "114", "219")


def _cell(text: str) -> str:
    """'0.41--0.45' as is; a negative end gets a math minus: '-0.10--0.06' -> '$-$0.10 to $-$0.06'."""
    ends = text.split("--")
    if not any(e.startswith("-") for e in ends):
        return text
    ends = [f"$-${e[1:]}" if e.startswith("-") else e for e in ends]
    return " to ".join(ends)


def build(run: Path) -> str:
    from paper.registry import collect

    per = {n.context["range_of"]: n.context["by_scale_text"] for n in collect(run) if "range_of" in n.context}
    missing = [p for _, _, p in ROWS if p not in per]
    if missing:
        raise KeyError(f"no per-scale values registered for {missing}")
    lines = ["\\begin{tabular}{llccc}", "\\toprule", " & Quantity & 68 & 114 & 219 \\\\", "\\midrule"]
    last = None
    for section, label, prefix in ROWS:
        if last is not None and section != last:
            lines.append("\\midrule")
        lead = section if section != last else ""
        cells = " & ".join(_cell(per[prefix][s]) for s in SCALES)
        lines.append(f"{lead} & {label} & {cells} \\\\")
        last = section
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def write(run: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "C_scales.tex"
    path.write_text(build(run))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manuscript", type=Path, default=None, help="Manuscript folder; copies the table to supp_tables/.")
    args = parser.parse_args()
    run = resolve_run_dir(args.run)
    path = write(run, run / "paper" / "tables")
    print(f"wrote {path}")
    if args.manuscript is not None:
        dest = args.manuscript / "supp_tables" / path.name
        shutil.copy2(path, dest)
        print(f"copied to {dest}")


if __name__ == "__main__":
    main()
