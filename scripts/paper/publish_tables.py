#!/usr/bin/env python
"""Publish every full table in supplementary/ as CSV, with an index.

    python scripts/paper/publish_tables.py --manuscript PATH            # list what would change
    python scripts/paper/publish_tables.py --manuscript PATH --apply    # write supplementary/

PATH is the manuscript folder holding full_tables/, the .tex tabulars written by scripts/paper/tables.py.
Groups and captions come from scripts/paper/table_catalog.py. LaTeX markup is converted to plain text; the
values are not changed. supplementary/additional_analyses.md (methods for the analyses outside the paper) is
written by hand and left alone.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from condense import RULES, _group_label  # noqa: E402
from table_catalog import GROUPS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "supplementary"
TITLE = "Hub Inputs Funnel Control into the Connectome's Dominant Mode"

# ----------------------------------------------------------------------------- LaTeX -> text
SYMBOLS = {r"\in": " in ", r"\_": "_", r"\log_{10}": "log10 ", r"\log": "log ", r"\alpha": "alpha", r"\epsilon": "eps", r"\rho": "rho", r"\lambda": "lambda", r"\ell": "l", r"\sigma": "sigma",
           r"\Delta": "Delta ", r"\tilde z": "median z", r"\geq": ">=", r"\leq": "<=", r"\infty": "inf", r"\times": "x",
           r"\pm": "+/-", r"\beta": "beta", r"\pi": "pi", r"\phi": "phi", r"\xi": "xi", r"\mu": "mu", r"\bar r": "r_bar",
           r"\perp": "perp", r"\mid": "|", r"\to": "->", r"\cdot": "."}


def plain(cell: str) -> str:
    s = cell.strip().replace("``", '"').replace("''", '"')
    s = re.sub(r"\\binom\{(\w+)\}\{(\w+)\}", r"C(\1, \2)", s)
    s = re.sub(r"\$?([0-9.+\-]+)\\times10\^\{(-?\d+)\}\$?", r"\1e\2", s)  # 3.1\times10^{-5} -> 3.1e-5
    s = re.sub(r"\$?10\^\{(-?\d+)\}\$?", r"1e\1", s)
    s = s.replace("$-$", "-").replace("$+$", "+")
    s = re.sub(r"\\(?:textit|emph|textbf|texttt|mathrm|text|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", s)
    for k, v in sorted(SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(k, v)
    s = re.sub(r"\\H\{(.)\}|\\'\{?(.)\}?", lambda m: m.group(1) or m.group(2), s)
    s = s.replace("\\%", "%").replace("--", "-").replace("\\ ", " ").replace("\\,", " ").replace("~", " ")
    s = s.replace("\\{", "\x00").replace("\\}", "\x01")
    s = re.sub(r"[{}$]|\\(?=[A-Za-z])", "", s)
    s = s.replace("\x00", "{").replace("\x01", "}")
    s = re.sub(r"_\{?([A-Za-z0-9+,]+)\}?", r"_\1", s)
    s = re.sub(r"\^\{?([A-Za-z0-9+\-*]+)\}?", r"^\1", s)
    return re.sub(r"\s+", " ", s).strip()


def split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.rstrip().removesuffix("\\\\").split("&")]


def expand(cells: list[str]) -> list[str]:
    """Repeat a \\multicolumn{n} cell n times so header rows line up with the body."""
    out = []
    for c in cells:
        m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c)
        out += [m.group(2)] * int(m.group(1)) if m else [c]
    return out


def to_rows(tex: str) -> list[list[str]]:
    """Header (one row, upper header levels joined with ' ') and body rows; group labels become a column."""
    head, body = tex.split("\\midrule", 1)
    head_rows = [expand(split_cells(l)) for l in head.splitlines()
                 if "&" in l and l.strip().endswith("\\\\") and not l.strip().startswith(RULES)]
    width = max(len(r) for r in head_rows)
    header = []
    for j in range(width):
        parts = [plain(r[j]) for r in head_rows if j < len(r) and plain(r[j])]
        header.append(" ".join(parts) or ("Row" if j == 0 else f"col{j}"))
    rows, group, has_group = [], "", False
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(RULES):
            continue
        g = _group_label(line)
        if g is not None:
            group, has_group = plain(g), True
            continue
        if "&" not in line or not line.endswith("\\\\"):
            if line.endswith("\\\\") and line.startswith("\\multicolumn"):  # a note row spanning the table
                rows.append(["#", plain(line.removesuffix("\\\\"))])
            continue
        rows.append([group, *[plain(c) for c in expand(split_cells(line))]])
    if has_group:
        return [["Group", *header], *rows]
    return [header, *[r[1:] if r[0] != "#" else r[1:] for r in rows]]


# ----------------------------------------------------------------------------- index
def csv_name(key: str, slug: str, supports: bool) -> str:
    return f"{key}_{slug}.csv" if supports else f"additional_{slug}.csv"


def caption_text(caption: str, names: dict[str, str]) -> str:
    s = re.sub(r"\\(?:citet|citep|cite)\{[^}]*\}", "", caption)
    s = re.sub(r"\\ref\{tab:([^}]*)\}", lambda m: names[m.group(1)].removesuffix(".csv"), s)
    s = re.sub(r"\s+([.,;:)])", r"\1", plain(s.replace("\\protect", "")))
    return s.replace("Table ", "")


def build(manuscript: Path) -> dict[Path, bytes]:
    names = {slug: csv_name(key, slug, sup) for key, _, sup, tabs in GROUPS for slug, _, _ in tabs}
    files: dict[Path, bytes] = {}
    index = [
        "# Supplementary tables",
        "",
        f"Complete numerical results for *{TITLE}*, one CSV per table (LaTeX markup removed; values unchanged).",
        "",
        "Unless a description says otherwise: scales are 68, 114 and 219 regions; $c = 0.1$, $\\epsilon = 10^{-5}$, "
        "infinite horizon, $k \\in \\{1,2,3,5\\}$, 70 subjects, 100 random driver sets per subject, scale and $k$. "
        "Counts such as `64/70` are subjects with a positive difference.",
        "",
        "**Tables for the Supporting Information.** The files `S<n>_*.csv` give every setting of the analyses "
        "summarized in Section S*n* of the Supporting Information.",
        "",
        "**Additional analyses.** The files `additional_*.csv` hold analyses that are not part of the paper. "
        "Their methods are described in [`additional_analyses.md`](additional_analyses.md).",
        "",
        "**Reproducing.** `python scripts/reproduce.py --run paper` runs every analysis and writes the full tables "
        "to `runs/paper/paper/tables/`. `scripts/paper/publish_tables.py` converts them to the CSV files here. "
        "[`docs/paper_map.md`](../docs/paper_map.md) maps each table to the code and outputs it is built from.",
        "",
    ]
    heading = None
    for key, title, sup, tabs in GROUPS:
        part = "Tables for the Supporting Information" if sup else "Additional analyses"
        if part != heading:
            heading = part
            index += ["", f"## {part}"]
        index += ["", f"### {f'S{key[1:]}. ' if sup else ''}{title}", "", "| Table | Description |", "| --- | --- |"]
        for slug, tid, caption in tabs:
            src = manuscript / "full_tables" / f"{tid}.tex"
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(to_rows(src.read_text()))
            files[OUT / "tables" / names[slug]] = buf.getvalue().encode()
            desc = caption_text(caption, names).replace("|", "/")
            index.append(f"| [`{names[slug].removesuffix('.csv')}`](tables/{names[slug]}) | {desc} |")
    files[OUT / "README.md"] = ("\n".join(index) + "\n").encode()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Write the files (default: only list changes).")
    args = parser.parse_args()
    files = build(args.manuscript)
    keep = set(files) | {OUT / "additional_analyses.md"}
    stale = [p for p in OUT.rglob("*") if p.is_file() and p not in keep] if OUT.exists() else []
    changed = [p for p, b in files.items() if not p.exists() or p.read_bytes() != b]
    for p in changed:
        print(f"{'new    ' if not p.exists() else 'changed'}  {p.relative_to(REPO)}")
    for p in stale:
        print(f"remove   {p.relative_to(REPO)}")
    print(f"{len(changed)} of {len(files)} files {'written' if args.apply else 'would change'}, {len(stale)} to remove")
    if args.apply:
        for p, b in files.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        for p in stale:
            p.unlink()


if __name__ == "__main__":
    main()
