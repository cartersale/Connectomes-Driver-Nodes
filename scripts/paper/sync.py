#!/usr/bin/env python
"""Copy a run's tables, number macros and figures into a manuscript folder.

    python scripts/paper/sync.py --run paper --manuscript PATH            # list what would change
    python scripts/paper/sync.py --run paper --manuscript PATH --apply    # copy
    python scripts/paper/sync.py --run paper --manuscript PATH --figures --apply    # also copy the figures

Copies from <run>/paper/: the full tables named in scripts/paper/table_catalog.py to PATH/full_tables/, the tables
the Supporting Information inputs (scripts/paper/condense.py) to PATH/supp_tables/, numbers.tex to PATH/numbers.tex,
and, with --figures, the main-text figures (PDF, SVG) from <run>/figures/ to PATH/figures/. Figures are skipped by
default because the manuscript's copies are finished by hand in a vector editor and copying would overwrite them.
The manuscript is not part of this repository; nothing outside runs/ is written unless --apply is given.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from hoc.runs import reader_run_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from condense import SUPPLEMENT_TABLES  # noqa: E402
from table_catalog import GROUPS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None)
    parser.add_argument("--manuscript", type=Path, required=True, help="Manuscript folder (holds supp_mat.tex).")
    parser.add_argument("--apply", action="store_true", help="Copy the files (default: only list changes).")
    parser.add_argument("--figures", action="store_true",
                        help="Also copy the main-text figures (overwrites the manuscript's finished figures).")
    args = parser.parse_args()

    run = reader_run_dir(["--run", args.run] if args.run else [])
    source = run / "paper"
    ids = sorted({tid for _, _, _, tabs in GROUPS for _, tid, _ in tabs})
    pairs = [(source / "tables" / f"{tid}.tex", args.manuscript / "full_tables" / f"{tid}.tex") for tid in ids]
    pairs += [(source / "tables" / f"{t}.tex", args.manuscript / "supp_tables" / f"{t}.tex") for t in SUPPLEMENT_TABLES]
    pairs.append((source / "numbers.tex", args.manuscript / "numbers.tex"))
    if args.figures:
        pairs += [(p, args.manuscript / "figures" / p.name) for p in sorted((run / "figures").glob("fig*.*"))
                  if p.suffix in (".pdf", ".svg")]
    missing = [str(src) for src, _ in pairs if not src.exists()]
    if missing or not (source / "provenance.json").exists():
        sys.exit("the run's paper folder is incomplete; run scripts/reproduce.py --run <name> first")

    changed = [(src, dst) for src, dst in pairs if not dst.exists() or dst.read_bytes() != src.read_bytes()]
    for src, dst in changed:
        print(f"{'new    ' if not dst.exists() else 'changed'}  {dst.relative_to(args.manuscript)}")
    print(f"{len(changed)} of {len(pairs)} files {'copied' if args.apply else 'would change'}")
    if args.apply:
        for src, dst in changed:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
