#!/usr/bin/env python
"""Copy a run's tables, number macros and figures into a manuscript folder.

    python scripts/paper/sync.py --run paper --manuscript PATH            # list what would change
    python scripts/paper/sync.py --run paper --manuscript PATH --apply    # copy

Copies <run>/paper/tables/*.tex to PATH/supp_tables/, <run>/paper/numbers.tex to PATH/numbers.tex, and the
main-text figures (PDF, SVG) to PATH/figs/. The manuscript is not part of this repository; nothing outside
runs/ is written unless --apply is given.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from hoc.runs import reader_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None)
    parser.add_argument("--manuscript", type=Path, required=True, help="Manuscript folder (holds supp_mat.tex).")
    parser.add_argument("--apply", action="store_true", help="Copy the files (default: only list changes).")
    args = parser.parse_args()

    source = reader_run_dir(["--run", args.run] if args.run else []) / "paper"
    pairs = [(p, args.manuscript / "supp_tables" / p.name) for p in sorted((source / "tables").glob("*.tex"))]
    pairs.append((source / "numbers.tex", args.manuscript / "numbers.tex"))
    pairs += [(p, args.manuscript / "figs" / p.name) for p in sorted((source / "figures").glob("fig*.*"))
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
