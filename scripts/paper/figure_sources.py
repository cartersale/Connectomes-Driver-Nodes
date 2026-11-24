#!/usr/bin/env python
"""Track which run figure each hand-finished manuscript figure was built from.

    python scripts/paper/figure_sources.py --run paper --manuscript PATH               # report stale figures
    python scripts/paper/figure_sources.py --run paper --manuscript PATH --mark fig4   # record fig4 as finished

The manuscript's figures (PATH/figures/figN.pdf) are finished by hand from <run>/figures/figN.svg, so the pipeline
never writes them. After finishing a figure, --mark records the sha256 of the run's PNG preview in
PATH/figures/sources.json. The report then lists every figure whose run PNG has changed since. The PNG is used
because matplotlib writes it byte-identically for identical content; the SVG and PDF embed dates and random ids.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from hoc.runs import reader_run_dir, sha256_file


def run_figures(run: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted((run / "figures").glob("fig*.png"))}


def status(run: Path, manuscript: Path) -> list[str]:
    record_path = manuscript / "figures" / "sources.json"
    recorded = json.loads(record_path.read_text()) if record_path.exists() else {}
    lines = []
    for name, png in run_figures(run).items():
        entry = recorded.get(name)
        if entry is None:
            lines.append(f"not recorded  {name}")
        elif entry["png_sha256"] != sha256_file(png):
            lines.append(f"stale         {name}  (run figure changed since {entry['recorded']}, run {entry['run']})")
        else:
            lines.append(f"up to date    {name}")
    return lines


def mark(run: Path, manuscript: Path, names: list[str]) -> None:
    figures = run_figures(run)
    unknown = [n for n in names if n not in figures]
    if unknown:
        raise SystemExit(f"no PNG in {run / 'figures'} for {unknown}")
    record_path = manuscript / "figures" / "sources.json"
    recorded = json.loads(record_path.read_text()) if record_path.exists() else {}
    for n in names:
        recorded[n] = {"run": run.name, "png_sha256": sha256_file(figures[n]), "recorded": dt.date.today().isoformat()}
    record_path.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--mark", nargs="+", default=[], metavar="FIG", help="Record these figures as finished.")
    args = parser.parse_args()
    run = reader_run_dir(["--run", args.run] if args.run else [])
    if args.mark:
        mark(run, args.manuscript, args.mark)
    print("\n".join(status(run, args.manuscript)))


if __name__ == "__main__":
    main()
