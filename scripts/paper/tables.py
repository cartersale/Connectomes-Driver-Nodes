#!/usr/bin/env python
"""Build every table from the outputs of a run.

    python scripts/paper/tables.py --run paper

Writes the full tables to <run>/paper/tables/<id>.tex, with <run>/paper/tables/notes/<id>.txt describing how
each was computed, and the tables the Supporting Information prints (C_<name>.tex and S_canonical.tex, built
from the full tables by scripts/paper/condense.py) to the same folder. scripts/paper/sync.py copies both into a
manuscript folder; scripts/paper/publish_tables.py publishes the full tables in supplementary/.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sections/ and common.py

import condense  # noqa: E402

from hoc.runs import stage_from_argv  # noqa: E402

# One module per group of full tables; each defines build(run) -> list[Table].
SECTIONS = ["s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s_canonical", "s_corecut"]
STAGE = "paper/tables"
REQUIRES = (
    "concentration/02_tables", "concentration/03_decay_collapse", "concentration/04_redundancy_geodesic",
    "null_models/03_gap_mediators", "null_models/04_rayleigh_bound", "target_states/09_report_tables",
    "lesions/02_analyze_reciprocal_lesion", "topology/summarize_perturbation", "topology/summarize_representatives",
    "mechanisms/01_hub_loading", "mechanisms/02_canonical_graphs", "mechanisms/03_coverage_distance", "mechanisms/07_core_cut", "sensitivity/00_validate",
    "sensitivity/07_report_tables", "spectra/02_analyze", "modal_controllability/02_analyze_modal",
)


def build_all(run: Path, sections: list[str] = SECTIONS) -> list:
    tables = []
    for name in sections:
        built = importlib.import_module(f"sections.{name}").build(run)
        ids = [t.id for t in built]
        assert len(ids) == len(set(ids)), f"{name}: duplicate table ids"
        tables.extend(built)
    return tables


def write(tables: list, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(exist_ok=True)
    written = []
    for table in tables:
        path = out / f"{table.id}.tex"
        path.write_text(table.tex.rstrip("\n") + "\n")
        written.append(path)
        if table.notes:
            note = out / "notes" / f"{table.id}.txt"
            note.write_text(table.notes.strip() + "\n")
            written.append(note)
    return written


def main() -> None:
    ctx = stage_from_argv(STAGE)
    for upstream in REQUIRES:
        ctx.require_stage(upstream)
    ctx.begin()
    out = ctx.root / "paper" / "tables"
    written = write(build_all(ctx.root), out)
    written += condense.build_all(out, out)
    print(f"wrote {len(written)} files to {out}")
    ctx.finish(outputs=written)


if __name__ == "__main__":
    main()
