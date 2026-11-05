#!/usr/bin/env python
"""Collect a run's paper outputs into <run>/paper/ and record their provenance.

    python scripts/paper/assemble.py --run paper

After this step <run>/paper/ holds the tables, numbers and provenance the manuscript uses (the figures are in
<run>/figures/):
  tables/         full tables and the tables the Supporting Information prints (LaTeX), built by tables.py
  key_results.json, numbers.csv, numbers.tex      the number registry, built by key_numbers.py
  validation_report.md                            built by validate.py
  provenance.json  config, seeds, input checksums, package versions, git commits and, per stage, the script
                   checksum, finish time and (for reused stages) where the outputs came from
"""

from __future__ import annotations

import json

from hoc.runs import read_manifest, stage_from_argv

STAGE = "paper/assemble"


def main() -> None:
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("paper/validate")
    ctx.begin()
    run, out = ctx.root, ctx.root / "paper"
    manifest = read_manifest(run)
    stages = manifest["stages"]
    last_env = next((rec["environment"] for rec in reversed(list(stages.values())) if rec.get("environment")), {})
    provenance = {
        "run": manifest.get("run"),
        "config": manifest.get("config"),
        "random_seed": ctx.config.get("analysis", {}).get("random_seed"),
        "stage_seeds": ctx.config.get("seeds", {}),
        "subjects": manifest.get("subjects"),
        "inputs": manifest.get("inputs"),
        "environment": {k: last_env.get(k) for k in ("python", "platform", "packages", "blas", "threads")},
        "commits": sorted({a["git"]["commit"] for rec in stages.values() for a in rec.get("attempts", [])
                           if a.get("git", {}).get("commit")}),
        "stages": {
            name: {"finished": rec.get("finished"), "script": rec.get("script"),
                   "commit": (rec.get("git") or {}).get("commit"), "dirty": (rec.get("git") or {}).get("dirty"),
                   **({"reused_from": rec["reused_from"]} if rec.get("reused_from") else {})}
            for name, rec in stages.items()
        },
    }
    path = out / "provenance.json"
    path.write_text(json.dumps(provenance, indent=1, default=str) + "\n")
    print(f"wrote {path}")
    ctx.finish(outputs=[path])


if __name__ == "__main__":
    main()
