#!/usr/bin/env python
"""Collect a run's paper outputs into <run>/paper/ and record their provenance.

    python scripts/paper/assemble.py --run paper

After this step <run>/paper/ holds everything the manuscript uses:
  figures/        main-text figures (PDF, SVG, PNG) and supp/ for the supplementary figures
  tables/         supplementary tables (LaTeX), built by tables.py
  key_results.json, numbers.csv, numbers.tex      the number registry, built by key_numbers.py
  validation_report.md                            built by validate.py
  provenance.json  config, seeds, input checksums, package versions, git commits and, per stage, the script
                   checksum, finish time and (for reused stages) where the outputs came from
"""

from __future__ import annotations

import json
import shutil

from hoc.runs import read_manifest, stage_from_argv

STAGE = "paper/assemble"
MAIN_FIGURES = ("fig1", "fig2", "fig3", "fig3b", "fig4", "fig5")
SUPP_FIGURES = {"concentration/fig_decay_collapse.png": "supp/fig_decay_collapse.png"}


def main() -> None:
    ctx = stage_from_argv(STAGE)
    ctx.require_stage("paper/validate")
    ctx.begin()
    run, out = ctx.root, ctx.root / "paper"
    written = []
    for name in MAIN_FIGURES:
        for ext in ("pdf", "svg", "png"):
            src = run / "figures" / f"{name}.{ext}"
            dst = out / "figures" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            written.append(dst)
    for src_rel, dst_rel in SUPP_FIGURES.items():
        dst = out / "figures" / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / src_rel, dst)
        written.append(dst)

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
    written.append(path)
    print(f"assembled {len(written)} files in {out}")
    ctx.finish(outputs=written)


if __name__ == "__main__":
    main()
