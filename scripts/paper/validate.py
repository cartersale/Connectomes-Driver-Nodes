#!/usr/bin/env python
"""Check a run's paper numbers and tables against the release reference, and report on the manuscript.

    python scripts/paper/validate.py --run paper                     # as the pipeline step paper/validate
    python scripts/paper/validate.py --run paper --manuscript DIR    # also check a manuscript folder (read-only)
    python scripts/paper/validate.py --run paper --write-reference --reason "..."   # (re)create the reference

Checks (any failure exits non-zero, and the pipeline step does not complete):
  1. every number in <run>/paper/key_results.json against reference/paper_numbers.json: same names, values
     equal to a relative tolerance of 1e-9, identical presentation text;
  2. every table in <run>/paper/tables/ against reference/tables.sha256 (byte-identical).
Reported, not failed:
  3. numbers whose computed text differs from the token the manuscript prints (registry ``printed``);
  4. stages whose outputs were reused from another run.
With --manuscript DIR (a folder holding new_main.tex and supp_mat.tex):
  5. tables the supplement inputs that the run does not generate, and generated tables it does not input;
  6. printed tokens no longer found in the manuscript (the registry's record of the text is out of date);
  7. LaTeX number macros the manuscript uses that the registry does not define.

The report is written to <run>/paper/validation_report.md. The reference is never updated implicitly:
--write-reference refuses to replace an existing reference without --reason, which is recorded in it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from hoc.paths import REPO_ROOT
from hoc.runs import git_state, read_manifest, resolve_run_dir, stage_from_argv

REFERENCE = REPO_ROOT / "reference"
NUMBERS_REF = REFERENCE / "paper_numbers.json"
TABLES_REF = REFERENCE / "tables.sha256"
RTOL = 1e-9
STAGE = "paper/validate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_value(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=1e-15)


def load_run(run_dir: Path) -> tuple[dict, dict[str, Path]]:
    numbers = json.loads((run_dir / "paper" / "key_results.json").read_text())
    tables = {p.name: p for p in sorted((run_dir / "paper" / "tables").glob("*.tex"))}
    return numbers, tables


def compare_reference(numbers: dict, tables: dict[str, Path]) -> list[str]:
    lines, failures = [], []
    if not NUMBERS_REF.exists() or not TABLES_REF.exists():
        return [f"- **FAIL** no release reference in {REFERENCE.relative_to(REPO_ROOT)}/ "
                "(create it from a validated run with --write-reference)"]
    ref = json.loads(NUMBERS_REF.read_text())["numbers"]
    for name in sorted(set(ref) - set(numbers)):
        failures.append(f"- **FAIL** `{name}` is in the reference but was not computed")
    for name in sorted(set(numbers) - set(ref)):
        failures.append(f"- **FAIL** `{name}` was computed but is not in the reference")
    for name in sorted(set(ref) & set(numbers)):
        r, n = ref[name], numbers[name]
        if not same_value(n["value"], r["value"]) or n["text"] != r["text"]:
            failures.append(f"- **FAIL** `{name}`: {n['value']!r} ({n['text']}) vs reference {r['value']!r} ({r['text']})")
    ref_tables = dict(line.split(maxsplit=1)[::-1] for line in TABLES_REF.read_text().splitlines() if line.strip())
    for name in sorted(set(ref_tables) - set(tables)):
        failures.append(f"- **FAIL** table `{name}` is in the reference but was not generated")
    for name in sorted(set(tables) - set(ref_tables)):
        failures.append(f"- **FAIL** table `{name}` was generated but is not in the reference")
    for name in sorted(set(ref_tables) & set(tables)):
        if sha256(tables[name]) != ref_tables[name]:
            failures.append(f"- **FAIL** table `{name}` differs from the reference")
    meta = json.loads(NUMBERS_REF.read_text()).get("meta", {})
    lines.append(f"Reference: {len(ref)} numbers, {len(ref_tables)} tables (created {meta.get('created')} from run "
                 f"`{meta.get('run')}` at commit {str(meta.get('commit'))[:10]}).")
    lines.append("")
    lines += failures or [f"All {len(numbers)} numbers and {len(tables)} tables match the reference."]
    return lines


def printed_differences(numbers: dict) -> list[str]:
    rows = [(k, v["text"], v["context"]["printed"], v["locus"], v["source"]) for k, v in numbers.items()
            if "printed" in v["context"] and v["context"]["printed"] != v["text"]]
    if not rows:
        return ["Every computed number matches the token the manuscript prints."]
    out = [f"{len(rows)} numbers differ from the manuscript text. The computed value is the result; the manuscript "
           "should be corrected (or the sentence's selection reconsidered):", "",
           "| name | computed | printed | where | source |", "|---|---|---|---|---|"]
    out += [f"| `{k}` | {t} | {p} | {', '.join(loc)} | {s.split('|')[0].strip()} |" for k, t, p, loc, s in rows]
    unprinted = [k for k, v in numbers.items() if "printed" not in v["context"]]
    if unprinted:
        out += ["", f"{len(unprinted)} numbers have no recorded printed token: " + ", ".join(f"`{k}`" for k in unprinted)]
    return out


def reused(run_dir: Path) -> list[str]:
    stages = (read_manifest(run_dir) or {}).get("stages", {})
    rows = [(name, rec["reused_from"]) for name, rec in stages.items() if rec.get("reused_from")]
    if not rows:
        return ["Every stage was computed in this run."]
    out = []
    for name, info in rows:
        out.append(f"- `{name}`: {info['files']} files taken from run `{info['run']}` (finished there "
                   f"{info.get('stage_record_finished')}); every file matched its recorded sha256.")
        if info.get("config_diff"):
            out += ["  The two runs' configs differ:", "", "  ```diff", *("  " + ln for ln in info["config_diff"].splitlines()),
                    "  ```"]
    return out


def manuscript_checks(numbers: dict, tables: dict[str, Path], folder: Path) -> list[str]:
    main = (folder / "new_main.tex").read_text() if (folder / "new_main.tex").exists() else ""
    supp = (folder / "supp_mat.tex").read_text() if (folder / "supp_mat.tex").exists() else ""
    if not main and not supp:
        return [f"- **no new_main.tex or supp_mat.tex in {folder}**"]
    out = []
    cited = set(re.findall(r"\\[sl]tab\{([^}]+)\}", supp))
    cited_files = {f"{c}_long.tex" if f"\\ltab{{{c}}}" in supp else f"{c}.tex" for c in cited}
    missing = sorted(c for c in cited_files if c not in tables)
    uncited = sorted(t for t in tables if t not in cited_files)
    out.append(f"- Supplement inputs {len(cited_files)} tables; " + (f"**not generated: {missing}**" if missing else "all generated."))
    out.append(f"- Generated but not input by the supplement ({len(uncited)}): {', '.join(uncited) or 'none'}")
    text = main + "\n" + supp
    stale = []
    for name, v in numbers.items():
        token = v["context"].get("printed")
        if token and not re.search(r"(?<![\d.])" + re.escape(token) + r"(?![\d])", text):
            stale.append(f"`{name}` ({token})")
    out.append(f"- Printed tokens no longer found in the manuscript ({len(stale)}): " + (", ".join(stale) or "none"))
    used = set(re.findall(r"\\([MS][A-Z][A-Za-z]+)\b", text))
    undefined = sorted(m for m in used if m not in numbers)
    out.append(f"- Number macros used in the manuscript: {len(used & set(numbers))} defined"
               + (f", **{len(undefined)} undefined: {undefined}**" if undefined else ""))
    return out


def write_reference(run_dir: Path, numbers: dict, tables: dict[str, Path], reason: str | None) -> None:
    if (NUMBERS_REF.exists() or TABLES_REF.exists()) and not reason:
        sys.exit("a release reference already exists; replacing it needs --reason (recorded in the file)")
    REFERENCE.mkdir(exist_ok=True)
    previous = json.loads(NUMBERS_REF.read_text()).get("meta", {}) if NUMBERS_REF.exists() else None
    meta = {"created": dt.date.today().isoformat(), "run": run_dir.name, "commit": git_state()["commit"],
            "reason": reason or "initial release reference", "replaces": previous}
    payload = {"meta": meta, "numbers": {k: {"value": v["value"], "text": v["text"], "source": v["source"],
                                             "locus": v["locus"]} for k, v in numbers.items()}}
    NUMBERS_REF.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    TABLES_REF.write_text("".join(f"{sha256(p)}  {name}\n" for name, p in sorted(tables.items())))
    print(f"wrote {NUMBERS_REF.relative_to(REPO_ROOT)} ({len(numbers)} numbers) and "
          f"{TABLES_REF.relative_to(REPO_ROOT)} ({len(tables)} tables)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--manuscript", type=Path, default=None, help="Folder with new_main.tex and supp_mat.tex.")
    p.add_argument("--write-reference", action="store_true", help="Write reference/ from this run.")
    p.add_argument("--reason", default=None, help="Why an existing reference is replaced (required to replace).")
    args = p.parse_args()
    standalone = args.manuscript is not None or args.write_reference
    ctx = None
    if standalone:
        run_dir = resolve_run_dir(args.run)
    else:
        ctx = stage_from_argv(STAGE)
        ctx.require_stage("paper/numbers")
        ctx.begin()
        run_dir = ctx.root
    numbers, tables = load_run(run_dir)
    if args.write_reference:
        write_reference(run_dir, numbers, tables, args.reason)
        return

    ref_lines = compare_reference(numbers, tables)
    failed = any("**FAIL**" in line for line in ref_lines)
    report = [f"# Validation report: run `{run_dir.name}`", "",
              f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')}, commit {str(git_state()['commit'])[:10]}", "",
              "## 1-2. Numbers and tables against the release reference", "", *ref_lines, "",
              "## 3. Computed numbers against the manuscript text", "", *printed_differences(numbers), "",
              "## 4. Reused stages", "", *reused(run_dir), ""]
    if args.manuscript:
        report += ["## 5-7. Manuscript folder " + f"`{args.manuscript}`", "", *manuscript_checks(numbers, tables, args.manuscript), ""]
    out = run_dir / "paper" / ("validation_report.md" if not standalone else "validation_report_manuscript.md")
    out.write_text("\n".join(report))
    print("\n".join(report))
    if failed:
        sys.exit(f"VALIDATION FAILED (report: {out})")
    if ctx is not None:
        ctx.finish(outputs=[out])


if __name__ == "__main__":
    main()
