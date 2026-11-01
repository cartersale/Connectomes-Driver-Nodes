#!/usr/bin/env python
"""Run every analysis behind the paper, in dependency order, into one run directory.

    python scripts/reproduce.py --run paper               # everything: analyses, figures, tables, numbers
    python scripts/reproduce.py --smoke                   # the same pipeline on synthetic data (about a minute)
    python scripts/reproduce.py --run paper --list        # every step and its status
    python scripts/reproduce.py --run paper --dry-run     # the commands that would run
    python scripts/reproduce.py --run paper --only figures paper     # these groups (and anything they need)
    python scripts/reproduce.py --check-inputs            # verify the external data against reference/inputs.sha256

A step is skipped when the run's manifest records it as complete and it is still current: same script, same
config, and finished after every step it depends on. A step that is complete but no longer current is *stale*;
the runner stops and names it rather than silently mixing old and new outputs. Rerun it (and everything
downstream) with ``--rerun-stale``, or rerun one step with ``--force STAGE``.

``--reuse STAGE=RUN`` takes the outputs of STAGE from another run instead of recomputing them. Every file's
sha256 is checked against the other run's manifest, the input data must have the same checksums, and the
reuse is recorded in this run's manifest and in the validation report. It is meant for the null-network step,
which takes several days.

After each step the runner checks that the stage marked itself complete and that its expected outputs exist
and are non-empty (CSV files must parse). Logs go to runs/<name>/logs/. BLAS threads are pinned to 1, which
the results need to be bit-reproducible (src/hoc/threads.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hoc  # noqa: F401 - sets BLAS thread defaults inherited by every step
from hoc.config import CONFIG_ENV, DEFAULT_CONFIG, config_digest
from hoc.paths import REPO_ROOT
from hoc.runs import MANIFEST_NAME, read_manifest, resolve_run_dir, sha256_file

SMOKE_CONFIG = "config/smoke.toml"
INPUTS_REFERENCE = REPO_ROOT / "reference" / "inputs.sha256"


@dataclass(frozen=True)
class Step:
    stage: str  # manifest stage id, "<group>/<name>"
    script: str  # repo-relative path
    requires: tuple[str, ...] = ()  # stages that must be complete first (the script checks these too)
    outputs: tuple[str, ...] = ()  # run-relative paths or globs that must exist and be non-empty afterwards
    args: tuple[str, ...] = ()
    smoke: bool = True  # also part of the synthetic smoke run

    @property
    def group(self) -> str:
        return self.stage.split("/")[0]


A = "scripts/analysis"
P0, P1, P2 = "primary/00_prepare_data", "primary/01_primary_analysis", "primary/02_robustness"


# Steps outside the smoke run: they need external atlas data that the synthetic dataset does not provide.
NEEDS_ATLAS_DATA = {"target_states"}  # Yeo (2011) networks and neuromaps gradient maps
SURFACE_FIGURES = {"fig1", "fig3", "fig4", "fig5"}  # brain surfaces need the Cammoun annotation files


def analysis(group: str, name: str, requires=(), outputs=(), **kw) -> Step:
    kw.setdefault("smoke", group not in NEEDS_ATLAS_DATA)
    return Step(f"{group}/{name}", f"{A}/{group}/{name}.py", tuple(requires), tuple(f"{group}/{o}" for o in outputs), **kw)


def figure(name: str, requires) -> Step:
    return Step(f"figures/{name}", f"scripts/figures/{name}.py", tuple(requires), (f"figures/{name}.pdf",),
                smoke=name not in SURFACE_FIGURES)


STEPS: list[Step] = [
    # Connectome preprocessing, node rankings, driver sets and the primary energy/geometry analysis
    Step(P0, f"{A}/primary/00_prepare_data.py", (), ("processed/connectomes_scale-*.npz", "metadata/*.csv")),
    Step(P1, f"{A}/primary/01_primary_analysis.py", (P0,),
         ("primary/node_scores.csv", "primary/driver_sets.csv", "primary/landscape_samples*", "statistics/*.csv")),
    Step(P2, f"{A}/primary/02_robustness.py", (P0,), ("robustness/sensitivity_summary.csv",), ("--skip-perturbations",)),
    # Gramian spectra of driver sets against random and matched sets
    analysis("spectra", "01_compute_spectra", (P0, P1), ("set_metrics.csv", "centroids_scale-*.csv", "spectra_scale-*.npz")),
    analysis("spectra", "02_analyze", ("spectra/01_compute_spectra",),
             ("tables/t0_cycle_minus_degree.csv", "tables/t2_eigenvalue_extremes.csv", "tables/t6_condition_number_check.csv")),
    # Damping, horizon, regularization and driver-set size
    analysis("sensitivity", "00_validate", (P0, P1, P2), ("validation.json",)),
    analysis("sensitivity", "01_csweep_perron", (P0, P1), ("csweep_sets_methods.csv", "csweep_spearman.csv")),
    analysis("sensitivity", "02_nctpy_equivalence", (P0, P1), ("nctpy_cohort_summary.csv", "nctpy_equivalence_detail.csv")),
    analysis("sensitivity", "03_horizon_eps_largek", (P0, P1),
             ("horizon_eps_largek_tidy.csv.gz", "epsilon_summary.csv", "finite_horizon_summary.csv", "large_k_summary.csv")),
    analysis("sensitivity", "04_energy_floor", (P0, P1),
             ("energy_floor_driver_sets.csv", "energy_floor_landscape.csv", "energy_floor_delta_log10E_summary.csv")),
    analysis("sensitivity", "05_summarize_csweep", ("sensitivity/01_csweep_perron",),
             ("csweep_cycle_minus_degree.csv", "csweep_vs_random.csv", "csweep_signflip.csv")),
    analysis("sensitivity", "06_perron_diagnostic", (P0, P1), ("perron_by_method.csv", "perron_diagnostic.csv")),
    analysis("sensitivity", "07_report_tables",
             tuple(f"sensitivity/{s}" for s in ("01_csweep_perron", "02_nctpy_equivalence", "03_horizon_eps_largek",
                                                "04_energy_floor", "05_summarize_csweep", "06_perron_diagnostic")),
             ("csweep_per_k.csv", "csweep_signflip_per_k.csv")),
    # Set-level concentration against node-level modal and average controllability
    analysis("modal_controllability", "01_compute_modal", (P0, P1), ("set_level_modal.csv.gz", "node_level_modal.csv.gz")),
    analysis("modal_controllability", "02_analyze_modal", (P1, "modal_controllability/01_compute_modal"),
             ("within_subject_summary.csv", "node_level_correlations.csv", "set_overlap.csv")),
    # Null networks (strength-, geometry- and topology-preserving)
    analysis("null_models", "00_validate", (P0, "spectra/01_compute_spectra"), ("validation.json",)),
    analysis("null_models", "01_run_nulls", (P0, "spectra/01_compute_spectra"), ("chunks/*/scale-*/*_sets.csv.gz",)),
    analysis("null_models", "02_analyze", ("null_models/01_run_nulls",),
             ("inference.csv", "network_inference.csv", "null_fidelity.csv", "sets.csv.gz")),
    analysis("null_models", "03_gap_mediators", (P0, "null_models/01_run_nulls", "null_models/02_analyze"),
             ("gap_split.csv", "mediation_noncircular.csv", "mediator_contrasts.csv")),
    analysis("null_models", "04_rayleigh_bound", (P0, "null_models/01_run_nulls"), ("rayleigh/per_subject.csv",)),
    # Leading-mode share q1 of the Gramian
    analysis("concentration", "01_compute_q1", (P0, P1, "modal_controllability/01_compute_modal", "spectra/01_compute_spectra"),
             ("set_q1.csv.gz",)),
    analysis("concentration", "02_tables", ("concentration/01_compute_q1",), ("q1_vs_random.csv", "q1_matched.csv", "q1_perp.csv")),
    analysis("concentration", "03_decay_collapse", (P0, P1),
             ("decay_curves.csv.gz", "decay_gap_correlations.csv", "decay_gap_partial_correlations.csv", "fig_decay_collapse.png")),
    analysis("concentration", "04_redundancy_geodesic", (P0, "concentration/01_compute_q1"), ("redundancy_summary.csv",)),
    analysis("concentration", "06_er_avgctrl", ("concentration/01_compute_q1",), ("er_avgctrl.csv",)),
    # Why hub sets load the leading mode; anatomy of v1, v2, v3; model networks; core cuts
    analysis("mechanisms", "01_hub_loading", (P0, "null_models/01_run_nulls"), ("hub_loading.csv.gz", "hub_loading_summary.csv")),
    analysis("mechanisms", "02_canonical_graphs", (P0,), ("canonical_graphs.csv.gz", "canonical_summary.csv")),
    analysis("mechanisms", "03_coverage_distance", (P0, "concentration/01_compute_q1"), ("coverage_by_criterion.csv", "coverage_rho.csv")),
    analysis("mechanisms", "04_leading_eigenvectors", (P0, P1, "spectra/01_compute_spectra"), ("eigvec_subjects.csv", "eigvec_regions.csv")),
    analysis("mechanisms", "05_second_subspace", (P0, P1, "spectra/01_compute_spectra"),
             ("second_subspace_subjects.csv", "second_subspace_axes.csv")),
    # not in the smoke run: on synthetic connectomes the matched control cuts within the core halves are empty
    analysis("mechanisms", "07_core_cut", (P0, P1, "spectra/01_compute_spectra"), ("core_cut.csv.gz", "core_cut_regions.csv"),
             smoke=False),
    analysis("mechanisms", "08_gramian_alignment", (P0, P1), ("gramian_alignment.csv", "node_v1_strength.csv")),
    # Target states (named, single-parcel, gradient) and their spin-test nulls
    analysis("target_states", "01_build_targets", (P0,), ("target_catalog.csv", "target_mask_parcels.csv", "targets_scale-*.npz")),
    analysis("target_states", "02_target_energies", (P0, P1, "target_states/01_build_targets"), ("target_energies_long.csv.gz",)),
    analysis("target_states", "03_target_stats", ("target_states/02_target_energies",),
             ("named_target_summary.csv", "single_parcel_summary.csv", "driver_target_overlap.csv")),
    analysis("target_states", "04_sensitivity", (P0, P1, "target_states/01_build_targets"), ("sensitivity_summary.csv",)),
    analysis("target_states", "05_spin_tests", (P0, P1), ("spin_tests.csv",)),
    analysis("target_states", "06_network_enrichment", (P0, "target_states/01_build_targets", "target_states/05_spin_tests"),
             ("rank_divergence_network_enrichment.csv", "top15_network_composition.csv")),
    analysis("target_states", "07_proximity_controls", ("target_states/02_target_energies",), ("proximity_controls.csv",)),
    analysis("target_states", "09_report_tables",
             tuple(f"target_states/{s}" for s in ("01_build_targets", "03_target_stats", "04_sensitivity",
                                                  "05_spin_tests", "06_network_enrichment", "07_proximity_controls")),
             ("*.md",)),
    # Reciprocal hub lesions
    analysis("lesions", "01_run_reciprocal_lesion", (P0,), ("lesion_rows.csv.gz",)),
    analysis("lesions", "02_analyze_reciprocal_lesion", ("lesions/01_run_reciprocal_lesion",), ("summary_3x2.csv", "crossover_stats.csv")),
    # Stability of cycle participation across representatives, scales and edge perturbations
    analysis("topology", "run_representatives_all_scales", (P0,), ("representatives/per_subject_scale-*.csv",)),
    analysis("topology", "run_perturbation_stability", (P0,), ("perturbation/per_draw_scale-*.csv",)),
    analysis("topology", "summarize_representatives", ("topology/run_representatives_all_scales",), ("tables/rep_agreement.csv",)),
    analysis("topology", "summarize_perturbation", ("topology/run_perturbation_stability", "topology/run_representatives_all_scales"),
             ("tables/perturb_rank_stability.csv",)),
    # Main-text figures
    figure("fig1", (P0,)),
    figure("fig2", ("concentration/01_compute_q1", "mechanisms/08_gramian_alignment")),
    figure("fig3", ("mechanisms/04_leading_eigenvectors", "mechanisms/08_gramian_alignment")),
    figure("fig3b", ("mechanisms/08_gramian_alignment",)),
    figure("fig4", ("mechanisms/05_second_subspace", "concentration/02_tables", "concentration/03_decay_collapse")),
    figure("fig5", ("null_models/02_analyze", "null_models/04_rayleigh_bound", "mechanisms/07_core_cut")),
]
STEPS += [
    # Supplementary tables, the number registry, validation, and the assembled paper folder
    Step("paper/tables", "scripts/paper/tables.py",
         ("concentration/02_tables", "concentration/03_decay_collapse", "concentration/04_redundancy_geodesic",
          "null_models/03_gap_mediators", "null_models/04_rayleigh_bound", "target_states/09_report_tables",
          "lesions/02_analyze_reciprocal_lesion", "topology/summarize_perturbation", "topology/summarize_representatives",
          "mechanisms/02_canonical_graphs", "mechanisms/03_coverage_distance", "sensitivity/00_validate",
          "sensitivity/07_report_tables", "spectra/02_analyze", "modal_controllability/02_analyze_modal"),
         ("paper/tables/*.tex",), smoke=False),
    Step("paper/numbers", "scripts/paper/key_numbers.py",
         ("paper/tables", "mechanisms/04_leading_eigenvectors", "mechanisms/05_second_subspace", "mechanisms/07_core_cut",
          "mechanisms/08_gramian_alignment", "concentration/06_er_avgctrl", "null_models/02_analyze",
          "null_models/00_validate", P1),
         ("paper/key_results.json", "paper/numbers.csv", "paper/numbers.tex"), smoke=False),
    Step("paper/validate", "scripts/paper/validate.py", ("paper/numbers",), ("paper/validation_report.md",), smoke=False),
    Step("paper/assemble", "scripts/paper/assemble.py", ("paper/validate",), ("paper/provenance.json", "paper/figures/fig1.pdf"),
         smoke=False),
]
BY_STAGE = {s.stage: s for s in STEPS}


# --------------------------------------------------------------------------- graph


def check_graph() -> list[Step]:
    """Steps in run order; fails on missing scripts, unknown stages, or a step listed before what it needs."""
    problems, seen = [], set()
    for step in STEPS:
        if not (REPO_ROOT / step.script).is_file():
            problems.append(f"{step.stage}: script {step.script} does not exist")
        for r in step.requires:
            if r not in BY_STAGE:
                problems.append(f"{step.stage}: requires unknown stage {r}")
            elif r not in seen:
                problems.append(f"{step.stage}: listed before {r}, which it requires")
        seen.add(step.stage)
    if len(seen) != len(STEPS):
        problems.append("duplicate stage names")
    if problems:
        sys.exit("pipeline definition is broken:\n  " + "\n  ".join(problems))
    return list(STEPS)


def with_upstream(names: set[str]) -> set[str]:
    out, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name not in out:
            out.add(name)
            todo.extend(BY_STAGE[name].requires)
    return out


def downstream(name: str) -> set[str]:
    out = {name}
    changed = True
    while changed:
        changed = False
        for s in STEPS:
            if s.stage not in out and any(r in out for r in s.requires):
                out.add(s.stage)
                changed = True
    return out


# --------------------------------------------------------------------------- status


def stage_record(manifest: dict, step: Step) -> dict:
    return manifest.get("stages", {}).get(step.stage, {})


def status(manifest: dict, step: Step, config_sha: str) -> tuple[str, str]:
    """('done' | 'reused' | 'stale' | 'running' | 'pending', reason)."""
    rec = stage_record(manifest, step)
    if not rec:
        return "pending", ""
    if rec.get("status") != "complete":
        return rec.get("status", "pending"), "interrupted; will resume"
    reasons = []
    script = rec.get("script") or {}
    current = sha256_file(REPO_ROOT / step.script)
    if script.get("sha256") != current:
        reasons.append("script changed" if script else "no script checksum recorded")
    if manifest.get("config", {}).get("sha256") not in (None, config_sha):
        reasons.append("config changed")
    finished = rec.get("finished", "")
    for r in step.requires:
        up = stage_record(manifest, BY_STAGE[r])
        if up.get("status") == "complete" and up.get("finished", "") > finished:
            reasons.append(f"{r} finished later")
    if reasons:
        return "stale", "; ".join(reasons)
    return ("reused" if rec.get("reused_from") else "done"), ""


# --------------------------------------------------------------------------- checks


def check_outputs(run_dir: Path, step: Step) -> list[str]:
    problems = []
    rec = stage_record(read_manifest(run_dir) or {}, step)
    if rec.get("status") != "complete":
        problems.append("stage did not mark itself complete in the manifest")
    for pattern in step.outputs:
        matches = sorted(run_dir.glob(pattern))
        if not matches:
            problems.append(f"missing output {pattern}")
            continue
        for path in matches:
            if path.is_file() and path.stat().st_size == 0:
                problems.append(f"empty output {path.relative_to(run_dir)}")
            elif path.suffix == ".csv" or path.name.endswith(".csv.gz"):
                try:
                    import pandas as pd

                    pd.read_csv(path, nrows=5)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"unreadable CSV {path.relative_to(run_dir)}: {exc}")
    return problems


def check_inputs() -> int:
    """Verify every external input file against reference/inputs.sha256.

    Paths are relative to the repository, except ``$NEUROMAPS_DATA/...`` (maps that neuromaps downloads; default
    ``~/neuromaps-data``).
    """
    neuromaps = Path(os.environ.get("NEUROMAPS_DATA", Path.home() / "neuromaps-data"))
    bad = total = 0
    for line in INPUTS_REFERENCE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        total += 1
        path = neuromaps / name.split("/", 1)[1] if name.startswith("$NEUROMAPS_DATA/") else REPO_ROOT / name
        if not path.is_file():
            print(f"MISSING  {name}")
            bad += 1
        elif sha256_file(path) != digest:
            print(f"DIFFERS  {name}")
            bad += 1
    print(f"{total - bad}/{total} input files match {INPUTS_REFERENCE.relative_to(REPO_ROOT)}")
    return bad


# --------------------------------------------------------------------------- reuse


def reuse(step: Step, source_run: str, run_dir: Path, config_path: Path) -> None:
    """Copy STAGE's outputs from another run after verifying every checksum; record it in the manifest."""
    src_dir = resolve_run_dir(source_run)
    src = read_manifest(src_dir) or {}
    rec = src.get("stages", {}).get(step.stage)
    if not rec or rec.get("status") != "complete":
        sys.exit(f"--reuse: stage {step.stage} is not complete in run {source_run}")
    dst = read_manifest(run_dir) or {}
    src_inputs = {v["sha256"] for v in src.get("inputs", {}).values()}
    dst_inputs = {v["sha256"] for v in dst.get("inputs", {}).values()}
    if not dst_inputs or not dst_inputs <= src_inputs:
        sys.exit(f"--reuse: the input data of this run are not all among run {source_run}'s inputs (by sha256); "
                 f"refusing to reuse {step.stage}")
    outputs = rec.get("outputs", {})
    print(f"reuse {step.stage} from {source_run}: verifying {len(outputs)} files", flush=True)
    for rel, digest in outputs.items():
        path = src_dir / rel
        if not path.is_file() or sha256_file(path) != digest:
            sys.exit(f"--reuse: {rel} in run {source_run} is missing or does not match its recorded checksum")
    for rel in outputs:
        target = run_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / rel, target)
    src_cfg = (src.get("config") or {}).get("sha256")
    cfg_note = None
    if src_cfg and src_cfg not in (sha256_file(config_path), config_digest(config_path)):
        snap = src_dir / "config" / Path((src.get("config") or {}).get("path") or "default.toml").name
        if snap.exists():
            cfg_note = "".join(difflib.unified_diff(snap.read_text().splitlines(True), config_path.read_text().splitlines(True),
                                                    "source run config", "this run config"))
        else:
            cfg_note = f"source config sha256 {src_cfg} differs; no snapshot to diff"
    manifest = read_manifest(run_dir)
    manifest["stages"][step.stage] = {
        **{k: v for k, v in rec.items() if k not in ("reused_from",)},
        "script": {"path": step.script, "sha256": sha256_file(REPO_ROOT / step.script)},
        "finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "reused_from": {"run": source_run, "stage_record_finished": rec.get("finished"), "files": len(outputs),
                        "source_script": rec.get("script"), "config_diff": cfg_note},
    }
    tmp = run_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(run_dir / MANIFEST_NAME)


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="Run name under runs/ (default: 'paper', or 'smoke' with --smoke).")
    p.add_argument("--config", default=None, help=f"Config file (default: {DEFAULT_CONFIG}).")
    p.add_argument("--smoke", action="store_true", help=f"Synthetic data ({SMOKE_CONFIG}); only steps marked for it.")
    p.add_argument("--only", nargs="+", default=None, help="Groups or stages to run, with everything they need.")
    p.add_argument("--from", dest="start", default=None, help="Skip steps before this stage (in run order).")
    p.add_argument("--list", action="store_true", help="List the steps and their status, then exit.")
    p.add_argument("--dry-run", action="store_true", help="Print what would run, without running it.")
    p.add_argument("--force", nargs="+", default=[], metavar="STAGE", help="Rerun these stages even if done.")
    p.add_argument("--rerun-stale", action="store_true", help="Rerun stale steps and everything downstream of them.")
    p.add_argument("--reuse", nargs="+", default=[], metavar="STAGE=RUN", help="Take STAGE's outputs from RUN (checksummed).")
    p.add_argument("--check-inputs", action="store_true", help="Check input files against reference/inputs.sha256 and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(args.config or (SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG))
    config = config if config.is_absolute() else REPO_ROOT / config
    run = args.run or ("smoke" if args.smoke else "paper")
    if args.check_inputs:
        sys.exit(1 if check_inputs() else 0)

    order = check_graph()
    if args.only:
        wanted = {s.stage for s in STEPS if s.group in args.only or s.stage in args.only}
        unknown = [o for o in args.only if o not in {s.group for s in STEPS} | set(BY_STAGE)]
        if unknown:
            sys.exit(f"unknown group or stage {unknown}; see --list")
        selected = with_upstream(wanted)
        order = [s for s in order if s.stage in selected]
    if args.start:
        if args.start not in BY_STAGE:
            sys.exit(f"unknown stage {args.start!r}; see --list")
        names = [s.stage for s in order]
        order = order[names.index(args.start):] if args.start in names else order
    skipped_smoke = [s for s in order if args.smoke and not s.smoke]
    order = [s for s in order if not (args.smoke and not s.smoke)]

    run_dir = resolve_run_dir(run)
    manifest = read_manifest(run_dir) or {}
    config_sha = config_digest(config)
    reuse_map = dict(r.split("=", 1) for r in args.reuse)
    for stage in [*reuse_map, *args.force]:
        if stage not in BY_STAGE:
            sys.exit(f"unknown stage {stage!r}; see --list")

    states = {s.stage: status(manifest, s, config_sha) for s in order}
    if args.list or args.dry_run:
        print(f"run {run} ({run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir}), "
              f"config {config.relative_to(REPO_ROOT)}")
        for s in order:
            state, why = states[s.stage]
            line = f"{state:<8} {s.stage}" + (f"   ({why})" if why else "")
            if args.dry_run and state not in ("done", "reused"):
                how = f"reuse from {reuse_map[s.stage]}" if s.stage in reuse_map else f"python {s.script} {' '.join(s.args)}"
                line += f"\n         {how}"
            print(line)
        for s in skipped_smoke:
            print(f"{'-':<8} {s.stage}   (not part of the smoke run)")
        return

    stale = [s.stage for s in order if states[s.stage][0] == "stale" and s.stage not in args.force]
    if stale and not args.rerun_stale:
        sys.exit("These steps finished, but their script, config or inputs changed since:\n  "
                 + "\n  ".join(f"{n}: {states[n][1]}" for n in stale)
                 + "\nRerun them and everything downstream with --rerun-stale, or one step with --force STAGE.")
    rerun = set()
    for name in [*args.force, *(stale if args.rerun_stale else [])]:
        rerun |= downstream(name)

    env = {**os.environ, CONFIG_ENV: str(config), "PYTHONUNBUFFERED": "1"}
    log_dir = run_dir / "logs"
    timings_path = log_dir / "timings.jsonl"
    for step in order:
        state, _ = states[step.stage]
        if state in ("done", "reused") and step.stage not in rerun:
            print(f"skip  {step.stage} ({state})")
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        if step.stage in reuse_map and step.stage not in rerun:
            reuse(step, reuse_map[step.stage], run_dir, config)
            returncode, log = 0, None
        else:
            cmd = [sys.executable, "-u", step.script, *step.args, "--run", run]  # config via $HOC_CONFIG
            if step.stage in rerun or state == "stale":
                cmd.append("--overwrite")
            print(f"run   {step.stage}", flush=True)
            log = log_dir / (step.stage.replace("/", "__") + ".log")
            with log.open("a") as handle:  # a resumed step keeps the log of the interrupted attempt
                handle.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[1:])}\n")
                handle.flush()
                returncode = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - start
        with timings_path.open("a") as handle:
            handle.write(json.dumps({"stage": step.stage, "seconds": round(elapsed, 1), "returncode": returncode,
                                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        if returncode != 0:
            tail = "".join(log.read_text().splitlines(keepends=True)[-25:]) if log else ""
            sys.exit(f"FAILED {step.stage} after {elapsed:.0f}s (log: {log})\n{tail}")
        problems = check_outputs(run_dir, step)
        if problems:
            sys.exit(f"FAILED {step.stage}: " + "; ".join(problems))
        print(f"      done in {elapsed / 60:.1f} min", flush=True)
    if skipped_smoke:
        print(f"{len(skipped_smoke)} steps are not part of the smoke run: " + ", ".join(s.stage for s in skipped_smoke))


if __name__ == "__main__":
    main()
