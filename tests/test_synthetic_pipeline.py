"""Scientific regression test: the pipeline on synthetic data.

Runs ``scripts/reproduce.py --smoke`` (config/smoke.toml: synthetic connectomes, labels and
gradient; no downloads), i.e. every step that does not need external atlas data, and
compares a fingerprint of every output table with tests/golden/synthetic.json:

- text columns (driver sets, labels, rankings as strings) must match exactly;
- numeric columns are compared through their sum and sum of absolute values
  (rtol 1e-6), which catches any change to a computed value;
- columns set purely by roundoff (no-floor eigenvalue extremes) are left out.

The fingerprints are computed on macOS (arm64, Apple Accelerate), and CI runs this test there: the synthetic
data has many values tied up to roundoff, which other BLAS libraries can rank differently.

If a change to the numbers is intended, regenerate the golden file with
    uv run python tests/test_synthetic_pipeline.py --update
and explain the diff in the PR.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hoc.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import reproduce

GOLDEN = REPO_ROOT / "tests" / "golden" / "synthetic.json"
RUN = "golden"

# Values decided by floating-point roundoff, not by the science.
ROUNDOFF_COLUMNS = {
    ("robustness/sensitivity_summary.csv", "no_floor"): [
        "average_energy",
        "log10_average_energy",
        "condition_number",
        "log10_condition_number",
        "min_eigenvalue",
        "log_min_eigenvalue",
    ],
}
# Counts of eigenvalues at the floor can flip by one when an eigenvalue sits
# within roundoff of it.
LOOSE_COLUMNS = {"eigenvalues_at_floor": 0.02, "fraction_eigenvalues_at_floor": 0.02}
RTOL = 1e-6
# Columns whose synthetic-data values depend on the machine: rank statistics over values tied up to roundoff,
# z statistics with a null standard deviation near zero, overlaps of near-degenerate eigenspaces, and statistics
# of the driver sets at the energy floor. Between Apple-silicon machines their sums differ by up to 1.4e-2
# (relative), and the variance of the non-floor energies by 2.5e-2.
PLATFORM_RTOL = 0.02
PLATFORM_RTOL_BY_COLUMN = {("sensitivity/energy_floor_landscape.csv", "var_log10E_nonfloor"): 0.05}
PLATFORM_COLUMNS = {
    "mechanisms/coverage_rho.csv": ["prho_hop|L", "prho_inv|L", "prho_log|L", "rho_hop", "rho_inv", "rho_log"],
    "modal_controllability/driver_gram_proxy.csv": ["rho_erG_er", "rho_pshare_q1", "rho_q1G_q1raw"],
    "modal_controllability/pooled_spearman.csv": ["rho"],
    "null_models/inference.csv": ["delta_median", "n_z_undefined", "wilcoxon_p", "z_mean", "z_median"],
    "null_models/mediation.csv": ["p_mediator"],
    "primary/eigenspace_overlap.csv": ["overlap"],
    "primary/target_geometry.csv": ["hard_overlap", "spectral_entropy_effective_modes", "spectral_spread_effective_modes"],
    "sensitivity/csweep_spearman.csv": [
        "rho_perronfrac_er",
        "rho_perronload_er",
        "rho_perronload_er_rankings_only",
        "rho_perronload_pr",
    ],
    "sensitivity/energy_floor_driver_sets.csv": ["fraction_at_floor", "log10_energy_nonfloor"],
    "sensitivity/energy_floor_landscape.csv": ["iqr_log10E_nonfloor", "mean_fraction_at_floor", "var_log10E_nonfloor"],
    "sensitivity/perron_diagnostic.csv": [
        "partial_rho_er_perron_given_strength",
        "rho_er_perron",
        "rho_er_strength",
        "rho_perron_strength",
    ],
    "sensitivity/perron_spearman.csv": [
        "median_rho_perronfrac_er",
        "median_rho_rankings_only",
        "median_rho_within_subject_k",
        "q25_rho",
        "q75_rho",
    ],
    "spectra/tables/t4_perron_er_spearman.csv": ["rho_within_mean"],
    "topology/representatives/per_subject_scale-114.csv": ["er_pct_strength_null"],
    "topology/representatives/per_subject_scale-68.csv": ["er_pct_overlap_null", "er_pct_strength_null"],
    "topology/tables/perturb_er_persistence.csv": ["fraction_draws_positive"],
    "topology/tables/rep_distance_regressions.csv": ["spearman_vs_degree_within_spearman"],
    "topology/tables/rep_null_comparisons.csv": ["mean_pct_overlap_null", "mean_pct_strength_null"],
}


def run_pipeline(runs_root: Path) -> Path:
    env = {**os.environ, "HOC_RUNS_ROOT": str(runs_root), "PYTHONWARNINGS": "ignore"}
    cmd = [sys.executable, "scripts/reproduce.py", "--smoke", "--run", RUN]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return runs_root / RUN


def _mask_roundoff(rel: str, frame: pd.DataFrame) -> pd.DataFrame:
    for (path, case), columns in ROUNDOFF_COLUMNS.items():
        if rel == path and "epsilon_case" in frame:
            frame.loc[frame["epsilon_case"] == case, columns] = np.nan
    return frame


def fingerprint(run_dir: Path) -> dict:
    tables = {}
    for path in sorted(run_dir.rglob("*.csv")):
        rel = path.relative_to(run_dir).as_posix()
        if rel.startswith(("metadata/", "config/", "logs/")):
            continue
        frame = _mask_roundoff(rel, pd.read_csv(path))
        numeric = frame.select_dtypes(include=["number", "bool"])
        text = frame.drop(columns=numeric.columns)
        entry = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "text_sha256": hashlib.sha256(text.to_csv(index=False).encode()).hexdigest(),
            "numeric": {},
        }
        for column in numeric.columns:
            values = numeric[column].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            entry["numeric"][column] = {
                "sum": float(finite.sum()),
                "abs_sum": float(np.abs(finite).sum()),
                "n_nonfinite": int((~np.isfinite(values)).sum() - np.isnan(values).sum()),
                "n_nan": int(np.isnan(values).sum()),
            }
        tables[rel] = entry
    return tables


def compare(actual: dict, expected: dict) -> list[str]:
    problems = []
    if set(actual) != set(expected):
        problems.append(f"tables differ: +{sorted(set(actual) - set(expected))} -{sorted(set(expected) - set(actual))}")
    for rel in sorted(set(actual) & set(expected)):
        a, e = actual[rel], expected[rel]
        for key in ("rows", "columns", "text_sha256"):
            if a[key] != e[key]:
                problems.append(f"{rel}: {key} differs")
        for column, ev in e["numeric"].items():
            av = a["numeric"].get(column)
            if av is None:
                continue
            rtol = PLATFORM_RTOL if column in PLATFORM_COLUMNS.get(rel, ()) else LOOSE_COLUMNS.get(column, RTOL)
            rtol = PLATFORM_RTOL_BY_COLUMN.get((rel, column), rtol)
            for stat in ("sum", "abs_sum"):
                if not np.isclose(av[stat], ev[stat], rtol=rtol, atol=1e-10):
                    problems.append(f"{rel}:{column} {stat} {av[stat]!r} != {ev[stat]!r}")
            for stat in ("n_nonfinite", "n_nan"):
                if av[stat] != ev[stat]:
                    problems.append(f"{rel}:{column} {stat} {av[stat]} != {ev[stat]}")
    return problems


@pytest.mark.slow
def test_synthetic_pipeline_matches_golden(tmp_path) -> None:
    run_dir = run_pipeline(tmp_path / "runs")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected = {s.stage for s in reproduce.STEPS if s.smoke}
    statuses = {name: stage["status"] for name, stage in manifest["stages"].items()}
    assert set(statuses) == expected and set(statuses.values()) == {"complete"}
    problems = compare(fingerprint(run_dir), json.loads(GOLDEN.read_text()))
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    import tempfile

    if "--update" not in sys.argv:
        sys.exit("usage: python tests/test_synthetic_pipeline.py --update")
    with tempfile.TemporaryDirectory() as tmp:
        result = fingerprint(run_pipeline(Path(tmp)))
    GOLDEN.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN.relative_to(REPO_ROOT)} ({len(result)} tables)")
