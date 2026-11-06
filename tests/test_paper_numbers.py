"""The paper numbers and tables of a finished run match the release reference and the manuscript (needs the real data).

    uv run pytest -m data tests/test_paper_numbers.py            # checks runs/paper
    HOC_RUN=<name> uv run pytest -m data tests/test_paper_numbers.py
    HOC_MANUSCRIPT=<folder> HOC_MANUSCRIPT_MAIN=main.tex uv run pytest -m data tests/test_paper_numbers.py

Same checks as the pipeline step paper/validate (scripts/paper/validate.py): every number in
<run>/paper/key_results.json against reference/paper_numbers.json (rtol 1e-9, identical text) and every
table against reference/tables.sha256; with HOC_MANUSCRIPT set (a folder holding the main text and
supp_mat.tex), every number's printed token is found at its locus and equals the computed text.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hoc.paths import REPO_ROOT
from hoc.runs import resolve_run_dir

sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper"))
import validate  # noqa: E402


@pytest.mark.data
def test_paper_numbers_match_reference() -> None:
    run_dir = resolve_run_dir(os.environ.get("HOC_RUN", "paper"))
    if not (run_dir / "paper" / "key_results.json").exists():
        pytest.skip(f"{run_dir}/paper/key_results.json not found; run scripts/reproduce.py first")
    numbers, tables = validate.load_run(run_dir)
    lines = validate.compare_reference(numbers, tables)
    failures = [line for line in lines if "**FAIL**" in line]
    assert not failures, "\n".join(failures[:50])


@pytest.mark.data
def test_paper_numbers_match_manuscript() -> None:
    folder = os.environ.get("HOC_MANUSCRIPT")
    if not folder or not Path(folder).is_dir():
        pytest.skip("HOC_MANUSCRIPT does not name a manuscript folder")
    main_name = os.environ.get("HOC_MANUSCRIPT_MAIN", "main.tex")
    main, supp = Path(folder) / main_name, Path(folder) / "supp_mat.tex"
    if not main.exists() or not supp.exists():
        pytest.skip(f"{main_name} or supp_mat.tex not found in {folder}")
    run_dir = resolve_run_dir(os.environ.get("HOC_RUN", "paper"))
    if not (run_dir / "paper" / "key_results.json").exists():
        pytest.skip(f"{run_dir}/paper/key_results.json not found; run scripts/reproduce.py first")
    numbers, _ = validate.load_run(run_dir)
    check = validate.anchored_check(numbers, main.read_text(), supp.read_text())
    missing = [f"{name} ({token}) at {label}" for name, token, label, _ in check["missing"]]
    assert not missing, "printed tokens not found at their locus: " + "; ".join(missing)
    differ = [f"{k}: computed {v['text']}, printed {v['context']['printed']}" for k, v in numbers.items()
              if v["context"].get("printed") is not None and v["context"]["printed"] != v["text"]]
    assert not differ, "computed numbers differ from the manuscript: " + "; ".join(differ)
