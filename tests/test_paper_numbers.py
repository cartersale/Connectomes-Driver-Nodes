"""The paper numbers and tables of a finished run match the release reference (needs the real data).

    uv run pytest -m data tests/test_paper_numbers.py            # checks runs/paper
    HOC_RUN=<name> uv run pytest -m data tests/test_paper_numbers.py

Same check as the pipeline step paper/validate (scripts/paper/validate.py): every number in
<run>/paper/key_results.json against reference/paper_numbers.json (rtol 1e-9, identical text) and every
table against reference/tables.sha256.
"""

from __future__ import annotations

import os
import sys

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
