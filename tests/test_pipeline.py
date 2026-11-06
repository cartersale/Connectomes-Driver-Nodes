"""The pipeline definition in scripts/reproduce.py matches the stage scripts it runs."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from hoc.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import reproduce  # noqa: E402

STEPS = reproduce.STEPS


def declared_requires(step: reproduce.Step) -> set[str]:
    """Stages a script checks with require_stage / start_stage / open_run, read from its source."""
    text = (REPO_ROOT / step.script).read_text()
    found = set(re.findall(r'require_stage\(\s*"([^"]+)"', text))
    found |= {f"{step.group}/{s}" for s in re.findall(r'require_stage\(\s*STAGE_PREFIX \+ "([^"]+)"', text)}
    found |= {f"{step.group}/{s}" for s in re.findall(r'C\.stage\("([^"]+)"\)', text)}
    for block in re.findall(r"start_stage\(\s*__file__,\s*\[(.*?)\]", text, flags=re.S):
        found |= set(re.findall(r'(?<![\w.(])"([a-z_]+/[0-9a-z_]+)"', block))
    for block in re.findall(r"open_run\(\s*\"[^\"]+\",\s*\((.*?)\)\)", text, flags=re.S):
        found |= set(re.findall(r'"([^"]+)"', block))
    for block in re.findall(r"^REQUIRES = \((.*?)\)$", text, flags=re.S | re.M):
        found |= set(re.findall(r'"([^"]+)"', block))
    return found


def test_graph_is_valid() -> None:
    order = reproduce.check_graph()
    assert [s.stage for s in order] == [s.stage for s in STEPS]


@pytest.mark.parametrize("step", STEPS, ids=lambda s: s.stage)
def test_requires_match_the_script(step: reproduce.Step) -> None:
    assert declared_requires(step) == set(step.requires)


@pytest.mark.parametrize("step", STEPS, ids=lambda s: s.stage)
def test_script_declares_its_stage(step: reproduce.Step) -> None:
    text = (REPO_ROOT / step.script).read_text()
    name = step.stage.split("/", 1)[1]
    assert (f'"{step.stage}"' in text or f'STAGE_PREFIX + "{name}"' in text or re.search(r"start_stage\(\s*__file__", text)
            or "STAGE_PREFIX + Path(__file__).stem" in text), f"{step.script} does not name stage {step.stage}"


def test_smoke_profile_has_no_gaps() -> None:
    """A smoke step never needs a step that is left out of the smoke run."""
    smoke = {s.stage for s in STEPS if s.smoke}
    for s in STEPS:
        if s.smoke:
            assert set(s.requires) <= smoke, s.stage


def test_imports_resolve() -> None:
    """Every name a script imports from hoc or from the paper modules exists (catches removed re-exports)."""
    import ast
    import importlib

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper"))
    local = ("hoc", "common", "sections", "registry")
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "scripts", "src"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.split()
    missing = []
    for f in files:
        if not f.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((REPO_ROOT / f).read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] in local:
                module = importlib.import_module(node.module)
                for alias in node.names:
                    if not hasattr(module, alias.name):
                        try:
                            importlib.import_module(f"{node.module}.{alias.name}")
                        except ImportError:
                            missing.append(f"{f}: {node.module}.{alias.name}")
    assert not missing


def _output_owners() -> dict[str, set[str]]:
    """File name -> stages that declare it as an output (globs left out)."""
    owners: dict[str, set[str]] = {}
    for s in STEPS:
        for pattern in s.outputs:
            name = pattern.rsplit("/", 1)[-1]
            if not any(ch in name for ch in "*?["):
                owners.setdefault(name, set()).add(s.stage)
    return owners


@pytest.mark.parametrize("step", [s for s in STEPS if s.group in ("paper", "figures")], ids=lambda s: s.stage)
def test_reads_come_from_declared_upstream(step: reproduce.Step) -> None:
    """Every declared output named in a paper or figure step's code (its script and the local modules it imports)
    is produced by a stage upstream of it, so --only and the staleness check see the dependency."""
    import ast

    from hoc.runs import code_record

    owners = _output_owners()
    upstream = reproduce.with_upstream({step.stage})
    missing = []
    for rel in code_record(REPO_ROOT / step.script)["files"]:
        if not rel.startswith("scripts/"):
            continue
        for node in ast.walk(ast.parse((REPO_ROOT / rel).read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value.rsplit("/", 1)[-1]
                if name in owners and not owners[name] & upstream:
                    missing.append(f"{rel}: {node.value!r} comes from {sorted(owners[name])}")
    assert not missing, "\n".join(sorted(set(missing)))


def test_status_sees_changes_to_imported_modules() -> None:
    from hoc.runs import code_record

    step = reproduce.BY_STAGE["paper/tables"]
    code = code_record(REPO_ROOT / step.script)
    rec = {"status": "complete", "finished": "9999", "code": code}
    assert reproduce.status({"stages": {step.stage: rec}}, step, "cfg") == ("done", "")
    changed = {**code["files"], "scripts/paper/condense.py": "0" * 64}
    rec = {**rec, "code": {"sha256": "x", "files": changed}}
    state, why = reproduce.status({"stages": {step.stage: rec}}, step, "cfg")
    assert state == "stale" and "scripts/paper/condense.py" in why
    assert "src/hoc/runs.py" in code["files"] and "scripts/paper/sections/s7.py" in code["files"]
