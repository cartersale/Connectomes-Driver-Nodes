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


def test_no_development_paths_in_code() -> None:
    files = subprocess.run(["git", "ls-files", "scripts", "src", "config"], cwd=REPO_ROOT, capture_output=True,
                           text=True).stdout.split()
    pattern = re.compile(r'"revision[/"]|/ "revision"|results-derived|"results" /|REPO_ROOT / "results"|_rescued')
    hits = [f for f in files if f.endswith((".py", ".toml")) and pattern.search((REPO_ROOT / f).read_text())]
    assert not hits


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
