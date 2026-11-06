"""Run directories: isolation, manifests and overwrite protection."""

from __future__ import annotations

import json

import pytest

from hoc import runs
from hoc.paths import REPO_ROOT


@pytest.fixture()
def runs_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv(runs.RUNS_ROOT_ENV, str(tmp_path / "runs"))
    monkeypatch.delenv(runs.RUN_ENV, raising=False)
    config = tmp_path / "cfg.toml"
    config.write_text('title = "t"\n[analysis]\nrandom_seed = 1\n')
    return tmp_path, config


def _ctx(config, stage="primary/00_prepare_data", **kwargs):
    return runs.RunContext(stage=stage, config_path=config, argv=["test"], **kwargs)


def test_subject_limit_gets_its_own_directory(runs_tmp):
    tmp, config = runs_tmp
    full = _ctx(config, run="canon")
    smoke = _ctx(config, run="canon", subject_limit=2)
    assert full.root == tmp / "runs" / "canon"
    assert smoke.root == tmp / "runs" / "canon__n2"
    assert full.root != smoke.root


def test_default_name_and_env(runs_tmp, monkeypatch):
    tmp, config = runs_tmp
    assert _ctx(config).root == tmp / "runs" / runs.DEFAULT_RUN
    monkeypatch.setenv(runs.RUN_ENV, "fromenv")
    assert _ctx(config).root == tmp / "runs" / "fromenv"
    assert _ctx(config, run="explicit").root == tmp / "runs" / "explicit"


def test_variant_directory(runs_tmp):
    tmp, config = runs_tmp
    ctx = _ctx(config, run="canon", variant="fraction_5pct")
    assert ctx.root == tmp / "runs" / "canon" / "variants" / "fraction_5pct"


def test_variant_from_config(runs_tmp):
    tmp, config = runs_tmp
    config.write_text('[run]\nvariant = "fraction_5pct"\n')
    assert _ctx(config, run="canon").root.name == "fraction_5pct"




def test_manifest_records_stage(runs_tmp):
    _, config = runs_tmp
    ctx = _ctx(config, run="canon").begin(["sub-000", "sub-001"], hash_inputs=False)
    out = ctx.path("primary", "x.csv")
    out.write_text("a\n1\n")
    ctx.finish()
    manifest = json.loads((ctx.root / "manifest.json").read_text())
    stage = manifest["stages"]["primary/00_prepare_data"]
    assert stage["status"] == "complete"
    assert "primary/x.csv" in stage["outputs"]
    assert manifest["subjects"] == ["sub-000", "sub-001"]
    from hoc.config import config_digest

    assert manifest["config"]["sha256"] == config_digest(config)
    assert "numpy" in stage["environment"]["packages"]
    assert (ctx.root / "config" / config.name).read_text() == config.read_text()


def test_finished_stage_needs_overwrite(runs_tmp):
    _, config = runs_tmp
    _ctx(config, run="canon").begin(["sub-000"], hash_inputs=False).finish()
    with pytest.raises(runs.RunConflictError, match="already finished"):
        _ctx(config, run="canon").begin(["sub-000"], hash_inputs=False)
    _ctx(config, run="canon", overwrite=True).begin(["sub-000"], hash_inputs=False).finish()


def test_different_config_or_subjects_refused(runs_tmp):
    tmp, config = runs_tmp
    _ctx(config, run="canon").begin(["sub-000", "sub-001"], hash_inputs=False).finish()
    other = tmp / "other.toml"
    other.write_text('title = "different"\n')
    with pytest.raises(runs.RunConflictError, match="config differs"):
        _ctx(other, stage="primary/01_primary_analysis", run="canon").begin(["sub-000", "sub-001"], hash_inputs=False)
    with pytest.raises(runs.RunConflictError, match="subject set differs"):
        _ctx(config, stage="primary/01_primary_analysis", run="canon").begin(["sub-000"], hash_inputs=False)


def test_unfinished_stage_can_resume(runs_tmp):
    _, config = runs_tmp
    _ctx(config, run="canon").begin(["sub-000"], hash_inputs=False)  # crashed: never finished
    _ctx(config, run="canon").begin(["sub-000"], hash_inputs=False).finish()


def test_require_stage(runs_tmp):
    _, config = runs_tmp
    ctx = _ctx(config, stage="primary/01_primary_analysis", run="canon")
    with pytest.raises(runs.RunConflictError, match="has not finished"):
        ctx.require_stage("primary/00_prepare_data")
    _ctx(config, run="canon").begin(["sub-000"], hash_inputs=False).finish()
    assert ctx.require_stage("primary/00_prepare_data")["status"] == "complete"


def test_input_hashes(runs_tmp):
    tmp, config = runs_tmp
    raw = tmp / "raw.mat"
    raw.write_bytes(b"connectomes")
    atlas = tmp / "atlas"
    atlas.mkdir()
    (atlas / "a.annot").write_bytes(b"labels")
    config.write_text(f'[paths]\nraw_connectomes = "{raw}"\natlas_dir = "{atlas}"\n')
    ctx = _ctx(config, run="canon").begin(["sub-000"])
    inputs = runs.read_manifest(ctx.root)["inputs"]
    assert {v["sha256"] for v in inputs.values()} == {runs.sha256_file(raw), runs.sha256_file(atlas / "a.annot")}


def test_reader_run_dir(runs_tmp, monkeypatch):
    tmp, _ = runs_tmp
    with pytest.raises(FileNotFoundError):
        runs.reader_run_dir(["--run", "missing"])
    (tmp / "runs" / "canon").mkdir(parents=True)
    assert runs.reader_run_dir(["--run", "canon"]) == tmp / "runs" / "canon"
    assert runs.figures_dir(tmp / "runs" / "canon", "section_x") == tmp / "runs" / "canon" / "figures" / "section_x"


def test_stage_from_argv(runs_tmp):
    tmp, config = runs_tmp
    ctx = runs.stage_from_argv("group/01_x", config_path=config, argv=["--run", "canon", "--overwrite", "--other", "1"])
    assert ctx.root == tmp / "runs" / "canon"
    assert ctx.overwrite


def test_partial_stage_names():
    assert runs.partial_stage("s", scales=([68, 114, 219], [219, 114, 68])) == "s"
    assert runs.partial_stage("s", scales=([68], [68, 114, 219])) == "s:partial[scales=68]"
    assert runs.partial_stage("s", subjects=(5, 70), nulls=(["real"], ["real", "weights"])) == "s:partial[subjects=5;nulls=real]"


def test_stage_record_has_script_seed_and_attempts(runs_tmp):
    _, config = runs_tmp
    config.write_text("[analysis]\nrandom_seed = 7\n[seeds]\nstage_x = 3\n")
    ctx = runs.RunContext(stage="group/01_x", config_path=config, run="canon", argv=[__file__]).begin(hash_inputs=False)
    record = runs.read_manifest(ctx.root)["stages"]["group/01_x"]
    assert record["script"]["sha256"] == runs.sha256_file(__file__)
    assert record["random_seed"] == 7 and record["stage_seeds"] == {"stage_x": 3}
    assert len(record["attempts"]) == 1 and "git" in record["attempts"][0]
    ctx2 = runs.RunContext(stage="group/01_x", config_path=config, run="canon", argv=[__file__]).begin(hash_inputs=False)
    assert len(runs.read_manifest(ctx2.root)["stages"]["group/01_x"]["attempts"]) == 2  # a resume keeps the history


def test_active_config_path(monkeypatch, tmp_path):
    from hoc import config

    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    assert config.active_config_path([]) == REPO_ROOT / config.DEFAULT_CONFIG
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "a.toml"))
    assert config.active_config_path([]) == tmp_path / "a.toml"
    assert config.active_config_path(["--config", "config/smoke.toml"]) == REPO_ROOT / "config" / "smoke.toml"


def test_config_digest_ignores_comments(tmp_path):
    from hoc.config import config_digest

    a, b, c = tmp_path / "a.toml", tmp_path / "b.toml", tmp_path / "c.toml"
    a.write_text("[x]\nn = 1\n")
    b.write_text("# a comment\n[x]\nn = 1  # same value\n")
    c.write_text("[x]\nn = 2\n")
    assert config_digest(a) == config_digest(b) != config_digest(c)
