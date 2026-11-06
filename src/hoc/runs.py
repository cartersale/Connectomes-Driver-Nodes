"""Run directories, manifests and overwrite protection.

Every analysis writes into its own run directory:

    runs/<name>/                      main configuration
    runs/<name>/variants/<variant>/   a config with ``[run] variant = "..."``

The name comes from ``--run``, then ``$HOC_RUN``, then ``"main"``. A
``--subject-limit N`` run always gets the suffix ``__nN``, so a smoke run can
never write into a full run's directory.

Each run directory holds ``manifest.json`` with the config hash, subject set,
input checksums and, per stage, the checksums of the stage's code (its script and every local module the script
imports, see :func:`code_record`), random seed, arguments, git state and package versions of every start or
resume, timing and output checksums. A stage refuses to run when the directory was made
with a different config or subject set, or when the stage has already finished,
unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata as _metadata
from pathlib import Path

from .paths import REPO_ROOT
from .threads import THREAD_ENV_VARS, threadpool_record

RUNS_ROOT_ENV = "HOC_RUNS_ROOT"
INPUTS_REFERENCE = REPO_ROOT / "reference" / "inputs.sha256"
NEUROMAPS_PREFIX = "$NEUROMAPS_DATA/"
RUN_ENV = "HOC_RUN"
DEFAULT_RUN = "main"
MANIFEST_NAME = "manifest.json"

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "ripser",
    "persim",
    "bctpy",
    "networkx",
    "neuromaps",
    "netneurotools",
    "nilearn",
    "nibabel",
    "statsmodels",
    "gudhi",
    "joblib",
    "matplotlib",
)


class RunConflictError(RuntimeError):
    """Raised when a stage would overwrite or mix outputs in a run directory."""


# --------------------------------------------------------------------------- paths


def runs_root() -> Path:
    """Directory that holds all run directories (``$HOC_RUNS_ROOT`` or ``runs/``)."""
    root = os.environ.get(RUNS_ROOT_ENV)
    return Path(root).resolve() if root else REPO_ROOT / "runs"


def run_name(run: str | None = None, subject_limit: int | None = None) -> str:
    """Resolve the run name from the argument, ``$HOC_RUN`` or the default."""
    name = run or os.environ.get(RUN_ENV) or DEFAULT_RUN
    if subject_limit is not None:
        suffix = f"__n{int(subject_limit)}"
        if not name.endswith(suffix):
            name = name + suffix
    return name


def resolve_run_dir(
    run: str | None = None,
    *,
    variant: str | None = None,
    subject_limit: int | None = None,
) -> Path:
    """Return the absolute run directory for a run name and optional variant."""
    root = runs_root() / run_name(run, subject_limit)
    return root / "variants" / variant if variant else root


def config_variant(config: dict) -> str | None:
    """Variant name declared by a config (``[run] variant``), if any."""
    value = config.get("run", {}).get("variant")
    return str(value) if value else None


# --------------------------------------------------------------------------- helpers


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict:
    """Commit, branch and whether tracked files differ from the commit."""
    commit = _git("rev-parse", "HEAD")
    dirty_files = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files) if dirty_files is not None else None,
        "dirty_files": dirty_files.splitlines() if dirty_files else [],
    }


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def blas_info() -> dict:
    """Which BLAS/LAPACK numpy is linked against (best effort)."""
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        deps = config.get("Build Dependencies", {})
        return {key: {k: deps[key].get(k) for k in ("name", "version")} for key in ("blas", "lapack") if key in deps}
    except Exception:  # noqa: BLE001 - informational only
        return {}


def environment_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "blas": blas_info(),
        "threads": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "threadpools": threadpool_record(),
    }


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def neuromaps_data_dir() -> Path:
    """Where neuromaps keeps its downloads (``$NEUROMAPS_DATA``, default ``~/neuromaps-data``)."""
    return Path(os.environ.get("NEUROMAPS_DATA", Path.home() / "neuromaps-data"))


def reference_inputs() -> list[tuple[str, str, Path]]:
    """(name, sha256, local path) for every external input listed in ``reference/inputs.sha256``.

    Names are repository-relative, except ``$NEUROMAPS_DATA/...`` for the maps and surfaces neuromaps downloads.
    """
    if not INPUTS_REFERENCE.is_file():
        return []
    entries = []
    for line in INPUTS_REFERENCE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        path = neuromaps_data_dir() / name[len(NEUROMAPS_PREFIX):] if name.startswith(NEUROMAPS_PREFIX) else REPO_ROOT / name
        entries.append((name, digest, path))
    return entries


def collect_inputs(config: dict) -> list[tuple[str, Path]]:
    """(name, path) of the input files a run with this config reads.

    These are the files named by the config's ``[paths]`` block and, for runs on the real data, every external input
    listed in ``reference/inputs.sha256`` that is present (the Yeo networks and the neuromaps maps and surfaces).
    Names are repository-relative, or ``$NEUROMAPS_DATA/...``, so the manifest holds no local paths.
    """
    from . import synthetic

    paths_cfg = config.get("paths", {})
    files: list[Path] = []
    raw = paths_cfg.get("raw_connectomes")
    if raw and not synthetic.is_synthetic(raw):
        files.append(REPO_ROOT / raw if not Path(raw).is_absolute() else Path(raw))
    atlas_dir = paths_cfg.get("atlas_dir")
    if atlas_dir and not synthetic.is_synthetic(atlas_dir):
        atlas = REPO_ROOT / atlas_dir if not Path(atlas_dir).is_absolute() else Path(atlas_dir)
        if atlas.is_dir():
            files.extend(sorted(p for p in atlas.iterdir() if p.is_file() and p.name != ".DS_Store"))
    named = {_relative(p): p for p in files if p.exists()}
    if named:  # real data: add the other reference inputs
        for name, _, path in reference_inputs():
            if path.is_file():
                named.setdefault(name, path)
    return sorted(named.items())


# --------------------------------------------------------------------------- code checksums


def _module_files(name: str, roots: tuple[Path, ...]) -> list[Path]:
    """Files executed by ``import name`` when ``name`` resolves under one of ``roots`` (package __init__ files
    included); empty for modules outside the repository."""
    parts = name.split(".")
    for root in roots:
        if not ((root / parts[0]).is_dir() or (root / f"{parts[0]}.py").is_file()):
            continue
        found = []
        for i in range(1, len(parts) + 1):
            base = root.joinpath(*parts[:i])
            if (base / "__init__.py").is_file():
                found.append(base / "__init__.py")
            elif base.with_suffix(".py").is_file():
                found.append(base.with_suffix(".py"))
                break
            else:
                break
        return found
    return []


def _imports(path: Path, roots: tuple[Path, ...]) -> list[Path]:
    """Local files imported by one Python file.

    Covers ``import a.b``, ``from a import b`` (b may be a submodule), relative imports, and
    ``importlib.import_module(f"pkg.{...}")``, which is taken to mean any module of ``pkg``.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    package = path.parent
    out: list[Path] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out += _module_files(alias.name, roots)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.parent
                names = [f"{node.module}.{a.name}" if node.module else a.name for a in node.names]
                if node.module:
                    out += _module_files(node.module, (base,))
                for n in names:
                    out += _module_files(n, (base,))
            elif node.module:
                out += _module_files(node.module, roots)
                for alias in node.names:
                    out += _module_files(f"{node.module}.{alias.name}", roots)
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
              and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out += _module_files(arg.value, roots)
            elif isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
                prefix = str(arg.values[0].value)
                if prefix.endswith("."):
                    for pkg_init in _module_files(prefix[:-1], roots):
                        if pkg_init.name == "__init__.py":
                            out += sorted(pkg_init.parent.glob("*.py"))
    return out


@lru_cache(maxsize=None)
def _code_files(script: Path) -> tuple[Path, ...]:
    roots = (script.parent, REPO_ROOT / "src")
    seen: set[Path] = set()
    todo = [script]
    while todo:
        path = todo.pop().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        todo += _imports(path, roots)
    return tuple(sorted(seen))


def code_record(script: str | Path) -> dict | None:
    """Checksums of a stage's code: its script and every local module the script imports, transitively.

    Local modules are those under ``src/`` (the ``hoc`` package) or next to the script (for example
    ``scripts/paper/registry/``). ``{"sha256": <digest of all files>, "files": {path: sha256}}``, or None when the
    script is not a Python file.
    """
    path = Path(script)
    path = path if path.is_absolute() else Path.cwd() / path
    if path.suffix != ".py" or not path.is_file():
        return None
    files = {_relative(p): sha256_file(p) for p in _code_files(path.resolve())}
    digest = hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def _script_record(argv: list[str]) -> dict | None:
    """Path and sha256 of the script that runs the stage (argv[0])."""
    if not argv:
        return None
    path = Path(argv[0])
    path = path if path.is_absolute() else Path.cwd() / path
    if not path.is_file():
        return None
    return {"path": _relative(path), "sha256": sha256_file(path)}


# --------------------------------------------------------------------------- manifest


def read_manifest(run_dir: str | Path) -> dict | None:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(run_dir / MANIFEST_NAME)


@dataclass
class RunContext:
    """One stage of an analysis writing into a run directory.

    Typical use::

        ctx = RunContext.from_args("primary/01_primary_analysis", args, config_path=args.config)
        ctx.begin(subject_ids)
        ...  # write outputs under ctx.root
        ctx.finish()
    """

    stage: str
    config_path: Path | None = None
    run: str | None = None
    subject_limit: int | None = None
    overwrite: bool = False
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    config: dict = field(default_factory=dict)
    variant: str | None = None
    root: Path = field(init=False)
    _started: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config_path is None:
            from .config import active_config_path

            self.config_path = active_config_path()
        if self.config_path is not None:
            self.config_path = Path(self.config_path)
            if not self.config_path.is_absolute():
                self.config_path = REPO_ROOT / self.config_path
            if not self.config:
                from .config import load_config

                self.config = load_config(self.config_path)
        if self.variant is None:
            self.variant = config_variant(self.config)
        self.root = resolve_run_dir(self.run, variant=self.variant, subject_limit=self.subject_limit)

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_args(cls, stage: str, args: argparse.Namespace, *, config_path: str | Path | None = None) -> "RunContext":
        return cls(
            stage=stage,
            config_path=config_path if config_path is not None else getattr(args, "config", None),
            run=getattr(args, "run", None),
            subject_limit=getattr(args, "subject_limit", None),
            overwrite=bool(getattr(args, "overwrite", False)),
        )

    # ------------------------------------------------------------------ properties

    @property
    def name(self) -> str:
        return run_name(self.run, self.subject_limit)

    @property
    def config_sha256(self) -> str | None:
        """Digest of the config's settings (see hoc.config.config_digest)."""
        from .config import config_digest

        return config_digest(self.config_path) if self.config_path else None

    def path(self, *parts: str) -> Path:
        """Path inside the run directory, with parent folders created."""
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------------ lifecycle

    def _check_compatible(self, manifest: dict, subject_ids: list | None) -> None:
        problems = []
        cfg_hash = self.config_sha256
        if cfg_hash and manifest.get("config", {}).get("sha256") not in (None, cfg_hash):
            problems.append(
                f"config differs (run made with {manifest['config'].get('path')}, "
                f"sha256 {manifest['config']['sha256'][:12]}; now {cfg_hash[:12]})"
            )
        if manifest.get("subject_limit") != self.subject_limit:
            problems.append(f"subject limit differs ({manifest.get('subject_limit')} vs {self.subject_limit})")
        if subject_ids is not None and manifest.get("subjects") is not None:
            if [str(s) for s in manifest["subjects"]] != [str(s) for s in subject_ids]:
                problems.append("subject set differs")
        stage_record = manifest.get("stages", {}).get(self.stage, {})
        if stage_record.get("status") == "complete":
            problems.append(f"stage '{self.stage}' already finished at {stage_record.get('finished')}")
        if problems and not self.overwrite:
            raise RunConflictError(
                f"Refusing to write into {_relative(self.root)}: " + "; ".join(problems)
                + ". Use a new --run name, or pass --overwrite to replace it."
            )

    def require_stage(self, stage: str) -> dict:
        """Fail unless ``stage`` finished in this run with the same config."""
        manifest = read_manifest(self.root) or {}
        record = manifest.get("stages", {}).get(stage)
        if not record or record.get("status") != "complete":
            raise RunConflictError(f"Stage '{stage}' has not finished in {_relative(self.root)}; run it first.")
        return record

    def begin(self, subject_ids: list | None = None, *, hash_inputs: bool = True) -> "RunContext":
        manifest = read_manifest(self.root)
        if manifest is not None:
            self._check_compatible(manifest, subject_ids)
        else:
            manifest = {
                "run": self.name,
                "variant": self.variant,
                "created": _now(),
                "config": {
                    "path": _relative(self.config_path) if self.config_path else None,
                    "sha256": self.config_sha256,
                },
                "subject_limit": self.subject_limit,
                "subjects": None,
                "inputs": {},
                "stages": {},
            }
        if self.overwrite and self.config_path is not None:
            manifest["config"] = {"path": _relative(self.config_path), "sha256": self.config_sha256}
        if subject_ids is not None:
            manifest["subjects"] = [str(s) if not isinstance(s, (int, float)) else s for s in subject_ids]
        if hash_inputs and self.config:
            for key, path in collect_inputs(self.config):
                stat = path.stat()
                cached = manifest["inputs"].get(key)
                if cached and cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime:
                    continue
                manifest["inputs"][key] = {"sha256": sha256_file(path), "size": stat.st_size, "mtime": stat.st_mtime}
        if self.config_path is not None:
            snapshot = self.root / "config" / self.config_path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(self.config_path.read_bytes())
        previous = manifest["stages"].get(self.stage, {})
        resumed = previous.get("status") == "running" and not self.overwrite
        now = _now()
        self._started = previous["started"] if resumed else now
        attempt = {"time": now, "argv": self.argv, "git": git_state(), "environment": environment_record()}
        manifest["stages"][self.stage] = {
            "status": "running",
            "started": self._started,
            **({"resumed": [*previous.get("resumed", []), now]} if resumed else {}),
            "script": _script_record(self.argv),
            "code": code_record(self.argv[0]) if self.argv else None,
            "random_seed": self.config.get("analysis", {}).get("random_seed"),
            "stage_seeds": self.config.get("seeds", {}),
            "argv": self.argv,
            "overwrite": self.overwrite,
            "git": attempt["git"],
            "environment": attempt["environment"],
            # every start and resume, so a stage that spanned several commits says so
            "attempts": [*(previous.get("attempts", []) if resumed else []), attempt],
        }
        _write_manifest(self.root, manifest)
        return self

    def finish(self, outputs: list[str | Path] | None = None, **extra: object) -> None:
        """Mark the stage complete and record checksums of its outputs.

        ``outputs`` defaults to every file in the run directory modified since
        the stage started, excluding variant folders, ``logs/`` folders, the config snapshot, ``.DS_Store``
        files, and the folders of other stage groups (``<group>/`` of every other stage in the manifest), which
        are written by those stages. Stages that write into another group's folder must pass ``outputs``.
        """
        manifest = read_manifest(self.root) or {"stages": {}}
        record = manifest["stages"].setdefault(self.stage, {})
        if outputs is None:
            started = _dt.datetime.fromisoformat(self._started).timestamp() if self._started else 0.0
            own = self.stage.split("/")[0]
            foreign = {name.split("/")[0] for name in manifest.get("stages", {})} - {own}
            outputs = []
            for p in self.root.rglob("*"):
                parts = p.relative_to(self.root).parts
                if (p.is_file()
                        and p.name not in (MANIFEST_NAME, MANIFEST_NAME + ".tmp", ".DS_Store")
                        and "variants" not in parts and "logs" not in parts
                        and parts[0] != "config" and parts[0] not in foreign
                        and p.stat().st_mtime >= started - 1):
                    outputs.append(p)
        record["outputs"] = {
            str(Path(p).resolve().relative_to(self.root.resolve())): sha256_file(p) for p in sorted(map(Path, outputs))
        }
        record.update({"status": "complete", "finished": _now(), **extra})
        _write_manifest(self.root, manifest)


# --------------------------------------------------------------------------- CLI glue


def add_run_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--run`` and ``--overwrite`` to a script's parser."""
    parser.add_argument(
        "--run",
        default=None,
        help=f"Run name under runs/ (default: ${RUN_ENV} or '{DEFAULT_RUN}').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a finished stage or a run made with a different config/subject set.",
    )
    return parser


def stage_from_argv(
    stage: str,
    *,
    config_path: str | Path | None = None,
    argv: list[str] | None = None,
    variant: str | None = None,
) -> RunContext:
    """Build a RunContext for scripts that take no other arguments.

    Reads ``--run``, ``--overwrite``, ``--subject-limit`` and ``--config`` from the
    command line (unknown arguments are left alone). Without ``config_path`` the
    active config is used (see :mod:`hoc.config`).
    """
    from .config import active_config_path

    parser = argparse.ArgumentParser(add_help=False)
    add_run_arguments(parser)
    parser.add_argument("--subject-limit", type=int, default=None)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return RunContext(
        stage=stage,
        config_path=config_path if config_path is not None else active_config_path(argv),
        run=known.run,
        subject_limit=known.subject_limit,
        overwrite=known.overwrite,
        variant=variant,
    )


def reader_run_dir(argv: list[str] | None = None, *, variant: str | None = None) -> Path:
    """Run directory for read-only consumers (figure and table scripts).

    Uses ``--run`` from ``argv`` if present, else ``$HOC_RUN``, else the default,
    and fails clearly when the directory does not exist.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default=None)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    path = resolve_run_dir(known.run, variant=variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Run directory {_relative(path)} does not exist. Pass --run <name> or set ${RUN_ENV}."
        )
    return path


def figures_dir(run_dir: str | Path, *parts: str) -> Path:
    """Figure output folder for a run (``<run>/figures/...``), created on demand."""
    path = Path(run_dir) / "figures"
    path = path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def n_subjects(root: str | Path) -> int:
    """Number of subjects in a run's processed connectomes (first configured scale)."""
    import numpy as np

    from .config import scales

    with np.load(Path(root) / "processed" / f"connectomes_scale-{scales()[0]}.npz") as data:
        return int(data["adjacency"].shape[0])


def partial_stage(stage: str, **subsets: tuple[object, object]) -> str:
    """Stage name for a run over a subset of the work.

    Each keyword is ``name=(selected, full)``. If any selection differs from the
    full set, the stage gets a ``:partial[...]`` suffix, so a partial run never
    counts as the finished stage that downstream scripts require.
    """
    parts = []
    for key, (selected, full) in subsets.items():
        sel = sorted(selected) if isinstance(selected, (list, tuple, set)) else selected
        ful = sorted(full) if isinstance(full, (list, tuple, set)) else full
        if sel != ful:
            shown = ",".join(map(str, sel)) if isinstance(sel, list) else str(sel)
            parts.append(f"{key}={shown}")
    return f"{stage}:partial[{';'.join(parts)}]" if parts else stage
