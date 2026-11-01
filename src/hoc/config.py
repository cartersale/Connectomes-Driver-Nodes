"""Configuration: loading TOML files and the settings shared by every analysis stage.

The active configuration is ``--config PATH`` on the command line, else ``$HOC_CONFIG``, else
``config/default.toml``. ``scripts/reproduce.py --config`` sets ``$HOC_CONFIG`` for every step it runs, so
all stages of a run read the same file (and the run manifest refuses a stage whose config differs).
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

from .paths import REPO_ROOT

CONFIG_ENV = "HOC_CONFIG"
DEFAULT_CONFIG = "config/default.toml"


def load_config(config_path: str | Path) -> dict:
    """Load a TOML configuration file."""
    path = Path(config_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def config_digest(config_path: str | Path) -> str:
    """SHA-256 of a config's settings (parsed TOML, so comments and layout do not count)."""
    import hashlib
    import json

    canonical = json.dumps(load_config(config_path), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def active_config_path(argv: list[str] | None = None) -> Path:
    """Path of the configuration this process should use (see module docstring)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    path = Path(known.config or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    return path if path.is_absolute() else REPO_ROOT / path


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    return load_config(path)


def active_config() -> dict:
    return _load(str(active_config_path()))


# Settings used across stages. Analysis-specific grids stay in the scripts that use them.
def scales() -> tuple[int, ...]:
    return tuple(int(s) for s in active_config()["analysis"]["parcellation_scales"])


def k_values() -> tuple[int, ...]:
    return tuple(int(k) for k in active_config()["analysis"]["top_k_values"])


def random_seed() -> int:
    return int(active_config()["analysis"]["random_seed"])


def stage_seed(name: str) -> int:
    """Seed of one stage's own random draws (``[seeds]`` table of the config)."""
    return int(active_config()["seeds"][name])


def sampling(name: str) -> int:
    """Size of a resampling or null ensemble (``[sampling]`` table: nulls per subject, spins, draws, ...)."""
    return int(active_config()["sampling"][name])


def n_random() -> int:
    return int(active_config()["rankings"]["n_random_rankings"])


def primary_c() -> float:
    return float(active_config()["control"]["primary_c"])


def primary_epsilon() -> float:
    return float(active_config()["control"]["primary_epsilon"])
