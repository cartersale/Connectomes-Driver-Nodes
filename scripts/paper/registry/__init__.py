"""Registry of every number quoted in the manuscript, recomputed from the outputs of a run.

Each module covers one part of the paper and defines ``collect(run) -> list[Number]``, where ``run`` is the run
directory (``runs/<name>/``). Modules for the main text also define ``PRINTED``: the token the manuscript
prints for each number; the Methods, supplement and additional-analyses modules keep that token in
``context["printed"]``. ``additional`` covers supplementary/additional_analyses.md. ``scripts/paper/validate.py`` reports every
number whose computed text differs from the printed token.

Formatting conventions (``Number.text``):

- The text holds the number token only: sign, digits, decimals; no ``$`` or ``\\%``. ``86--96\\%`` is two
  numbers, each end of the range computed as the min / max over the set the sentence describes.
- Percentages are printed as the percentage number; counts of subjects as the count ("65 of 70" -> ``65``).
- A stated bound ("within 0.07", "at most 25%", "p < 10^{-5}") is rounded so that the statement stays true
  (up for "within"/"at most", down for "at least"); ``value`` holds the actual extreme.
- A few results are printed as words ("one", "five to six times", "every"); the text holds the word and
  ``value`` the number.
- Names: ``M`` (main text), ``S`` (supplement) or ``A`` (additional analyses) + CamelCase letters; per-scale numbers end in
  ``SixtyEight`` / ``OneFourteen`` / ``TwoNineteen``; range ends in ``Min`` / ``Max``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from . import additional, main_core_divisions, main_energy_hubs, main_leading_mode, main_nulls, methods, supplement
from .record import Number

MODULES = (main_energy_hubs, main_leading_mode, main_core_divisions, main_nulls, methods, supplement, additional)


def collect(run: Path) -> list[Number]:
    """Every registered number, in manuscript order, with the printed token in ``context["printed"]``."""
    numbers: list[Number] = []
    for module in MODULES:
        printed = getattr(module, "PRINTED", {})
        for n in module.collect(Path(run)):
            context = dict(n.context)  # the two ends of a range may share one context dict
            if n.name in printed:
                context.setdefault("printed", printed[n.name])
            context.setdefault("section", module.__name__.rsplit(".", 1)[-1])
            numbers.append(replace(n, context=context))
    names = [n.name for n in numbers]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate number names: {dupes}")
    bad = [n for n in names if not re.fullmatch(r"[A-Za-z]+", n)]
    if bad:
        raise ValueError(f"names must be letters only (they are LaTeX macros): {bad}")
    return numbers
