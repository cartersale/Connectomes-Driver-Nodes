"""The record every registry module returns: one number quoted in the paper."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Number:
    """One number quoted in the main text or supplement.

    ``name``     stable identifier and LaTeX macro name (letters only), e.g. ``MQOneHubTwoFiveMin``;
                 ``M`` = main text, ``S`` = supplement.
    ``value``    the unrounded value (``nan`` only when no saved output gives it).
    ``text``     the value as it is presented (rounded, formatted), computed from ``value``.
    ``source``   ``<stage>: <file in the run directory> | <selection / formula>``.
    ``locus``    manuscript labels where it is quoted (``subsec:...``, ``fig:...``, ``tab:...``).
    ``context``  what the number refers to: scale, k, c, criterion, comparison, statistic, ...
    """

    name: str
    value: float
    text: str
    source: str
    locus: tuple[str, ...]
    context: dict = field(default_factory=dict, hash=False, compare=False)
