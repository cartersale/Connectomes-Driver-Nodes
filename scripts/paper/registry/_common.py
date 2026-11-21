"""Formatting helpers and the number collector shared by the supplement registry."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .record import Number

SC = (68, 114, 219)
SCN = {68: "SixtyEight", 114: "OneFourteen", 219: "TwoNineteen"}
NULLS = ("strength", "geometric", "weights")
YEO7 = ("net_Visual", "net_SomMot", "net_DorsAttn", "net_VentAttn", "net_Limbic", "net_FrontoParietal", "net_Default")
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def f2(x, d=2):
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def pct(x):
    """Fraction -> integer percent text."""
    return f"{100 * x:.0f}"


def ceil_to(x, d):
    f = 10.0 ** d
    return math.ceil(round(x * f, 9)) / f


def sig1_ceil(x: float) -> str:
    """One significant figure rounded UP (bound), LaTeX exponent form: 4.1e-15 -> '5\\times10^{-15}'."""
    e = math.floor(math.log10(abs(x)))
    m = math.ceil(round(x / 10**e, 9))
    if m == 10:
        m, e = 1, e + 1
    return f"{m}\\times10^{{{e}}}"


class Collector:
    def __init__(self, run: Path):
        self.run = Path(run)
        self.out: list[Number] = []

    def path(self, rel: str) -> Path:
        p = self.run / rel
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    def csv(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.path(rel), **kw)

    def add(self, name, value, computed, printed, source, locus, context=None, bound=None):
        """``computed``: text from the saved output; ``printed``: the token in the document.
        ``bound``: (holds: bool, description) for statements of a bound ("at most", "within")."""
        ctx = dict(context or {})
        if bound is not None:
            ctx["bound"] = bound[1]
            ctx["bound_holds"] = bool(bound[0])
        ctx["printed"] = printed
        if isinstance(locus, str):
            locus = (locus,)
        self.out.append(Number(name, float(value), computed, source, tuple(locus), ctx))

    def done(self) -> list[Number]:
        names = [n.name for n in self.out]
        dup = sorted({n for n in names if names.count(n) > 1})
        assert not dup, dup
        return self.out


# ----------------------------------------------------------------------------- per-scale values of a quoted range
# The Results quote the 114-region value and the supplement lists every scale (Table C_scales). A range helper passes
# its (key, value) pairs here; a key is a scale, or a tuple containing one (criterion, scale), (k, scale), ...

TEXT_SCALE = 114


def _scale_of(key) -> int | None:
    if isinstance(key, (int, float)) and int(key) in SC and key == int(key):
        return int(key)
    if isinstance(key, tuple):
        hits = [x for x in key if isinstance(x, (int, float)) and not isinstance(x, bool) and x in SC]
        return int(hits[-1]) if len(hits) == 1 else None
    return None


def by_scale(items) -> dict | None:
    """{scale: (min, max)} over the values at each scale, or None when the keys do not identify every value's scale."""
    groups: dict[int, list[float]] = {}
    for key, value in items:
        s = _scale_of(key)
        if s is None:
            return None
        groups.setdefault(s, []).append(float(value))
    if set(groups) != set(SC):
        return None
    return {s: (min(v), max(v)) for s, v in sorted(groups.items())}


def scale_siblings(prefix: str, items, fmt, source: str, locus, context: dict) -> list[Number]:
    """The 114-region value of a range: <prefix>OneFourteen, or <prefix>OneFourteenMin/Max when several values
    (criteria, k) share that scale. Empty when the keys do not give each value's scale."""
    bs = by_scale(items)
    if bs is None:
        return []
    lo, hi = bs[TEXT_SCALE]
    text = {str(s): fmt(a) if fmt(a) == fmt(b) else f"{fmt(a)}--{fmt(b)}" for s, (a, b) in bs.items()}
    ctx = {**context, "scale": TEXT_SCALE, "by_scale": {str(s): list(v) for s, v in bs.items()}, "by_scale_text": text,
           "range_of": prefix}
    src = source + f" | scale == {TEXT_SCALE}"
    name = prefix + SCN[TEXT_SCALE]
    if lo == hi or fmt(lo) == fmt(hi):
        return [Number(name, lo, fmt(lo), src, tuple(locus), ctx)]
    return [Number(name + "Min", lo, fmt(lo), src + " | min", tuple(locus), {**ctx, "end": "min"}),
            Number(name + "Max", hi, fmt(hi), src + " | max", tuple(locus), {**ctx, "end": "max"})]
