"""thinning.py - stateless, nested, evenly spread forgetting.

A token at position p is alive at time `now` if   phase(p) < rate_for_age(now - p).
Nothing is stored per token and nothing is rewritten when a tier boundary passes.  Because phase(p) never
changes and rates only fall with age, the survivors of a later time are always a subset of the survivors of an
earlier time (for the same positions).
"""
from __future__ import annotations
import numpy as np
from .sieve_core import phase
from .tiers import Schedule


def keep(positions, rate):
    """One-shot thinning: True where the token is kept at a flat keep rate."""
    return phase(positions) < rate


def alive(positions, now: int, schedule: Schedule):
    """True where the token at each position (<= now) is still in memory at time `now`."""
    pos = np.asarray(positions)
    return phase(pos) < schedule.rate_for_age(now - pos)


def survivors(now: int, schedule: Schedule, start: int = 1) -> np.ndarray:
    pos = np.arange(start, now + 1)
    return pos[alive(pos, now, schedule)]


def max_gap(positions) -> int:
    """largest distance between neighbouring kept positions (the biggest hole in memory)"""
    p = np.asarray(positions)
    return int(np.diff(p).max()) if len(p) > 1 else 0
