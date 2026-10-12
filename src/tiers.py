"""tiers.py - an age schedule: how much of the memory to keep at each age.

A Schedule maps a token's age (now - position) to a keep rate.  Rates must never increase with age, because an
evicted token cannot come back.  That one rule is what makes the whole scheme nested and stateless.
"""
from __future__ import annotations
import numpy as np


class Schedule:
    def __init__(self, breaks, rates):
        """breaks: ascending ages where each tier starts, first must be 0.  rates: keep rate (0..1) of each tier."""
        b = np.asarray(breaks, dtype=np.int64); r = np.asarray(rates, dtype=np.float64)
        if len(b) != len(r) or len(b) == 0: raise ValueError("breaks and rates must be the same non-zero length")
        if b[0] != 0 or np.any(np.diff(b) <= 0): raise ValueError("breaks must start at 0 and strictly increase")
        if np.any(r < 0) or np.any(r > 1): raise ValueError("rates must be between 0 and 1")
        if np.any(np.diff(r) > 1e-12): raise ValueError("rates must not increase with age (evicted tokens cannot return)")
        self.breaks, self.rates = b, r

    def rate_for_age(self, age):
        """keep rate for an age or an array of ages"""
        idx = np.searchsorted(self.breaks, np.asarray(age), side="right") - 1
        return self.rates[np.clip(idx, 0, len(self.rates) - 1)]

    @classmethod
    def geometric(cls, first_window=256, tiers=4, length_factor=4, rate_factor=0.5):
        """full detail for the newest `first_window` tokens, then each older tier is `length_factor` times longer
        and keeps `rate_factor` times as many tokens as the one before"""
        breaks, rates, start, length, rate = [], [], 0, first_window, 1.0
        for _ in range(tiers):
            breaks.append(start); rates.append(rate); start += length; length *= length_factor; rate *= rate_factor
        return cls(breaks, rates)
