"""sieve_core.py - the golden-ratio phase that everything else in this repo builds on.

    phase(n) = frac(n / phi)        n = 1, 2, 3, ...  (a token's position)

Every position gets one fixed number in [0, 1).  The numbers are evenly spread (three-gap theorem), so
"keep the positions whose phase is below r" keeps a share r of them with no big holes.

Precision: float64 is accurate to about 1e-4 in the phase for positions up to ~1e12.  Beyond that, use
exact integer (Fibonacci) arithmetic instead.
"""
from __future__ import annotations
import numpy as np

PHI = (1 + 5 ** 0.5) / 2
INV_PHI = 1 / PHI                      # 0.6180339887...


def phase(n):
    """frac(n / phi).  n: an int or an array of ints.  Returns a float, or an array of floats, in [0, 1)."""
    out = (np.asarray(n, dtype=np.float64) * INV_PHI) % 1.0
    return float(out) if out.ndim == 0 else out


def channels(count: int, k: int = 3, start: int = 1) -> np.ndarray:
    """Split positions start .. start+count-1 into k evenly shared channels 0..k-1 (the 'fixed' sieve for k=3)."""
    pos = np.arange(start, start + count)
    return np.floor(k * phase(pos)).astype(np.int64)
