"""How big is the largest hole in memory after thinning?  golden vs random vs perfectly even."""
import numpy as np
from _common import save
from src.sieve_core import phase
from src.thinning import keep, max_gap

T = 20000; pos = np.arange(1, T + 1); rng = np.random.default_rng(42)
lines = [f"largest hole after one-shot thinning of {T} tokens (ideal = 1/rate)",
         f"{'rate':>7}{'golden':>9}{'random':>9}{'even':>7}"]
for r in (1 / 2, 1 / 3, 0.3, 1 / 5, 1 / 8, 1 / 16):
    gold = max_gap(pos[keep(pos, r)])
    rand = np.mean([max_gap(pos[rng.random(T) < r]) for _ in range(20)])
    even = max_gap(pos[np.floor(pos * r) != np.floor((pos - 1) * r)])
    lines.append(f"{r:>7.3f}{gold:>9d}{rand:>9.1f}{even:>7d}")
lines.append("even = evenly spaced but not nested across rates; random = independent coin flips")
out = "\n".join(lines); print(out); save("gaps", out)
