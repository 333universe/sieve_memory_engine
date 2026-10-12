"""Ten rounds of 20% eviction: are the survivors still evenly spread?"""
import numpy as np
from _common import save
from src.thinning import keep, max_gap

T = 20000; pos = np.arange(1, T + 1); rng = np.random.default_rng(42)
final = 0.8 ** 10
gold = pos[keep(pos, final)]
alive = np.ones(T, bool)
for _ in range(10):
    ids = np.flatnonzero(alive); alive[rng.choice(ids, int(round(.2 * len(ids))), replace=False)] = False
rnd = pos[alive]
alive = np.ones(T, bool)
for _ in range(10):
    alive[np.flatnonzero(alive)[4::5]] = False
st = pos[alive]
ideal = 1 / final
lines = [f"after 10 rounds of dropping 20% (final keep rate {final:.3f}, ideal gap {ideal:.1f})",
         f"{'method':<30}{'kept':>7}{'max gap':>9}{'x ideal':>9}"]
for nm, s in (("golden threshold (nested)", gold), ("random 20% each round", rnd), ("evict every 5th survivor", st)):
    lines.append(f"{nm:<30}{len(s):>7d}{max_gap(s):>9d}{max_gap(s) / ideal:>9.2f}")
out = "\n".join(lines); print(out); save("decay", out)
