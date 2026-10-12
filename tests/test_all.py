"""Self-checks for sieve_core, tiers and thinning.   python tests/test_all.py   ->  'all N checks passed'"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.sieve_core import phase, channels
from src.tiers import Schedule
from src.thinning import keep, alive, survivors, max_gap

k = 0
def check(cond, msg):
    global k
    assert cond, "FAILED: " + msg
    k += 1

rng = np.random.default_rng(0)
T = 20000
pos = np.arange(1, T + 1)

# --- sieve_core
u = phase(pos)
check(u.min() >= 0 and u.max() < 1, "phase stays in [0, 1)")
check(phase(777) == phase(np.arange(1, 1000))[776], "a position's phase does not depend on how many others are computed")
check(abs(u.mean() - 0.5) < 0.01, "phases are evenly spread (mean near 0.5)")
ch = channels(3000, 3)
check(set(np.unique(ch)) == {0, 1, 2} and abs((ch == 0).mean() - 1 / 3) < 0.01, "3 channels each get about a third")

# --- one-shot thinning: rate, holes, nesting, aliasing
for r in (1 / 2, 1 / 3, 0.3, 1 / 5, 1 / 8, 1 / 16):
    kept = pos[keep(pos, r)]
    check(abs(len(kept) / T - r) < 0.005, f"keep rate {r:.3f} is accurate")
    check(max_gap(kept) <= int(1.75 / r) + 1, f"largest hole at rate {r:.3f} is within 1.75x the ideal (got {max_gap(kept)})")
    rnd = pos[rng.random(T) < r]
    check(max_gap(rnd) > max_gap(kept), f"random keeping leaves a bigger hole than golden at rate {r:.3f}")
for r_small, r_big in ((1 / 3, 1 / 2), (0.1, 0.3), (0.05, 0.06)):
    check(not np.any(keep(pos, r_small) & ~keep(pos, r_big)), f"keep({r_small}) is a subset of keep({r_big}) (nested)")
for per in (3, 4, 5, 8, 12):
    important = (pos % per) == 1
    share = keep(pos, 1 / per)[important].mean()
    check(abs(share - 1 / per) < 0.03, f"period-{per} content keeps a fair share ({share:.3f})")
check(keep(pos, 1.0).all() and not keep(pos, 0.0).any(), "rate 1 keeps everything, rate 0 keeps nothing")

# --- schedules
check(np.allclose(Schedule([0, 100, 500], [1, .5, .1]).rate_for_age([0, 99, 100, 499, 500, 10 ** 6]), [1, 1, .5, .5, .1, .1]), "rate_for_age picks the right tier")
for bad in (([0, 10], [0.2, 0.5]), ([5, 10], [1, .5]), ([0, 10, 10], [1, .5, .2]), ([0, 10], [1.2, .5])):
    try: Schedule(*bad); ok = False
    except ValueError: ok = True
    check(ok, f"invalid schedule rejected: {bad}")
g = Schedule.geometric(256, 4, 4, 0.5)
check(list(g.rates) == [1, .5, .25, .125] and list(g.breaks) == [0, 256, 1280, 5376], "geometric schedule has the expected tiers")

# --- stateless aging
S = Schedule([0, 500, 2000, 6000], [1, .5, .2, .05])
now1, now2 = 8000, 9500
s1 = set(survivors(now1, S).tolist()); s2 = set(survivors(now2, S).tolist())
check({p for p in s2 if p <= now1} <= s1, "later survivors are a subset of earlier ones (a token never comes back)")
check(all(alive(np.arange(now1 - 499, now1 + 1), now1, S)), "the newest tier keeps everything")
exp = 500 * 1 + 1500 * .5 + 4000 * .2 + 2000 * .05
check(abs(len(s1) - exp) / exp < 0.03, f"memory size matches the schedule ({len(s1)} vs {exp:.0f})")
oldest = np.array(sorted(p for p in s1 if p <= 2000))
check(max_gap(oldest) <= int(1.75 / .05) + 1, f"the oldest tier still has no big hole (got {max_gap(oldest)})")

print(f"all {k} checks passed")
