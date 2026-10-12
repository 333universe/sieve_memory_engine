"""Guards the null-model fix in src/patterns.py.
Skewed-but-random symbol streams must NOT be 'recognised' (they used to come out as a length-1 'loop'),
while real patterns, including ones whose symbols are unevenly used, must still be found.
   python tests/test_skew.py   ->  'all N checks passed'   (about 15 seconds)"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.patterns import PatternEngine

k = 0
def check(cond, msg):
    global k
    assert cond, "FAILED: " + msg
    k += 1

eng = PatternEngine(); rng = np.random.default_rng(5); K = 4; TRIALS = 10
for name, p in (("mild skew", [.4, .3, .2, .1]), ("Zipf-like", [.55, .25, .12, .08]), ("heavy skew", [.8, .1, .06, .04])):
    fa = sum(eng.recognise(rng.choice(K, 300, p=p), K=K)["verdict"] == "recognised" for _ in range(TRIALS))
    check(fa <= 1, f"{name}: random skewed data falsely recognised {fa}/{TRIALS} times (allowed: 1)")

# real patterns with uneven symbol use must still be found
uneven = {"family": "loop", "K": 3, "word": [0, 0, 0, 0, 1, 0, 0, 2]}
rep = eng.recognise(eng.create(uneven, 300, start=501), K=3)
check(rep["verdict"] == "recognised" and rep["family"] == "loop" and len(rep["spec"]["word"]) == 8, "uneven loop (word of 8) is still recognised")
rot = {"family": "rotation", "K": 3, "alpha": 0.7236, "offset": 0.3}
rep = eng.recognise(eng.create(rot, 300, start=77), K=3)
check(rep["verdict"] == "recognised" and rep["family"] == "rotation", "clean rotation is still recognised")
check(eng.recognise(rng.integers(0, 3, 300), K=3)["verdict"] != "recognised", "uniform random data is not recognised")
print(f"all {k} checks passed")
