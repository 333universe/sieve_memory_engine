"""patterns.py (was pattern_engine.py, with the empirical-entropy null-model fix) - create, recognise, predict, flag.
pattern_engine.py - a small pattern engine built on the golden-ratio sieve.  One file, numpy only (+ matplotlib for --plot).

It can CREATE patterns from a small spec, RECOGNISE which pattern (and which settings) made a symbol sequence, even a noisy
window from the middle, PREDICT what comes next, and FLAG the positions that break the pattern.

    spec = {"family": "rotation", "K": 3, "alpha": 0.7236, "offset": 0.3}      # the sieve, with free dials
    eng = PatternEngine()
    seq = eng.create(spec, 200)                    # make a pattern
    rep = eng.recognise(seq)                       # -> family, settings, mismatches, bits saved, verdict
    nxt = eng.continue_pattern(rep, 50)            # predict the next 50 symbols

Families built in
    rotation   floor(K*frac(n*alpha + offset))        the sieve itself (alpha irrational: never repeats)
    loop       a short word repeated forever          (Euclidean-rhythm style loops, any rational sieve)
    fibword    the Fibonacci word 0100101001001...    (a different quasi-periodic pattern, K=2)
Recognition = pick the family that describes the data in the fewest bits (minimum description length).  If nothing saves
at least `margin_bits` bits versus storing the raw symbols, the verdict is "unknown".  Random data should come out unknown.

To ADD a family: subclass Family, give it `name`, `generate(spec, n, start)` and `fit(seq, K)`, then @register it
(or put an instance into engine.families).  fit() returns {"spec":..., "mismatches":int, "bits":float} or None.

    python pattern_engine.py --demo      create -> recognise round trips
    python pattern_engine.py --bench     recognition accuracy, noise tolerance, false alarms
    python pattern_engine.py --plot      write pattern_engine_demo.png
    python pattern_engine.py --test      self-checks; prints "all N checks passed"
"""
from __future__ import annotations
import json, math, sys
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def mismatch_bits(N: int, K: int, m: int) -> float:
    """bits to say WHICH m of N symbols differ from the pattern, and what they are instead"""
    if m <= 0: return 0.0
    return (math.lgamma(N + 1) - math.lgamma(m + 1) - math.lgamma(N - m + 1)) / math.log(2) + m * math.log2(max(K - 1, 1))


def _pick(cost):
    """index of the middle of the longest run of minimum-cost entries"""
    idx = np.flatnonzero(cost == cost.min())
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    r = max(runs, key=len)
    return int(r[len(r) // 2])


# ------------------------------------------------------------------ families
REGISTRY: dict = {}

class Family:
    name = "base"
    def generate(self, spec, n, start=1): raise NotImplementedError
    def fit(self, seq, K): raise NotImplementedError

def register(cls):
    REGISTRY[cls.name] = cls()
    return cls


@register
class Rotation(Family):
    name = "rotation"

    def generate(self, spec, n, start=1):
        pos = np.arange(start, start + n, dtype=np.float64)
        return np.floor(spec["K"] * ((pos * spec["alpha"] + spec["offset"]) % 1.0)).astype(np.int64) % spec["K"]

    @staticmethod
    def _cost(alphas, i, s, K):
        """order the symbols by phase frac(i*alpha); a true alpha gives K runs in cyclic order 0,1,..,K-1"""
        order = np.argsort((alphas[:, None] * i[None, :]) % 1.0, axis=1, kind="stable")
        ss = s[order]; nxt = np.roll(ss, -1, axis=1)
        diff = ss != nxt
        bad = diff & (nxt != (ss + 1) % K)
        return diff.sum(1) + bad.sum(1)

    def fit(self, seq, K):
        s = np.asarray(seq, dtype=np.int64); N = len(s)
        if K < 2 or N < 12: return None
        n_fit = min(N, 2400)
        stages = [m for m in (80, 250, 800, 2400) if m < n_fit] + [n_fit]
        M0 = stages[0]
        ind = np.stack([(s[:M0] == j).astype(float) for j in range(K)]); ind -= ind.mean(1, keepdims=True)
        P = np.abs(np.fft.rfft(ind, n=16 * M0, axis=1)).sum(0); f = np.fft.rfftfreq(16 * M0)
        pk = np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])) + 1
        pk = pk[np.argsort(-P[pk])][:4]
        finalists = []
        for c in [v for k in pk for v in (f[k], 1 - f[k])]:
            a, h = c, 1.5 / M0
            for m in stages:
                d = 1.0 / (2 * K * m * m)
                alphas = np.linspace(a - h, a + h, min(int(2 * h / d) + 1, 4001))
                cost = self._cost(alphas, np.arange(1, m + 1), s[:m], K)
                cmin = int(cost.min())
                # every alpha that explains the data about as well as the best one (within one extra bad symbol):
                # take the middle of the widest such run, and search that whole run (plus margin) at the next, finer stage
                idx = np.flatnonzero(cost <= cmin + 2)
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                r = max(runs, key=len); step = alphas[1] - alphas[0]
                a = float(alphas[r[len(r) // 2]]); h = (r[-1] - r[0]) * step + 3 * step
            finalists.append((cmin, a))
        # a mirrored alpha (1-alpha) orders the data backwards but can tie on cost, so judge the best few finalists by how
        # well the real generating rule (with its best offset) reproduces the symbols
        finalists.sort(key=lambda t: t[0]); chosen = []
        for c, a in finalists:
            if all(abs(a - b) > 1e-4 for _, b in chosen): chosen.append((c, a))
            if len(chosen) == 3: break
        if not chosen: return None
        best = None
        for _, a in chosen:
            phase = (a * np.arange(1, n_fit + 1)) % 1.0
            G = 4 * K * n_fit; th = np.arange(G) / G; agree = np.empty(G, dtype=np.int64)
            for lo in range(0, G, 400):
                pred = np.floor(K * ((phase[None, :] + th[lo:lo + 400, None]) % 1.0)).astype(np.int64) % K
                agree[lo:lo + 400] = (pred == s[None, :n_fit]).sum(1)
            spec = {"family": "rotation", "K": K, "alpha": a, "offset": float(th[_pick(-agree)])}
            m = int((self.generate(spec, N) != s).sum())
            if best is None or m < best[0]: best = (m, spec)
        m, spec = best
        return {"spec": spec, "mismatches": m, "bits": math.log2(8 * K * K * N ** 3) + mismatch_bits(N, K, m)}


@register
class Loop(Family):
    name = "loop"

    def generate(self, spec, n, start=1):
        w = np.asarray(spec["word"], dtype=np.int64)
        return w[(start - 1 + np.arange(n)) % len(w)]

    def fit(self, seq, K):
        s = np.asarray(seq, dtype=np.int64); N = len(s); best = None
        if K < 2 or N < 6: return None
        for p in range(1, N // 3 + 1):
            cnt = np.bincount((np.arange(N) % p) * K + s, minlength=p * K).reshape(p, K)
            m = int(N - cnt.max(1).sum())
            bits = math.log2(max(N // 3, 1)) + p * math.log2(K) + mismatch_bits(N, K, m)
            if best is None or bits < best["bits"]:
                best = {"spec": {"family": "loop", "K": K, "word": [int(v) for v in cnt.argmax(1)]}, "mismatches": m, "bits": bits}
        return best


def _fibword(n):
    w = [0]
    while len(w) < n:
        w = [y for x in w for y in ((0, 1) if x == 0 else (0,))]
    return np.array(w[:n], dtype=np.int64)

@register
class FibWord(Family):
    name = "fibword"
    MAXSHIFT = 400

    def generate(self, spec, n, start=1):
        return _fibword(spec["shift"] + start - 1 + n)[spec["shift"] + start - 1:]

    def fit(self, seq, K):
        s = np.asarray(seq, dtype=np.int64); N = len(s)
        if K != 2 or N < 12: return None
        n = min(N, 600); W = _fibword(self.MAXSHIFT + n)
        win = np.lib.stride_tricks.sliding_window_view(W, n)[:self.MAXSHIFT]
        sh = int(np.argmin((win != s[:n]).sum(1)))
        spec = {"family": "fibword", "K": 2, "shift": sh}
        m = int((self.generate(spec, N) != s).sum())
        return {"spec": spec, "mismatches": m, "bits": math.log2(self.MAXSHIFT) + mismatch_bits(N, 2, m)}


# ------------------------------------------------------------------ the engine
class PatternEngine:
    def __init__(self):
        self.families = dict(REGISTRY)

    def create(self, spec, n, start=1):
        return self.families[spec["family"]].generate(spec, n, start)

    def recognise(self, seq, K=None, margin_bits=16.0):
        s = np.asarray(seq, dtype=np.int64); N = len(s)
        out = {"N": N, "verdict": "unknown", "spec": None, "family": None, "mismatches": None, "mismatch_positions": [],
               "bits": None, "raw_bits": None, "saved_bits": 0.0, "noise": None, "candidates": []}
        if N < 24: out["verdict"] = "too short (need at least 24 symbols)"; return out
        if len(np.unique(s)) < 2: out["verdict"] = "constant"; return out
        K = K or int(s.max()) + 1
        # null model: the cheaper of 'uniform symbols' and 'fixed symbol frequencies' (empirical entropy + cost of stating them).
        # With only the uniform null, any skewed-but-random source looks structured (a length-1 'loop' of the commonest symbol).
        p = np.bincount(s, minlength=K) / N; p = p[p > 0]
        raw = min(N * math.log2(K), float(-(p * np.log2(p)).sum()) * N + 0.5 * (K - 1) * math.log2(N)); out["raw_bits"] = raw
        for name, fam in self.families.items():
            r = fam.fit(s, K)
            if r: out["candidates"].append((name, round(r["bits"], 1), r["mismatches"], r["spec"]))
        if not out["candidates"]: return out
        out["candidates"].sort(key=lambda c: c[1])
        name, bits, m, spec = out["candidates"][0]
        out.update(bits=bits, mismatches=m, noise=m / N, saved_bits=raw - bits)
        if raw - bits >= margin_bits:
            out.update(verdict="recognised", family=name, spec=spec)
            out["mismatch_positions"] = [int(i) for i in np.flatnonzero(self.create(spec, N) != s)]
        return out

    def continue_pattern(self, report, n):
        if report["spec"] is None: raise ValueError("pattern not recognised; nothing to continue")
        return self.create(report["spec"], n, start=report["N"] + 1)

    @staticmethod
    def explore(count, seed=0):
        """a reproducible set of random pattern specs (a catalogue to browse)"""
        rng = np.random.default_rng(seed); specs = []
        for _ in range(count):
            t = rng.choice(["rotation", "rotation", "loop", "fibword"])
            K = int(rng.integers(2, 6))
            if t == "rotation": specs.append({"family": "rotation", "K": K, "alpha": round(float(rng.uniform(0.05, 0.95)), 6), "offset": round(float(rng.random()), 4)})
            elif t == "loop": specs.append({"family": "loop", "K": K, "word": [int(v) for v in rng.integers(0, K, int(rng.integers(2, 13)))]})
            else: specs.append({"family": "fibword", "K": 2, "shift": int(rng.integers(0, 100))})
        return specs


def to_json(spec): return json.dumps(spec)
def from_json(txt): return json.loads(txt)
def show(seq, n=60): return "".join(".#@%*"[v] if v < 5 else str(v) for v in np.asarray(seq)[:n])


# ------------------------------------------------------------------ demo / bench / plot / tests
def noisy(seq, rate, K, rng):
    s = np.array(seq).copy(); flip = np.flatnonzero(rng.random(len(s)) < rate)
    s[flip] = (s[flip] + rng.integers(1, K, len(flip))) % K
    return s, flip

def demo():
    eng = PatternEngine(); rng = np.random.default_rng(1)
    print("create -> window from the middle (3% noise) -> recognise -> predict the next 200\n")
    for spec in eng.explore(6, seed=3):
        K = spec["K"]; far = int(rng.integers(0, 5000)); win, _ = noisy(eng.create(spec, 300, start=1 + far), 0.03, K, rng)
        rep = eng.recognise(win)
        print("made     :", to_json(spec)); print("pattern  :", show(win, 70))
        if rep["verdict"] == "recognised":
            truth = eng.create(spec, 200, start=1 + far + 300); pred = eng.continue_pattern(rep, 200)
            print(f"found    : {rep['family']} {to_json(rep['spec'])[:90]}  | flagged {len(rep['mismatch_positions'])} odd symbols | saved {rep['saved_bits']:.0f} bits")
            print(f"predicted: next 200 symbols correct {np.mean(pred == truth):.1%}\n")
        else: print("found    :", rep["verdict"], "\n")

def bench():
    eng = PatternEngine(); rng = np.random.default_rng(7)
    def trial(spec, N, rate, ahead=400):
        far = int(rng.integers(0, 5000)); K = spec["K"]
        win, _ = noisy(eng.create(spec, N, start=1 + far), rate, K, rng); rep = eng.recognise(win, K=K)
        ok = rep["verdict"] == "recognised"; fam = ok and rep["family"] == spec["family"]
        acc = np.mean(eng.continue_pattern(rep, ahead) == eng.create(spec, ahead, start=1 + far + N)) if ok else np.nan
        return ok, fam, acc
    def rot(): K = int(rng.integers(2, 6)); return {"family": "rotation", "K": K, "alpha": float(rng.uniform(0.02, 0.98)), "offset": float(rng.random())}
    def loop(): K = int(rng.integers(2, 6)); return {"family": "loop", "K": K, "word": [int(v) for v in rng.integers(0, K, int(rng.integers(2, 31)))]}
    def fib(): return {"family": "fibword", "K": 2, "shift": int(rng.integers(0, 300))}
    print("1  recognition of patterns the engine can make   (window of N symbols from a random position)")
    print(f"  {'family':<10}{'N':>5}{'noise':>7}{'trials':>8}{'recognised':>12}{'right family':>14}{'predict next 400 (mean / worst)':>34}")
    for nm, g, N, rates in [("rotation", rot, 300, [0, .02, .05, .10]), ("loop", loop, 300, [0, .02, .05]), ("fibword", fib, 300, [0, .05])]:
        for r in rates:
            R = [trial(g(), N, r) for _ in range(40)]; acc = np.array([x[2] for x in R]); okc = np.isfinite(acc)
            print(f"  {nm:<10}{N:>5}{r:>7.2f}{len(R):>8}{np.mean([x[0] for x in R]):>12.0%}{np.mean([x[1] for x in R]):>14.0%}"
                  f"{(f'{np.nanmean(acc):.3f} / {np.nanmin(acc):.3f}' if okc.any() else '-'):>34}")
    print("\n2  how short can the window be?  (rotation, K=3, no noise, 40 trials each)")
    for N in [24, 30, 40, 60, 100, 200]:
        R = [trial({"family": "rotation", "K": 3, "alpha": float(rng.uniform(0.02, 0.98)), "offset": float(rng.random())}, N, 0) for _ in range(40)]
        print(f"  N={N:<4} recognised {np.mean([x[0] for x in R]):>5.0%}   right family {np.mean([x[1] for x in R]):>5.0%}   predict next 400 correct {np.nanmean([x[2] for x in R]):.3f}")
    print("\n3  false alarms: random sequences that should come out 'unknown'  (200 each)")
    for K in (2, 3, 5):
        for N in (40, 100, 300):
            fa = sum(eng.recognise(rng.integers(0, K, N), K=K)["verdict"] == "recognised" for _ in range(200))
            print(f"  K={K} N={N:<4} falsely recognised {fa}/200")
    print("\n4  sequences that are structured but NOT in the engine's families (should be unknown, or a fair description)")
    n = np.arange(1, 301); tests = {"original Pi-Phi sieve (floor(n*2pi/phi^2) mod 3)": np.floor(n * 2 * np.pi / PHI ** 2).astype(int) % 3,
                                    "n*n mod 3": (n * n) % 3, "primes indicator (K=2)": np.array([int(all(k % d for d in range(2, int(k ** .5) + 1)) and k > 1) for k in n])}
    try:
        import mpmath; mpmath.mp.dps = 320; digs = [int(c) for c in str(mpmath.pi)[2:302]]; tests["digits of pi, mod 3"] = np.array(digs) % 3
    except Exception: pass
    for nm, seq in tests.items():
        r = eng.recognise(seq); extra = f" ({r['family']}, mismatches {r['mismatches']})" if r["verdict"] == "recognised" else ""
        print(f"  {nm:<52}{r['verdict']}{extra}")
        if r["family"] == "loop": print(f"      loop word: {r['spec']['word']}")

def plot(path="pattern_engine_demo.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    eng = PatternEngine(); rng = np.random.default_rng(5); spec = {"family": "rotation", "K": 3, "alpha": 0.7236, "offset": 0.3}
    clean = eng.create(spec, 120); obs, flip = noisy(clean, 0.05, 3, rng); rep = eng.recognise(obs)
    regen = eng.create(rep["spec"], 120); pred = eng.continue_pattern(rep, 60); truth = eng.create(spec, 60, start=121)
    fig, ax = plt.subplots(4, 1, figsize=(13, 5.2), gridspec_kw={"hspace": 0.9})
    cm = "tab10"
    for a, row, t in [(ax[0], obs, "what the engine sees (5% of symbols corrupted)"), (ax[1], regen, "pattern the engine recognised and rebuilt")]:
        a.imshow(row[None, :], aspect="auto", cmap=cm, vmin=0, vmax=9, interpolation="nearest"); a.set_title(t, fontsize=10); a.set_yticks([])
    ax[2].imshow(np.zeros((1, 120)), aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax[2].scatter(rep["mismatch_positions"], np.zeros(len(rep["mismatch_positions"])), c="red", marker="v", s=60)
    ax[2].set_title(f"positions flagged as breaking the pattern ({len(rep['mismatch_positions'])} flagged, {len(flip)} corrupted)", fontsize=10); ax[2].set_yticks([])
    ax[3].imshow(np.vstack([truth, pred]), aspect="auto", cmap=cm, vmin=0, vmax=9, interpolation="nearest")
    ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["true next 60", "predicted"]); ax[3].set_title(f"prediction of the next 60 symbols: {np.mean(pred == truth):.0%} correct", fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight"); print("wrote", path)

def run_tests():
    k = 0
    def check(c, m):
        nonlocal k; assert c, m; k += 1
    eng = PatternEngine(); rng = np.random.default_rng(11)
    for spec in [{"family": "rotation", "K": 3, "alpha": PHI, "offset": 0.1}, {"family": "rotation", "K": 4, "alpha": 0.3141, "offset": 0.7},
                 {"family": "loop", "K": 3, "word": [0, 2, 1, 1, 0]}, {"family": "fibword", "K": 2, "shift": 37}]:
        win = eng.create(spec, 250, start=1001); rep = eng.recognise(win, K=spec["K"])
        check(rep["verdict"] == "recognised" and rep["family"] == spec["family"], f"recognises clean {spec['family']}")
        acc = np.mean(eng.continue_pattern(rep, 300) == eng.create(spec, 300, start=1251))
        check(acc == 1.0 if spec["family"] != "rotation" else acc >= 0.95, f"predicts clean {spec['family']} (got {acc:.3f})")
    spec = {"family": "rotation", "K": 3, "alpha": 0.7236, "offset": 0.3}; obs, flip = noisy(eng.create(spec, 200), 0.04, 3, rng); rep = eng.recognise(obs)
    check(rep["verdict"] == "recognised" and len(set(rep["mismatch_positions"]) ^ set(flip.tolist())) <= 2, "flags the corrupted symbols")
    check(all(eng.recognise(rng.integers(0, 3, 200))["verdict"] != "recognised" for _ in range(30)), "random data is not recognised")
    check(eng.recognise([0, 1, 2])["verdict"].startswith("too short"), "short input refused")
    check(eng.recognise([1] * 50)["verdict"] == "constant", "constant input named")
    n = np.arange(1, 301); r = eng.recognise(np.floor(n * 2 * np.pi / PHI ** 2).astype(int) % 3)
    check(r["family"] == "loop" and len(r["spec"]["word"]) == 5, "the original sieve is seen as a loop of 5")
    check(from_json(to_json(spec)) == spec, "spec survives JSON")
    class Squares(Family):                                        # a new family in ~10 lines: floor(sqrt(n)) mod K, shifted
        name = "squares"
        def generate(self, spec, n, start=1): return (np.floor(np.sqrt(np.arange(start, start + n) + spec["shift"])).astype(int)) % spec["K"]
        def fit(self, seq, K):
            s = np.asarray(seq); best = min(range(200), key=lambda sh: (self.generate({"K": K, "shift": sh}, len(s)) != s).sum())
            sp = {"family": "squares", "K": K, "shift": best}; m = int((self.generate(sp, len(s)) != s).sum())
            return {"spec": sp, "mismatches": m, "bits": math.log2(200) + mismatch_bits(len(s), K, m)}
    e2 = PatternEngine(); e2.families["squares"] = Squares(); sq = Squares().generate({"K": 3, "shift": 50}, 300)
    check(e2.recognise(sq)["family"] == "squares" and eng.recognise(sq)["family"] != "squares", "a newly added family becomes recognisable")
    print(f"all {k} checks passed")

if __name__ == "__main__":
    a = sys.argv
    if "--test" in a: run_tests()
    elif "--bench" in a: bench()
    elif "--plot" in a: plot()
    else: demo()
