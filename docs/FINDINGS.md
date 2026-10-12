# Findings

## Results (reproduce with `python run_project.py`)
| Test | Result |
|---|---|
| Largest hole, keep rates 1/2 to 1/16 | golden 3, 5, 5, 8, 13, 21; random 14-127; even 2-16 |
| 10 rounds of 20% decay | golden 1.40x ideal gap; evict-every-5th 2.58x; random 6.8x (6-9x depending on seed) |
| Nested (smaller set inside bigger set) | 100% golden; about 50% for even or independent random |
| Periodic content, keep rate 1/p | golden and random keep a fair share; stride keeps 0% or 100% |
| Original Pi-Phi sieve | keeps 20%, not 33%; alias score 0.80. The `fixed` variant keeps 33.4%, alias 0.01 |
| Pattern engine, loops | perfect, but a simple period+repeat predictor ties it |
| Pattern engine, noisy loops | 0.96 vs 0.54 for period+repeat |
| Pattern engine, random data | "unknown" (no false alarms); after the null-model fix also 0/30 on skewed random data |
| Pattern engine speed | 0.25-0.8 s per call at N=300 |

## What did not work
- As a cipher: predictable (serial correlation 0.2, 62% self-match at one lag). Use ChaCha20-Poly1305 or AES-GCM.
- As a noise filter and as a spinor stabilizer: simple baselines beat both by large margins.
- As a quasi-random sampler: better than random, worse than R2, Halton, Sobol; very seed-dependent.
- Golden hash-ring placement under random node removal: busiest node 4.0x average vs 1.3x for 100 virtual nodes.

## Not yet tested
- Real LLM attention or KV-cache behaviour with this thinning rule.
- Ring placement with churn and slot reuse.
- Cache coloring on a real set-associative cache model (the earlier test was simplified).

## Next
1. `experiments/ring_churn.py`: random joins and leaves, removed slots reused.
2. Hybrid: recency window + golden thinning for older tokens + a content score, on a small real model.
3. `experiments/predict.py`: access-pattern traces vs period+repeat and Markov baselines.

## Origins
The sieve began as a token pre-filter (keep / compress / prune). Its audit (`sieve_lab.py` in the earlier repo)
found the original constant gives period 5, not 3.
