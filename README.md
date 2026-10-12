# sieve_memory_engine

Stateless, nested, evenly spread forgetting for aging memory (for example, an AI's context window), built on the
golden-ratio sieve.

## The one claim
Give every position `n` a fixed number `phase(n) = frac(n / phi)`. Keep a token while
`phase(n) < rate(age)`, where the rate only falls as the token ages. Then:

- **Even:** no big holes. The largest gap is about 1.3-1.7x the ideal; random keeping leaves 4-7x.
- **Nested:** the smaller memory is always a subset of the bigger one, so nothing ever has to come back.
- **Stateless:** a token's fate is computed from its position and age alone. No list, no rewriting at tier changes.
- **Fair on periodic content:** unlike stride keeping, it does not drop every row of a repeating table.

## Status
| Part | State |
|---|---|
| `sieve_core`, `tiers`, `thinning` | working, 41 checks pass |
| `patterns` (recognise, predict, flag) | working, 15 checks pass, null-model fix applied |
| `experiments/gaps, decay, aliasing` | working |
| `placement`, `ring_churn`, `predict` | not written yet |
| Real model / KV-cache test | not done (position-only so far) |

## Run
    python run_project.py            # menu
    python tests/test_all.py         # all N checks passed
Needs Python 3 and numpy (matplotlib only for `src/patterns.py --plot`).

## Limits (read these)
- The rule is **position-only**. It ignores content, so it is a coverage floor under a smarter scorer, not a policy alone.
- A perfectly even rule has smaller gaps; the advantage here is nestedness plus statelessness.
- Float64 phase is accurate to about 1e-4 up to positions near 1e12.
- Not a cipher and not a general memory allocator: it does nothing for fragmentation or lifetimes.
- Pattern recognition takes 0.25-0.8 s per call: fine for profiling, too slow for per-access decisions.

See `docs/FINDINGS.md` for results and next steps.
