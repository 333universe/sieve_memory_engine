# Update for github.com/333universe/sieve_memory_engine

## Add these two files
| File in this zip | Put it in the repo at |
|---|---|
| notebooks/sieve_memory_engine.ipynb | notebooks/sieve_memory_engine.ipynb (replaces the root copy, if you saved one there) |
| tests/test_skew.py | tests/test_skew.py |

## README: add one line near the top (Colab badge)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/333universe/sieve_memory_engine/blob/main/notebooks/sieve_memory_engine.ipynb)

## Why
- The old notebook's cells were out of run order (the clone cell ran 15th, after cells that need the repo), so
  Run all on a fresh runtime would fail. The new one clones first and is safe to re-run.
- Four leftover cells (copies of patterns.py, run_project.py and two stubs) are dropped; the repo already holds those files.
- tests/test_skew.py is the missing guard for the null-model fix: passes with the fix (6 checks, about 8 s) and fails
  without it (random skewed data falsely recognised 6/10 times).

## Layout after the update (adds one folder)
    notebooks/
        sieve_memory_engine.ipynb
