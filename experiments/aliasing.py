"""Periodic content (tables, templates): does thinning keep a fair share of the important tokens?"""
import numpy as np
from _common import save
from src.thinning import keep

T = 20000; pos = np.arange(1, T + 1); rng = np.random.default_rng(42)
lines = ["share of important tokens kept (important = every p-th position, keep rate 1/p; fair = 1/p)",
         f"{'p':>3}{'fair':>7}{'golden':>8}{'random':>8}{'stride':>8}"]
for p in (3, 4, 5, 8, 12):
    imp = (pos % p) == 1
    lines.append(f"{p:>3}{1 / p:>7.2f}{keep(pos, 1 / p)[imp].mean():>8.2f}{(rng.random(T) < 1 / p)[imp].mean():>8.2f}{((pos % p) == 0)[imp].mean():>8.2f}")
lines.append("stride keeps all or none of a periodic pattern; golden and random stay fair, golden without big holes")
out = "\n".join(lines); print(out); save("aliasing", out)
