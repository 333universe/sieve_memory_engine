import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.chdir(ROOT)
def save(name, text):
    os.makedirs("outputs/benchmarks", exist_ok=True)
    with open(f"outputs/benchmarks/{name}.txt", "w", encoding="utf-8") as f: f.write(text + "\n")
