"""Menu launcher.   python run_project.py   (or pass a number: python run_project.py 1)"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS = [("Run self-checks", "tests/test_all.py"), ("Hole sizes after thinning", "experiments/gaps.py"),
         ("Repeated decay", "experiments/decay.py"), ("Periodic content (aliasing)", "experiments/aliasing.py"),
         ("Pattern engine self-checks", "src/patterns.py --test")]
def main():
    pick = sys.argv[1] if len(sys.argv) > 1 else None
    if pick is None:
        for i, (n, _) in enumerate(ITEMS, 1): print(f"  {i}. {n}")
        pick = input("Choose a number (or press Enter to quit): ").strip()
    if not pick.isdigit() or not 1 <= int(pick) <= len(ITEMS): return
    cmd = ITEMS[int(pick) - 1][1].split()
    subprocess.run([sys.executable, os.path.join(ROOT, cmd[0])] + cmd[1:], cwd=ROOT)
if __name__ == "__main__": main()
