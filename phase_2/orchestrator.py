import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ROOT.parent / "parts"


def run_part(part: int, demo: bool = False):
    d = PARTS / f"part_{part}"
    cmd = [sys.executable, "orchestrator.py"]
    if demo:
        cmd.append("--demo")
    print(f"\n>> part {part}")
    subprocess.run(cmd, cwd=d, check=True)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 2")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--part", type=int, choices=[4, 5, 6])
    args = p.parse_args()

    print("Phase 2: Scaling")
    for n in ([args.part] if args.part else [4, 5, 6]):
        run_part(n, demo=args.demo)


if __name__ == "__main__":
    main()
