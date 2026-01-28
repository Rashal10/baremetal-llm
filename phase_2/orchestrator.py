                                                 
                                                                         
import subprocess
import sys
import pathlib
import argparse
import shlex

ROOT = pathlib.Path(__file__).resolve().parent


def run(cmd: str, cwd: pathlib.Path):
    """Run a command in the specified directory."""
    print(f"\n>>> [{cwd.name}] {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=cwd)
    if res.returncode != 0:
        print(f"Command failed with return code {res.returncode}")
        sys.exit(res.returncode)


def run_part(part_dir: pathlib.Path, demo: bool = False):
    """Run the orchestrator for a specific part."""
    orchestrator = part_dir / "orchestrator.py"
    if not orchestrator.exists():
        print(f"Warning: {orchestrator} not found, skipping...")
        return
    
    cmd = "python orchestrator.py"
    if demo:
        cmd += " --demo"
    
    run(cmd, part_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Scaling & Specialization")
    parser.add_argument("--demo", action="store_true", help="Run demos for all parts")
    parser.add_argument("--part", type=int, choices=[4, 5, 6], help="Run only a specific part")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2: Scaling & Specialization")
    print("=" * 60)

                        
    tiny_txt = ROOT.parent / "phase_1" / "part_2" / "tiny.txt"
    if not tiny_txt.exists():
        print(f"Warning: {tiny_txt} not found. Phase 1 may not have been run yet.")

    parts = [4, 5, 6] if args.part is None else [args.part]

    for part_num in parts:
        part_dir = ROOT / f"part_{part_num}"
        print(f"\n{'=' * 60}")
        print(f"Running Part {part_num}...")
        print("=" * 60)
        run_part(part_dir, demo=args.demo)

    print("\n" + "=" * 60)
    print("Phase 2 complete! ")
    print("=" * 60)


if __name__ == "__main__":
    main()
