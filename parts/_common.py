import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def part_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--gpu", action="store_true")
    return p


def resolve_mode(args: argparse.Namespace) -> argparse.Namespace:
    if not args.demo and not args.train:
        args.demo = True
    return args


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def save_loss_plot(losses: list[float], out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved {out}")


def timed(label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            print(f"{label}: {time.perf_counter() - self.t0:.3f}s")

    return _Timer()
