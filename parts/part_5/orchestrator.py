import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode  # noqa: E402

from baremetal_llm.scaling import MixtureOfExperts
from baremetal_llm.utils.paths import part_run_dir


def main():
    resolve_mode(part_parser("Part 5").parse_args())
    banner("Part 5: MoE routing")

    torch.manual_seed(1)
    moe = MixtureOfExperts(dim=64, n_experts=8, k=2)
    x = torch.randn(4, 16, 64)
    _, aux = moe(x)

    with torch.no_grad():
        flat = x.reshape(-1, 64)
        idx, _, _ = moe.router(flat)
        counts = torch.bincount(idx[:, 0], minlength=8).float()

    out = part_run_dir(5, "demo") / "expert_routing.png"
    plt.figure(figsize=(6, 3))
    plt.bar(range(8), counts.numpy())
    plt.title(f"expert load (aux={aux.item():.3f})")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
