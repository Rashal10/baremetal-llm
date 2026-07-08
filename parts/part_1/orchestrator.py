import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode  # noqa: E402

from baremetal_llm.foundation import GroupedAttention
from baremetal_llm.utils.paths import part_run_dir


def main():
    args = resolve_mode(part_parser("Part 1").parse_args())
    if args.train:
        print("part 1 is visualization only")
        return

    banner("Part 1: attention heatmap")
    torch.manual_seed(0)
    x = torch.randn(1, 12, 64)
    attn = GroupedAttention(64, 4)
    with torch.no_grad():
        _, weights = attn(x)

    out = part_run_dir(1, "demo") / "attention_heatmap.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(weights[0, 0].cpu().numpy(), cmap="viridis")
    plt.title("head 0")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
