import sys
from pathlib import Path

import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode, save_loss_plot  # noqa: E402

from baremetal_llm.alignment import PreferenceBatcher, PreferenceScorer, bt_loss, load_rankings
from baremetal_llm.utils.checkpoints import save_checkpoint
from baremetal_llm.utils.device import device_label, get_device
from baremetal_llm.utils.paths import part_run_dir


def main():
    args = resolve_mode(part_parser("Part 7").parse_args())
    device = get_device(args.gpu or args.train)
    steps = 30 if args.demo else 150

    banner(f"Part 7: reward model ({device_label(device)})")
    rankings = load_rankings()[:16]
    batcher = PreferenceBatcher(ctx_len=96)
    pos, neg = batcher.collate([(r.prompt, r.chosen, r.rejected) for r in rankings])
    pos, neg = pos.to(device), neg.to(device)

    rm = PreferenceScorer(vocab=256, ctx_len=96, n_layers=2, n_heads=2, dim=64).to(device)
    opt = optim.AdamW(rm.parameters(), lr=2e-3)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = bt_loss(rm(pos), rm(neg))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    acc = (rm(pos) > rm(neg)).float().mean().item()
    run_dir = part_run_dir(7, "rm-demo")
    save_loss_plot(losses, run_dir / "loss.png", "reward model")
    ckpt = run_dir / "model.pt"
    save_checkpoint(ckpt, rm, {"model_type": "reward"})
    print(f"batch accuracy {acc:.1%}")
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
