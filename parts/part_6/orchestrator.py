import sys
from pathlib import Path

import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode, save_loss_plot  # noqa: E402

from baremetal_llm.foundation import ModernLM
from baremetal_llm.scaling import InstructionCollator, load_instruction_data
from baremetal_llm.utils.checkpoints import save_checkpoint
from baremetal_llm.utils.device import device_label, get_device
from baremetal_llm.utils.paths import part_run_dir


def sft_loss(model, x, y):
    logits, loss, _ = model(x, y)
    if loss is not None:
        return loss
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)


def main():
    args = resolve_mode(part_parser("Part 6").parse_args())
    device = get_device(args.gpu or args.train)
    steps = 40 if args.demo else 200

    banner(f"Part 6: SFT ({device_label(device)})")
    items = load_instruction_data(use_fallback=True)[: 8 if args.demo else 32]
    collator = InstructionCollator(ctx_len=96)
    x, y = collator.collate([(d.prompt, d.response) for d in items])
    x, y = x.to(device), y.to(device)

    model = ModernLM(vocab=256, ctx_len=96, n_layers=2, n_heads=2, dim=64).to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-3)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = sft_loss(model, x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    run_dir = part_run_dir(6, "sft-demo")
    save_loss_plot(losses, run_dir / "loss.png", "SFT")
    ckpt = run_dir / "model_last.pt"
    config = {"vocab": 256, "ctx_len": 96, "n_layers": 2, "n_heads": 2, "dim": 64}
    save_checkpoint(ckpt, model, {"model_type": "modern", "task": "sft", "config": config})
    print(f"final loss {losses[-1]:.4f}")
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
