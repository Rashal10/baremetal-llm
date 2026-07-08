import sys
from pathlib import Path

import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode, save_loss_plot, timed  # noqa: E402

from baremetal_llm.foundation import CharDataLoader, CharTokenizer, ModernLM, train_step
from baremetal_llm.utils.checkpoints import save_checkpoint
from baremetal_llm.utils.device import device_label, get_device
from baremetal_llm.utils.paths import data_path, part_run_dir


def main():
    args = resolve_mode(part_parser("Part 3").parse_args())
    device = get_device(args.gpu or args.train)
    steps = 60 if args.demo else 300

    banner(f"Part 3: ModernLM ({device_label(device)})")
    data = CharDataLoader(str(data_path()), ctx_len=32)
    model = ModernLM(vocab=256, ctx_len=64, n_layers=2, n_heads=2, dim=64).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(steps):
        x, y = data.get_batch("train", 8, device)
        losses.append(train_step(model, x, y, opt))

    run_dir = part_run_dir(3, "modern-lm")
    save_loss_plot(losses, run_dir / "loss.png", "ModernLM")
    config = {"vocab": 256, "ctx_len": 64, "n_layers": 2, "n_heads": 2, "dim": 64}
    save_checkpoint(run_dir / "model.pt", model, {"model_type": "modern", "config": config})

    tok = CharTokenizer()
    prompt = tok.encode("Attention ").unsqueeze(0).to(device)
    model.eval()
    with timed("cached generate"):
        model.generate(prompt, max_tokens=32)
    print(f"saved {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
