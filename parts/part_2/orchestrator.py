import sys
from pathlib import Path

import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode, save_loss_plot  # noqa: E402

from baremetal_llm.foundation import CharDataLoader, CharTokenizer, TinyLM, train_step
from baremetal_llm.utils.checkpoints import save_checkpoint
from baremetal_llm.utils.device import device_label, get_device
from baremetal_llm.utils.paths import data_path, part_run_dir


def main():
    args = resolve_mode(part_parser("Part 2").parse_args())
    device = get_device(args.gpu or args.train)
    steps = 80 if args.demo else 400

    banner(f"Part 2: TinyLM ({device_label(device)})")
    data = CharDataLoader(str(data_path()), ctx_len=32)
    model = TinyLM(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for step in range(steps):
        x, y = data.get_batch("train", 8, device)
        losses.append(train_step(model, x, y, opt))
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"step {step + 1}/{steps} loss={losses[-1]:.4f}")

    run_dir = part_run_dir(2, "tiny-lm")
    save_loss_plot(losses, run_dir / "loss.png", "TinyLM")
    ckpt = run_dir / "model.pt"
    config = {"vocab": 256, "ctx_len": 32, "n_layers": 2, "n_heads": 2, "dim": 64}
    save_checkpoint(ckpt, model, {"model_type": "tiny", "config": config})

    tok = CharTokenizer()
    sample = model.generate(tok.encode("Transformers ").unsqueeze(0).to(device), max_tokens=40)
    print("sample:", tok.decode(sample[0]))
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
