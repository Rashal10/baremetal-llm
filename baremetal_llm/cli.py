import argparse
import subprocess
import sys
from pathlib import Path

from baremetal_llm.utils.paths import repo_root

PARTS = list(range(1, 10))


def _part_dir(part: int) -> Path:
    return repo_root() / "parts" / f"part_{part}"


def _run_part(part: int, demo: bool, train: bool, gpu: bool) -> int:
    orch = _part_dir(part) / "orchestrator.py"
    if not orch.exists():
        print(f"missing part {part}")
        return 1
    cmd = [sys.executable, str(orch)]
    if demo:
        cmd.append("--demo")
    if train:
        cmd.append("--train")
    if gpu:
        cmd.append("--gpu")
    print(f"\n>> part {part}")
    return subprocess.call(cmd, cwd=orch.parent)


def cmd_demo(args: argparse.Namespace) -> int:
    parts = [args.part] if args.part else PARTS
    code = 0
    for p in parts:
        if _run_part(p, demo=True, train=False, gpu=args.gpu):
            code = 1
    return code


def cmd_train(args: argparse.Namespace) -> int:
    if not args.part:
        print("--part required")
        return 1
    return _run_part(args.part, demo=False, train=True, gpu=args.gpu)


def cmd_generate(args: argparse.Namespace) -> int:
    import torch

    from baremetal_llm.foundation import CharTokenizer, ModernLM, TinyLM
    from baremetal_llm.utils.device import get_device

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"not found: {ckpt}")
        return 1

    device = get_device(args.gpu)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    kind = payload.get("model_type", "modern")
    config = payload.get("config", {"vocab": 256, "ctx_len": 128, "n_layers": 2, "n_heads": 2, "dim": 64})
    model_cls = TinyLM if kind == "tiny" else ModernLM
    model = model_cls(**config)

    model.load_state_dict(payload["model"])
    model.to(device)
    tok = CharTokenizer()
    ids = tok.encode(args.prompt).unsqueeze(0).to(device)
    out = model.generate(ids, max_tokens=args.max_tokens)
    print(tok.decode(out[0]))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="baremetal")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--part", type=int, choices=PARTS)
    demo.add_argument("--cpu", action="store_true")
    demo.add_argument("--gpu", action="store_true")
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train")
    train.add_argument("--part", type=int, choices=PARTS, required=True)
    train.add_argument("--gpu", action="store_true")
    train.set_defaults(func=cmd_train)

    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--prompt", default="Explain attention in transformers.")
    gen.add_argument("--max-tokens", type=int, default=80)
    gen.add_argument("--gpu", action="store_true")
    gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
