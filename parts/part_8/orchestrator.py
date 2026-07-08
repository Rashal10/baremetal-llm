import sys
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import banner, part_parser, resolve_mode  # noqa: E402

from baremetal_llm.alignment import ActorCritic, compute_logprobs, compute_ppo_loss
from baremetal_llm.foundation import ModernLM
from baremetal_llm.utils.checkpoints import save_checkpoint
from baremetal_llm.utils.device import device_label, get_device
from baremetal_llm.utils.paths import part_run_dir


def main():
    args = resolve_mode(part_parser("Part 8").parse_args())
    device = get_device(args.gpu or args.train)
    steps = 5 if args.demo else 20

    banner(f"Part 8: PPO ({device_label(device)})")
    lm = ModernLM(vocab=256, ctx_len=48, n_layers=2, n_heads=2, dim=64).to(device)
    policy = ActorCritic(lm).to(device)
    opt = optim.AdamW(policy.parameters(), lr=1e-4)
    x = torch.randint(0, 250, (4, 32), device=device)

    for step in range(steps):
        with torch.no_grad():
            old_logp = compute_logprobs(policy, x)
            old_val = policy(x)[1]
            returns = old_val + 0.1
            adv = returns - old_val

        logits, values, _ = policy(x, x)
        labels = x[:, 1:].contiguous()
        new_logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        result = compute_ppo_loss(new_logp, old_logp, adv[:, 1:], values[:, 1:], old_val[:, 1:], returns[:, 1:])

        opt.zero_grad()
        result.total_loss.backward()
        opt.step()
        print(f"step {step + 1} policy={result.policy_loss.item():.4f} kl={result.approx_kl.item():.4f}")

    run_dir = part_run_dir(8, "ppo-demo")
    save_checkpoint(run_dir / "model.pt", policy, {"model_type": "ppo"})
    print(f"saved {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
