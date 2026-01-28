# Phase 3: Alignment & RLHF

Reward modeling and reinforcement learning from human feedback.

## Components

- **PreferenceScorer** - Reward model for ranking responses
- **ActorCritic** - Policy with value head for PPO
- **compute_ppo_loss** - Clipped surrogate objective
- **compute_grpo_loss** - Value-free policy optimization

## Quick Start

```python
from alignment import PreferenceScorer, bt_loss
import torch

rm = PreferenceScorer(vocab=256, ctx_len=128)
pos = torch.randint(0, 256, (4, 64))
neg = torch.randint(0, 256, (4, 64))

r_pos, r_neg = rm(pos), rm(neg)
loss = bt_loss(r_pos, r_neg)
```

## PPO Training

```python
from alignment import compute_ppo_loss

result = compute_ppo_loss(
    new_logp, old_logp, advantages,
    new_values, old_values, returns
)
loss = result.total_loss
```
