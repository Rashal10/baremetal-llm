# API Overview

Core public surface of the `baremetal_llm` package.

## Foundation (`baremetal_llm.foundation`)

| Class / function | Role |
|------------------|------|
| `TinyLM` | Baseline GPT with learned positions |
| `ModernLM` | RoPE + RMSNorm + SwiGLU + KV cache |
| `GroupedAttention` | Multi-head causal attention |
| `ModernAttn` | RoPE, GQA, sliding window, cache |
| `CharTokenizer` | Byte-level encode/decode |
| `CharDataLoader` | On-disk byte training batches |
| `train_step` | Single optimization step |
| `estimate_loss` | Train/val loss over N batches |

## Scaling (`baremetal_llm.scaling`)

| Class / function | Role |
|------------------|------|
| `SubwordTokenizer` | BPE train / save / load |
| `MixtureOfExperts` | Top-k sparse FFN routing |
| `InstructionCollator` | SFT prompt masking |
| `CosineScheduler` | Warmup + cosine LR |
| `MixedPrecisionGrad` | AMP + gradient accumulation |

## Alignment (`baremetal_llm.alignment`)

| Class / function | Role |
|------------------|------|
| `PreferenceScorer` | Reward model |
| `ActorCritic` | Policy + value head |
| `bt_loss` | Bradley-Terry preference loss |
| `compute_ppo_loss` | Clipped PPO objective |
| `compute_grpo_loss` | Group-relative policy loss |

## CLI

```bash
python -m baremetal_llm.cli demo --part N
python -m baremetal_llm.cli train --part N [--gpu]
python -m baremetal_llm.cli generate --checkpoint PATH --prompt TEXT
```
