# Baremetal LLM

<div class="hero" markdown>

A PyTorch implementation of a modern LLM training stack, written by hand: attention through RoPE and MoE, up through RLHF alignment. No `Trainer` class, no hidden abstractions. If it happens during training, it is readable in the source.

[:octicons-mark-github-16: Source](https://github.com/Rashal10/baremetal-llm){ .md-button }
[:octicons-rocket-16: Get started](getting-started.md){ .md-button .md-button--primary }

</div>

## Why this exists

I wanted to actually understand the pieces that make up a modern LLM stack instead of importing them. So every module here, RoPE, grouped-query attention, MoE routing, PPO, is implemented directly with `torch.nn`, not wrapped from a library. It is split into nine runnable parts so each concept can be tested in isolation before the next one builds on it.

## What is covered

<div class="grid cards" markdown>

-   __Modern architecture__

    RoPE, RMSNorm, SwiGLU, grouped-query attention, KV-cache inference

-   __Nine-part curriculum__

    From an attention heatmap up to PPO and GRPO, each runnable on a CPU in a couple of minutes

-   __Tested__

    Pytest suite + GitHub Actions running on 3.11 and 3.12

-   __Try it without installing anything__

    Gradio demo on Hugging Face Spaces, notebooks on Colab

</div>

## The three phases

| Phase | Focus | Key modules |
|-------|--------|-------------|
| Foundation | Transformer core | `TinyLM`, `ModernLM`, RoPE, KV cache |
| Scaling | Capacity & specialization | BPE tokenizer, MoE, SFT collator |
| Alignment | Human preference | Reward model, PPO, GRPO |

## Quick start

=== "Install"

    ```bash
    git clone https://github.com/Rashal10/baremetal-llm.git
    cd baremetal-llm
    pip install -e ".[dev,demo]"
    ```

=== "Run"

    ```bash
    python -m baremetal_llm.cli demo --part 2
    python -m baremetal_llm.cli demo --cpu
    pytest
    ```

=== "Demo UI"

    ```bash
    python demos/app.py
    ```

!!! tip "If you are reviewing this for a role"
    The fastest path is the [live demo](demo.md), or `python demos/app.py` locally. Train a tiny model in the **Train** tab, then generate from it right away in **Generate**.

## A few decisions worth explaining

1. No trainer abstraction. Every forward/loss/optimizer step is spelled out, on purpose.
2. Complexity ramps up gradually: char-level LM first, then BPE, then MoE, then alignment.
3. Every part writes its own checkpoints and plots under `parts/`, so you can inspect what actually happened.
