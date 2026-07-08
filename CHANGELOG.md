# Changelog

## [1.0.0] - 2026-07

First complete pass. The project moved from a set of standalone phase scripts to an installable package with a real curriculum, tests, docs, and a live demo.

Added
- `baremetal_llm` package with a `baremetal` CLI (`demo`, `train`, `generate`)
- Nine runnable curriculum parts (`parts/part_1` ... `part_9`), each with a fast CPU `--demo` path
- Pytest suite covering foundation, scaling, alignment, and the CLI
- GitHub Actions CI (lint + tests)
- MkDocs documentation site, deployed via GitHub Pages
- Gradio demo app (`demos/app.py`) with Generate / Train / Attention / MoE / Align tabs
- Hugging Face Spaces bundle (`space/`)
- Colab notebooks for a no-install quick start
- MIT license, this changelog, contributing guide

Fixed
- `ModernLM` output head had its dimensions swapped (`nn.Linear(vocab, dim)` instead of `nn.Linear(dim, vocab)`), which silently broke training
- `estimate_loss` assumed every model returned a 2-tuple; `ModernLM` returns 3
- `ActorCritic` could not wrap `TinyLM` because it assumed a fixed forward signature
- `SubwordTokenizer.encode()`/`decode()` failed with a confusing error if called before `train()`/`load()`
- Deprecated `torch.cuda.amp.GradScaler` usage, now uses `torch.amp.GradScaler("cuda", ...)` with a fallback for older PyTorch
- `baremetal generate` and the Gradio demo hardcoded `ctx_len=128`, so loading any checkpoint trained with a different context length (parts 2, 3, 6 all use different values) crashed with a state-dict shape mismatch. Checkpoints now carry their model config and both call sites rebuild the model from it

## [0.1.0] - 2026-01

Initial commit, phase-based scripts for foundation, scaling, and alignment concepts.
