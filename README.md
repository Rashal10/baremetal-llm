# Baremetal LLM

Build a large language model from scratch—no high-level APIs.

## Quick Start

```bash
conda create -n baremetal-llm python=3.11
conda activate baremetal-llm
pip install -r requirements.txt
```

## Phases

| Phase | Theme | Run Command |
|-------|-------|-------------|
| 1 | Foundation & Architecture | `cd phase_1 && python orchestrator.py --demo` |
| 2 | Scaling & Specialization | `cd phase_2 && python orchestrator.py --demo` |
| 3 | Alignment & RLHF | `cd phase_3 && python orchestrator.py --demo` |

## Curriculum

**Phase 1** — Core transformer, training from scratch, modern architecture (RoPE, RMSNorm, SwiGLU, KV cache)

**Phase 2** — BPE tokenization, gradient accumulation, Mixture-of-Experts, supervised fine-tuning

**Phase 3** — Reward modeling, PPO, GRPO

## Structure

```
baremetal-llm/
├── phase_1/          # Transformer fundamentals
├── phase_2/          # Scaling & MoE
├── phase_3/          # RLHF & alignment
├── requirements.txt
└── README.md
```
