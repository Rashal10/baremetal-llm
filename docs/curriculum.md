# Curriculum

Nine self-contained parts. Each writes artifacts under `parts/part_N/runs/`.

| Part | Topic | Command | Output |
|------|-------|---------|--------|
| 1 | Attention heatmap | `demo --part 1` | `attention_heatmap.png` |
| 2 | Char TinyLM | `demo --part 2` | `model.pt`, loss plot |
| 3 | ModernLM + KV cache | `demo --part 3` | `model.pt` |
| 4 | BPE tokenizer | `demo --part 4` | `tokenizer/` |
| 5 | MoE routing | `demo --part 5` | `expert_routing.png` |
| 6 | Supervised FT | `demo --part 6` | `model_last.pt` |
| 7 | Reward model | `demo --part 7` | `model.pt` |
| 8 | PPO | `demo --part 8` | `model.pt` |
| 9 | GRPO | `demo --part 9` | `model.pt` |

## Phase grouping

- **Phase 1** (parts 1–3): transformer fundamentals
- **Phase 2** (parts 4–6): scaling and specialization
- **Phase 3** (parts 7–9): alignment and RLHF

## Suggested order for learning

1. Run part 1 to visualize what attention does.
2. Run part 2 to train the smallest LM and generate text.
3. Read `baremetal_llm/foundation.py` (`ModernAttn`, `ModernLM`).
4. Run parts 4 through 6 for tokenization, sparsity, and instruction tuning.
5. Run parts 7 through 9 for preference learning and policy optimization.

Training data for offline demos lives in `data/tiny_corpus.txt`.
