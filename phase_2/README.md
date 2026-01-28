# Phase 2: Scaling & Specialization

BPE tokenization, Mixture-of-Experts, and instruction fine-tuning.

## Components

- **SubwordTokenizer** - BPE tokenization via HuggingFace tokenizers
- **CosineScheduler** - Warmup + cosine decay learning rate
- **MixedPrecisionGrad** - AMP with gradient accumulation
- **MixtureOfExperts** - Sparse MoE with top-k routing
- **InstructionCollator** - Formats prompt/response pairs for SFT

## Quick Start

```python
from scaling import MixtureOfExperts, CosineScheduler
import torch

moe = MixtureOfExperts(dim=256, n_experts=8, k=2)
x = torch.randn(4, 32, 256)
out, aux_loss = moe(x)
```

## SFT Example

```python
from scaling import InstructionCollator, load_instruction_data

data = load_instruction_data()
collator = InstructionCollator(ctx_len=256)
batch = [(d.prompt, d.response) for d in data[:4]]
x, y = collator.collate(batch)
```
