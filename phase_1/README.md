# Phase 1: Foundation & Architecture

Core transformer building blocks and training.

## Components

- **Positional Embeddings** - Learned and sinusoidal variants
- **Attention** - Single-head, multi-head, causal with RoPE
- **Feed-Forward** - Standard MLP and SwiGLU gated variant  
- **Normalization** - LayerNorm and RMSNorm
- **Models** - `TinyLM` (basic GPT) and `ModernLM` (RoPE + RMSNorm + KV cache)

## Quick Start

```python
from foundation import ModernLM, CharTokenizer

model = ModernLM(vocab_size=256, block_size=128)
tok = CharTokenizer()

ids = tok.encode("Hello world")
out = model.generate(ids.unsqueeze(0), max_new=20)
print(tok.decode(out[0]))
```

## Training

```python
from foundation import CharDataLoader, train_step

loader = CharDataLoader("data.txt", block_size=64, batch_size=16)
# ... training loop
```
