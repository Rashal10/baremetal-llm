# Phase 1: Foundation

This was the starting point. I wanted to understand what is actually inside a transformer before touching any of the fancier stuff, so this phase covers the core building blocks: attention, positional encodings, normalization, and the two model variants I ended up using throughout the project.

`TinyLM` is the simple baseline I kept going back to for quick experiments. It uses standard learned positional embeddings and plain LayerNorm, which made it easier to reason about when something went wrong early on.

`ModernLM` is where things get more interesting: RoPE for position, RMSNorm instead of LayerNorm, SwiGLU feed-forward, and grouped-query attention to cut down on KV memory during generation. Getting the KV cache right here took a few iterations.

The code lives in `baremetal_llm/foundation.py`. These files in `phase_1/` are kept as a reference to show how the modules were first written before they got moved into the package.

```bash
cd phase_1
python orchestrator.py --demo
```
