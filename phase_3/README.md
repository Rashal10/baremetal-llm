# Phase 3: Alignment

This phase was the most challenging to understand conceptually. The goal is to move from a model that predicts tokens to one that produces outputs humans actually prefer.

The reward model uses Bradley-Terry preference loss on chosen/rejected response pairs. Getting the `ActorCritic` wrapper right was tricky because it needs to share the transformer backbone between the policy head and the value head without breaking the existing model interface.

PPO and GRPO are both implemented here. GRPO was simpler to write since it does not need a separate value network: it normalizes rewards within each group of responses instead of using a learned baseline.

The code lives in `baremetal_llm/alignment.py`.

```bash
cd phase_3
python orchestrator.py --demo
```
