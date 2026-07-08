---
title: Baremetal LLM
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: From-scratch PyTorch LLM stack with MoE, SFT & RLHF
---

# Baremetal LLM

Interactive demo for the [baremetal-llm](https://github.com/Rashal10/baremetal-llm) project.

**Workflow:** open the **Train** tab → then **Generate**. Other tabs visualize attention, MoE routing, and alignment.

| Tab | Description |
|-----|-------------|
| Generate | Text completion from a checkpoint |
| Train | Quick TinyLM training on CPU |
| Attention | Causal attention heatmap |
| MoE | Expert routing histogram |
| Align | Base vs. SFT comparison |

[Documentation](https://rashal10.github.io/baremetal-llm/) · [Source code](https://github.com/Rashal10/baremetal-llm)
