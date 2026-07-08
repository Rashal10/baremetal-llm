# Phase 2: Scaling

Once I had a working transformer, the next thing I wanted to figure out was how you actually scale it: better tokenization, conditional computation, and instruction tuning.

The BPE tokenizer here is trained from scratch using the HuggingFace `tokenizers` library. It is not the fastest approach but it made the internals obvious. I kept a small offline corpus (`baremetal_llm/data/tiny_corpus.txt`) so the tokenizer training and demo runs work without any network access.

The Mixture-of-Experts layer was the most interesting part of this phase. The routing logic took a few tries to get right, especially the auxiliary load-balancing loss that keeps tokens from all going to the same expert.

SFT (supervised fine-tuning) is also here. The `InstructionCollator` handles the label masking so that loss is only computed on the response tokens, not the prompt.

```bash
cd phase_2
python orchestrator.py --demo
```
