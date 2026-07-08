# Baremetal LLM

I built this to actually understand how LLMs work under the hood instead of just calling `.fit()` on someone else's trainer. It is a PyTorch implementation of a modern transformer stack: attention, RoPE, MoE, instruction tuning, and RLHF (PPO/GRPO), all written from scratch with plain `nn.Module` code.

It is organized as a nine-part curriculum, so you can run through it piece by piece: start with a basic attention visualization, end with a policy trained via GRPO.

[CI](https://github.com/Rashal10/baremetal-llm/actions/workflows/ci.yml)
[docs](https://rashal10.github.io/baremetal-llm/)
[license](LICENSE)

- **Interactive Demo:** [Google Colab Notebook](https://colab.research.google.com/github/Rashal10/baremetal-llm/blob/main/notebooks/01_quick_demo.ipynb) (Runs instantly online for free)
- **Docs:** [rashal10.github.io/baremetal-llm](https://rashal10.github.io/baremetal-llm/)
- **Colab:** [01_quick_demo.ipynb](notebooks/01_quick_demo.ipynb)



## What is in here


| Area       | What it covers                                                                            |
| ---------- | ----------------------------------------------------------------------------------------- |
| Foundation | Causal attention, RoPE, RMSNorm, SwiGLU, grouped-query attention, KV cache                |
| Scaling    | BPE tokenizer, Mixture-of-Experts routing, gradient accumulation, instruction fine-tuning |
| Alignment  | Reward modeling (Bradley-Terry), PPO, GRPO                                                |


Everything is deliberately small-scale (byte or char-level, a few million params) so it runs on a laptop CPU in a couple of minutes per part. The point was never to compete with a real foundation model, it was to get every piece of the pipeline right by hand.

## Getting started

```bash
git clone https://github.com/Rashal10/baremetal-llm.git
cd baremetal-llm
pip install -e ".[dev,demo]"
```

Run a single lesson:

```bash
python -m baremetal_llm.cli demo --part 2
```

Or run the whole thing (about 10 minutes on CPU):

```bash
python -m baremetal_llm.cli demo --cpu
```

There is also a small Gradio app if you would rather click through it:

```bash
# Run local server
python demos/app.py

# Or run it and generate a free, temporary public link to share with others
python demos/app.py --share
```

And the test suite, if you want to check nothing is broken:

```bash
pytest
```

On Windows, if `baremetal` or `mkdocs` are not recognized as commands, use `python -m baremetal_llm.cli ...` and `python -m mkdocs ...` instead. That is just a PATH thing, not a bug.

## Layout

```
baremetal-llm/
├── baremetal_llm/     core library (foundation, scaling, alignment) + CLI
├── parts/part_1..9/   the curriculum itself, one folder per lesson
├── demos/app.py       the Gradio demo
├── docs/              documentation site (MkDocs)
├── notebooks/         Colab notebooks
└── tests/             pytest suite
```



## The curriculum


| #   | Topic                      | Try it          |
| --- | -------------------------- | --------------- |
| 1   | Attention, visualized      | `demo --part 1` |
| 2   | A tiny char-level LM       | `demo --part 2` |
| 3   | Modern LM plus KV cache    | `demo --part 3` |
| 4   | Training a BPE tokenizer   | `demo --part 4` |
| 5   | Mixture-of-Experts routing | `demo --part 5` |
| 6   | Supervised fine-tuning     | `demo --part 6` |
| 7   | Reward modeling            | `demo --part 7` |
| 8   | PPO                        | `demo --part 8` |
| 9   | GRPO                       | `demo --part 9` |


More detail on each in [docs/curriculum.md](docs/curriculum.md).

## Deploying your own copy

If you want to run this yourself, or fork it, enable GitHub Pages (Settings, Pages, Build with GitHub Actions) for the docs site.

## Contributing

This is a small, personal project, but PRs and issues are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).