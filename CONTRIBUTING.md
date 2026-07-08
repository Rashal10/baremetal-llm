# Contributing

This started as a personal learning project, but issues and PRs are welcome if you find a bug or want to extend it.

## Setup:

```bash
git clone https://github.com/Rashal10/baremetal-llm.git
cd baremetal-llm
pip install -e ".[dev,demo]"
```

## Before opening a PR:

```bash
pytest
python -m ruff check baremetal_llm tests parts
```

Both should pass. If you are adding a new module or function, add a corresponding test in `tests/`.

## Style:

- No trainer abstractions. Keep forward/loss/optimizer steps explicit, that is the whole point of the project.
- Match the existing code style (see `pyproject.toml` for the ruff config).
- Keep new curriculum parts CPU-runnable in under a couple of minutes for the `--demo` path; GPU-only work belongs behind `--train`/`--gpu`.



## Reporting bugs:

Open an issue with the command you ran, the full traceback, and your Python/PyTorch versions. A minimal repro helps a lot. Thank You.