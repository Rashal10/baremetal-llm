# Getting Started

## Requirements

- Python 3.10+
- PyTorch 2.2+
- About 2 GB of RAM is enough for the CPU demos, no GPU required to try things out

## Installation

```bash
git clone https://github.com/Rashal10/baremetal-llm.git
cd baremetal-llm
pip install -e ".[dev,demo]"
```

!!! note "Windows"
    If `baremetal` or `mkdocs` is not found, use the module form:

    ```bash
    python -m baremetal_llm.cli demo --part 2
    python -m mkdocs serve
    ```

## Command-line interface

### Run one lesson

```bash
python -m baremetal_llm.cli demo --part 2
```

### Run the full curriculum

```bash
python -m baremetal_llm.cli demo --cpu
```

Approximately ten minutes on CPU. Artifacts are saved under `parts/part_N/runs/`.

### Longer training

```bash
python -m baremetal_llm.cli train --part 6
python -m baremetal_llm.cli train --part 6 --gpu   # when CUDA is available
```

### Text generation

```bash
python -m baremetal_llm.cli generate \
  --checkpoint parts/part_2/runs/tiny-lm/model.pt \
  --prompt "Transformers "
```

## Interactive demo

```bash
python demos/app.py
```

Open **http://127.0.0.1:7860**. Recommended flow:

1. **Train**: create a checkpoint
2. **Generate**: sample text from it
3. **Attention** / **MoE**: inspect internals
4. **Align**: compare before and after SFT

## Phase orchestrators

```bash
cd phase_1 && python orchestrator.py --demo   # parts 1–3
cd phase_2 && python orchestrator.py --demo   # parts 4–6
cd phase_3 && python orchestrator.py --demo   # parts 7–9
```

## Quality checks

```bash
pytest
python -m ruff check baremetal_llm tests parts
```

## Documentation site

```bash
python -m mkdocs serve
```

Browse at [http://127.0.0.1:8000](http://127.0.0.1:8000). Deployed copy: [rashal10.github.io/baremetal-llm](https://rashal10.github.io/baremetal-llm/).
