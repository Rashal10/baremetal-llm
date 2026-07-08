import subprocess
import sys

import pytest

from baremetal_llm.foundation import TinyLM
from baremetal_llm.utils.checkpoints import save_checkpoint


def test_generate_uses_checkpoint_config(tmp_path):
    # Regression test: generate used to hardcode ctx_len=128, which broke
    # loading any checkpoint trained with a different context length.
    config = {"vocab": 256, "ctx_len": 32, "n_layers": 1, "n_heads": 2, "dim": 32}
    model = TinyLM(**config)
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, model, {"model_type": "tiny", "config": config})

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "baremetal_llm.cli",
            "generate",
            "--checkpoint",
            str(ckpt),
            "--prompt",
            "hi",
            "--max-tokens",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "baremetal_llm.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "demo" in result.stdout


@pytest.mark.slow
def test_part1_demo():
    result = subprocess.run(
        [sys.executable, "orchestrator.py", "--demo"],
        cwd="parts/part_1",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "attention_heatmap.png" in result.stdout
