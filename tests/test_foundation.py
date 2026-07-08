import tempfile
from pathlib import Path

import torch
import torch.optim as optim

from baremetal_llm.foundation import (
    CharDataLoader,
    CharTokenizer,
    GroupedAttention,
    ModernLM,
    TinyLM,
    estimate_loss,
    train_step,
)


def test_grouped_attention_shape():
    x = torch.randn(2, 8, 64)
    attn = GroupedAttention(64, 4)
    out, weights = attn(x)
    assert out.shape == (2, 8, 64)
    assert weights.shape == (2, 4, 8, 8)


def test_tinylm_forward():
    model = TinyLM(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss.ndim == 0


def test_modernlm_forward_and_generate():
    model = ModernLM(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64)
    x = torch.randint(0, 256, (1, 10))
    logits, loss, caches = model(x, x)
    assert logits.shape == (1, 10, 256)
    assert loss.ndim == 0
    assert len(caches) == 2

    out = model.generate(x, max_tokens=5)
    assert out.shape[1] == 15


def test_train_step_and_estimate_loss():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
        f.write(b"hello world " * 50)
        path = f.name

    try:
        data = CharDataLoader(path, ctx_len=16)
        model = TinyLM(vocab=256, ctx_len=16, n_layers=2, n_heads=2, dim=32)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        x, y = data.get_batch("train", 4, "cpu")
        loss = train_step(model, x, y, opt)
        assert loss > 0

        stats = estimate_loss(model, data, 4, 2, "cpu")
        assert "train" in stats and "val" in stats
    finally:
        Path(path).unlink(missing_ok=True)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer()
    text = "baremetal"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
