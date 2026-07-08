import torch

from baremetal_llm.scaling import (
    InstructionCollator,
    MixtureOfExperts,
    SubwordTokenizer,
    load_instruction_data,
)


def test_moe_forward():
    moe = MixtureOfExperts(dim=64, n_experts=4, k=2)
    x = torch.randn(2, 8, 64)
    out, aux = moe(x)
    assert out.shape == x.shape
    assert aux.ndim == 0


def test_instruction_collator_masks_prompt():
    collator = InstructionCollator(ctx_len=64)
    batch = [("What is 2+2?", "4")]
    x, y = collator.collate(batch)
    assert x.shape == (1, 64)
    assert (y[0] == -100).any()


def test_load_instruction_data_fallback():
    items = load_instruction_data(use_fallback=True)
    assert len(items) >= 1
    assert items[0].prompt and items[0].response


def test_subword_tokenizer_requires_training():
    tok = SubwordTokenizer(vocab=256)
    try:
        tok.encode("hello")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
