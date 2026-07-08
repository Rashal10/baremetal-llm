import torch

from baremetal_llm.alignment import (
    ActorCritic,
    PreferenceScorer,
    bt_loss,
    compute_grpo_loss,
    compute_logprobs,
    compute_ppo_loss,
    load_rankings,
)
from baremetal_llm.foundation import ModernLM, TinyLM


def test_preference_scorer():
    rm = PreferenceScorer(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64)
    x = torch.randint(0, 256, (2, 16))
    scores = rm(x)
    assert scores.shape == (2,)


def test_bt_loss():
    r_pos = torch.tensor([1.0, 0.5])
    r_neg = torch.tensor([0.2, 0.1])
    loss = bt_loss(r_pos, r_neg)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_actor_critic_with_both_lms():
    x = torch.randint(0, 250, (2, 12))
    for factory in (
        lambda: TinyLM(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64),
        lambda: ModernLM(vocab=256, ctx_len=32, n_layers=2, n_heads=2, dim=64),
    ):
        ac = ActorCritic(factory())
        logits, values, loss = ac(x, x)
        assert logits.shape[-1] == 256
        assert values.shape == logits.shape[:2]
        lp = compute_logprobs(ac, x)
        assert lp.shape == (2, 11)


def test_ppo_and_grpo_losses():
    new_lp = torch.randn(4)
    old_lp = torch.randn(4)
    adv = torch.randn(4)
    ppo = compute_ppo_loss(
        new_lp, old_lp, adv,
        torch.randn(4), torch.randn(4), torch.randn(4),
    )
    grpo = compute_grpo_loss(new_lp, old_lp, adv, ent_coef=0.01)
    assert torch.isfinite(ppo.total_loss)
    assert torch.isfinite(grpo.total_loss)


def test_load_rankings_fallback():
    items = load_rankings()
    assert len(items) >= 1
    assert items[0].chosen and items[0].rejected
