from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    load_dataset = None
    _HAS_DATASETS = False

PROMPT_TEMPLATE = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

@dataclass
class Example:
    instruction: str
    response: str

def fmt_example(ex: Example) -> str:
    return PROMPT_TEMPLATE.format(instruction=ex.instruction.strip(), response=ex.response.strip())

def fmt_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip(), response="")

class RLTokenizer:
    """Tokenizer for RLHF."""
    def __init__(self, ctx_len: int = 256):
        self.ctx_len = ctx_len
        self._tok = _ByteTokenizer()

    @property
    def vocab_size(self) -> int:
        return getattr(self._tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        ids = self._tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

class _ByteTokenizer:
    """Byte tokenizer."""
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256

@dataclass
class RankingExample:
    prompt: str
    chosen: str
    rejected: str

def load_rankings(split: str = "train[:200]") -> List[RankingExample]:
    items: List[RankingExample] = []
    if _HAS_DATASETS:
        try:
            ds = load_dataset("Anthropic/hh-rlhf", split=split)
            for row in ds:
                ch = str(row.get("chosen", "")).strip()
                rj = str(row.get("rejected", "")).strip()
                if ch and rj:
                    items.append(RankingExample(prompt="", chosen=ch, rejected=rj))
        except Exception:
            pass
    if not items:
        items = [
            RankingExample(
                "Explain scaling laws for neural LMs.",
                "Performance improves predictably with model size, data, and compute.",
                "Scaling means making images bigger."
            ),
            RankingExample(
                "Give two uses of attention.",
                "It focuses on relevant tokens and enables parallel context integration.",
                "It remembers all past words exactly."
            ),
        ]
    return items

def sample_prompts(n: int) -> List[str]:
    if _HAS_DATASETS:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split="train[:24]")
            arr = []
            for r in ds:
                inst = (r.get('instruction') or '').strip()
                inp = (r.get('input') or '').strip()
                if inp:
                    inst = inst + "\n" + inp
                if inst:
                    arr.append(inst)
                if len(arr) >= n:
                    break
            if arr:
                return arr
        except Exception:
            pass

    base = [
        "Explain attention in transformers.",
        "Give pros and cons of BPE tokenization.",
        "Why is PPO used in RLHF?",
        "Write a Python function to reverse a list.",
    ]
    return (base * ((n + len(base) - 1) // len(base)))[:n]

class PreferenceBatcher:
    """Batch preference pairs."""
    def __init__(self, ctx_len: int = 256, tokenizer=None):
        self.ctx_len = ctx_len
        self.tok = tokenizer or RLTokenizer(ctx_len)
        self.vocab_size = getattr(self.tok, 'vocab_size', 256)

    def _encode(self, text: str) -> List[int]:
        return self.tok.encode(text)[:self.ctx_len]

    def collate(self, batch: List[Tuple[str, str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_ids, neg_ids = [], []
        for prompt, chosen, rejected in batch:
            pos_text = fmt_example(Example(prompt, chosen))
            neg_text = fmt_example(Example(prompt, rejected))
            pos_ids.append(self._encode(pos_text))
            neg_ids.append(self._encode(neg_text))

        def pad(x, val=2):
            if len(x) < self.ctx_len:
                x = x + [val] * (self.ctx_len - len(x))
            return x[:self.ctx_len]

        pos = torch.tensor([pad(x) for x in pos_ids], dtype=torch.long)
        neg = torch.tensor([pad(x) for x in neg_ids], dtype=torch.long)
        return pos, neg

class PreferenceScorer(nn.Module):
    """Reward model."""
    def __init__(self, vocab: int, ctx_len: int, n_layers: int = 4,
                 n_heads: int = 4, dim: int = 256, drop: float = 0.1):
        super().__init__()
        self.vocab_size = vocab
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx_len, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4*dim,
            dropout=drop, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == 2)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)

def bt_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(r_pos - r_neg)).mean()

def margin_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    y = torch.ones_like(r_pos)
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)

def _lm_forward(lm: nn.Module, x: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
    out = lm(x, targets)
    if isinstance(out, tuple):
        if len(out) == 3:
            return out[0], out[1], out[2]
        return out[0], out[1], None
    return out, None, None

def _lm_vocab_size(lm: nn.Module) -> int:
    if hasattr(lm, 'head') and hasattr(lm.head, 'out_features'):
        return lm.head.out_features
    return getattr(lm, 'vocab_size', 256)

class ActorCritic(nn.Module):
    """Policy + value head."""
    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm
        vocab = _lm_vocab_size(lm)
        self.val_head = nn.Linear(vocab, 1, bias=False)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        logits, loss, _caches = _lm_forward(self.lm, x, targets)
        values = self.val_head(logits).squeeze(-1)
        return logits, values, loss

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)

def shift_labels(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:].contiguous()

def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def compute_logprobs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, 'lm'):
        logits, _, _ = _lm_forward(model.lm, x, None)
    else:
        logits, _, _ = _lm_forward(model, x, None)
    labels = shift_labels(x)
    return gather_logprobs(logits[:, :-1, :], labels)

def approx_kl(policy_lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
    return (policy_lp - ref_lp).mean()

@dataclass
class PPOLossResult:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    total_loss: torch.Tensor

def compute_ppo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor,
                     adv: torch.Tensor, new_values: torch.Tensor,
                     old_values: torch.Tensor, returns: torch.Tensor,
                     clip_ratio: float = 0.2, vf_coef: float = 0.5,
                     ent_coef: float = 0.0) -> PPOLossResult:

    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    value_loss = F.mse_loss(new_values, returns)
    entropy = -new_logp.mean()
    kl = torch.mean(old_logp - new_logp)

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return PPOLossResult(policy_loss, value_loss, entropy, kl, total)

@dataclass
class GRPOLossResult:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    kl_ref: torch.Tensor
    total_loss: torch.Tensor

def compute_grpo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor,
                      adv: torch.Tensor, clip_ratio: float = 0.2,
                      ent_coef: float = 0.0, kl_coef: float = 0.0,
                      kl_mean: Optional[torch.Tensor] = None) -> GRPOLossResult:

    device = new_logp.device
    if new_logp.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return GRPOLossResult(zero, zero, zero, zero, zero)

    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    entropy = -new_logp.mean() if ent_coef != 0.0 else new_logp.new_tensor(0.0)
    approx_kl = torch.mean(old_logp - new_logp)
    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)

    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref
    return GRPOLossResult(policy_loss, entropy, approx_kl, kl_ref, total)

if __name__ == '__main__':
    print("Alignment module loaded successfully.")
    print("\nAvailable components:")
    print("  - Data: RankingExample, load_rankings, sample_prompts, PreferenceBatcher")
    print("  - Reward: PreferenceScorer, bt_loss, margin_loss")
    print("  - Policy: ActorCritic, compute_logprobs, approx_kl")
    print("  - PPO: compute_ppo_loss, PPOLossResult")
    print("  - GRPO: compute_grpo_loss, GRPOLossResult")

    rm = PreferenceScorer(vocab=256, ctx_len=64)
    x = torch.randint(0, 256, (2, 32))
    r = rm(x)
    print(f"\nReward model test: input {x.shape} -> rewards {r.shape}")
