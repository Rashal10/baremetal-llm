from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
    _HAS_TOKENIZERS = True
except ImportError:
    ByteLevelBPETokenizer = None
    Tokenizer = None
    _HAS_TOKENIZERS = False

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    load_dataset = None
    _HAS_DATASETS = False

class SubwordTokenizer:
    """BPE tokenizer wrapper."""
    def __init__(self, vocab: int = 32000, specials: Optional[List[str]] = None):
        if not _HAS_TOKENIZERS:
            raise ImportError("Run: pip install tokenizers")
        self.vocab_size = vocab
        self.specials = specials or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: Union[str, Path]):
        p = Path(data_path)
        files = [str(fp) for fp in p.glob("**/*.txt")] if p.is_dir() else [str(p)]
        tok = ByteLevelBPETokenizer()
        tok.train(files=files, vocab_size=self.vocab_size, min_frequency=2, special_tokens=self.specials)
        self._tok = tok

    def save(self, out_dir: Union[str, Path]):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "Train or load first"
        self._tok.save_model(str(out))
        self._tok.save(str(out / "tokenizer.json"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.specials}
        (out / "bpe_meta.json").write_text(json.dumps(meta))

    def load(self, dir_path: Union[str, Path]):
        dirp = Path(dir_path)
        tokenizer = dirp / "tokenizer.json"
        if not tokenizer.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {dirp}")
        self._tok = Tokenizer.from_file(str(tokenizer))
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.specials = meta.get("special_tokens", self.specials)

    def _require_trained(self):
        if self._tok is None:
            raise RuntimeError("call train() or load() first")

    def encode(self, text: str) -> List[int]:
        self._require_trained()
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        self._require_trained()
        return self._tok.decode(ids)

class CosineScheduler:
    """Warmup + cosine decay."""
    def __init__(self, optimizer, warmup: int, total: int, base_lr: float):
        self.optim = optimizer
        self.warmup = max(1, warmup)
        self.total = max(warmup + 1, total)
        self.base = base_lr
        self.step_n = 0

    def step(self) -> float:
        self.step_n += 1
        if self.step_n <= self.warmup:
            lr = self.base * self.step_n / self.warmup
        else:
            progress = (self.step_n - self.warmup) / (self.total - self.warmup)
            lr = 0.5 * self.base * (1.0 + math.cos(math.pi * progress))
        for g in self.optim.param_groups:
            g['lr'] = lr
        return lr

class MixedPrecisionGrad:
    """AMP + grad accumulation."""
    def __init__(self, optimizer, accum: int = 1, use_amp: bool = True):
        self.optim = optimizer
        self.accum = max(1, accum)
        self.amp = use_amp and torch.cuda.is_available()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self._count = 0

    def backward(self, loss: torch.Tensor):
        scaled = loss / self.accum
        if self.amp:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()
        self._count += 1

    def should_step(self) -> bool:
        return (self._count % self.accum) == 0

    def step(self):
        if self.amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad(set_to_none=True)

class TokenBuffer(Dataset):
    """Text as token tensor."""
    def __init__(self, path: str, tokenizer: SubwordTokenizer, ctx_len: int = 256):
        super().__init__()
        self.ctx_len = ctx_len
        text = Path(path).read_text(encoding='utf-8')
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return max(0, self.ids.numel() - self.ctx_len - 1)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[i:i + self.ctx_len]
        y = self.ids[i + 1:i + self.ctx_len + 1]
        return x, y

def create_dataloader(path: str, tokenizer: SubwordTokenizer, ctx_len: int,
                      batch: int, shuffle: bool = True) -> DataLoader:
    ds = TokenBuffer(path, tokenizer, ctx_len)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True)

class ExpertRouter(nn.Module):
    """Top-k gating."""
    def __init__(self, dim: int, n_experts: int, k: int = 1):
        super().__init__()
        assert 1 <= k <= n_experts
        self.n = n_experts
        self.k = k
        self.gate = nn.Linear(dim, n_experts, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, k=self.k, dim=-1)

        # load balancing
        S, E = probs.size(0), probs.size(1)
        importance = probs.mean(dim=0)
        primary = top_idx[:, 0]
        load = torch.zeros(E, device=x.device)
        load.scatter_add_(0, primary, torch.ones_like(primary, dtype=load.dtype))
        load = load / max(S, 1)
        aux_loss = E * (importance * load).sum()

        return top_idx, top_vals, aux_loss

class ExpertFFN(nn.Module):
    """Single expert FFN."""
    def __init__(self, dim: int, mult: int = 4, use_swiglu: bool = True, drop: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.w1 = nn.Linear(dim, inner, bias=False)
            self.w2 = nn.Linear(dim, inner, bias=False)
            self.w3 = nn.Linear(inner, dim, bias=False)
            self.act = nn.SiLU()
            self.drop = nn.Dropout(drop)
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, inner), nn.GELU(),
                nn.Linear(inner, dim), nn.Dropout(drop)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            a = self.w1(x)
            b = self.act(self.w2(x))
            return self.drop(self.w3(a * b))
        return self.ff(x)

class MixtureOfExperts(nn.Module):
    """Sparse MoE layer."""
    def __init__(self, dim: int, n_experts: int, k: int = 1,
                 mult: int = 4, use_swiglu: bool = True, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        self.router = ExpertRouter(dim, n_experts, k)
        self.experts = nn.ModuleList([
            ExpertFFN(dim, mult, use_swiglu, drop) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        S = B * T
        flat = x.reshape(S, C)
        idx, weights, aux = self.router(flat)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            for slot in range(self.k):
                sel = (idx[:, slot] == e)
                if sel.any():
                    expert_out = self.experts[e](flat[sel])
                    out[sel] += weights[sel, slot:slot+1] * expert_out

        return out.view(B, T, C), aux

class HybridFFN(nn.Module):
    """Dense + MoE blend."""
    def __init__(self, dim: int, alpha: float = 0.5, mult: int = 4,
                 use_swiglu: bool = True, n_experts: int = 4, k: int = 1, drop: float = 0.0):
        super().__init__()
        self.alpha = alpha
        inner = mult * dim
        self.dense = nn.Sequential(
            nn.Linear(dim, inner), nn.GELU(),
            nn.Linear(inner, dim), nn.Dropout(drop)
        )
        self.moe = MixtureOfExperts(dim, n_experts, k, mult, use_swiglu, drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_dense = self.dense(x)
        y_moe, aux = self.moe(x)
        return self.alpha * y_dense + (1.0 - self.alpha) * y_moe, aux

PROMPT_TEMPLATE = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

@dataclass
class InstructionExample:
    instruction: str
    response: str

def fmt_example(ex: InstructionExample) -> str:
    return PROMPT_TEMPLATE.format(instruction=ex.instruction.strip(), response=ex.response.strip())

def fmt_prompt_only(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip(), response="")

@dataclass
class SFTItem:
    prompt: str
    response: str

def load_instruction_data(split: str = "train[:200]", use_fallback: bool = False) -> List[SFTItem]:
    items: List[SFTItem] = []
    if _HAS_DATASETS and not use_fallback:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split=split)
            for row in ds:
                instr = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()
                out = row.get("output", "").strip()
                if inp:
                    instr = instr + "\n" + inp
                if instr and out:
                    items.append(SFTItem(prompt=instr, response=out))
        except Exception:
            pass
    if not items:

        samples = [
            ("What is the first prime number?", "2"),
            ("What are the three primary colors?", "Red, blue, and yellow"),
            ("Name a device that points to magnetic north", "A compass"),
        ]
        items = [SFTItem(prompt=p, response=r) for p, r in samples]
    return items

class ProgressiveCurriculum:
    """Short to long ordering."""
    def __init__(self, items: List[Tuple[str, str]]):
        self.items = sorted(items, key=lambda p: len(p[0]))
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self) -> Tuple[str, str]:
        if self._i >= len(self.items):
            raise StopIteration
        it = self.items[self._i]
        self._i += 1
        return it

class InstructionCollator:
    """Tokenize and mask labels."""
    def __init__(self, ctx_len: int = 256, tokenizer=None):
        self.ctx_len = ctx_len
        self.tok = tokenizer
        if self.tok is None:
            self.tok = _ByteTokenizer()
        self.vocab_size = getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def collate(self, batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = [], []
        for prompt, response in batch:
            prefix = fmt_prompt_only(prompt).replace('</s>', '')
            full = fmt_example(InstructionExample(prompt, response))
            ids = self.encode(full)[:self.ctx_len]
            prompt_ids = self.encode(prefix)[:self.ctx_len]
            n_prompt = min(len(prompt_ids), len(ids))

            x = ids
            y = ids.copy()
            for t in range(len(y) - 1):
                y[t] = ids[t + 1]
            y[-1] = -100
            for i in range(n_prompt - 1):
                y[i] = -100
            input_ids.append(x)
            labels.append(y)

        def pad(seq, val):
            if len(seq) < self.ctx_len:
                seq = seq + [val] * (self.ctx_len - len(seq))
            return seq[:self.ctx_len]

        x = torch.tensor([pad(s, 2) for s in input_ids], dtype=torch.long)
        y = torch.tensor([pad(s, -100) for s in labels], dtype=torch.long)
        return x, y

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

if __name__ == '__main__':
    print("Scaling module loaded successfully.")
    print("\nAvailable components:")
    print("  - Tokenization: SubwordTokenizer")
    print("  - Scheduling: CosineScheduler, MixedPrecisionGrad")
    print("  - MoE: ExpertRouter, ExpertFFN, MixtureOfExperts, HybridFFN")
    print("  - SFT: InstructionCollator, ProgressiveCurriculum, load_instruction_data")

    moe = MixtureOfExperts(dim=64, n_experts=4, k=2)
    x = torch.randn(2, 16, 64)
    out, aux = moe(x)
    print(f"\nMoE test: input {x.shape} -> output {out.shape}, aux_loss={aux.item():.4f}")
