from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPosEmbed(nn.Module):
    """Learned positional embeddings."""
    def __init__(self, max_seq: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_seq, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        return x + self.emb(positions).unsqueeze(0)

class SinPosEmbed(nn.Module):
    """Sinusoidal positional embeddings."""
    def __init__(self, max_seq: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_seq, dim)
        pos = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)

class RootMeanNorm(nn.Module):
    """RMSNorm layer."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale

class RotaryCache:
    """RoPE cos/sin cache."""
    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0, device=None):
        assert head_dim % 2 == 0, "RoPE requires even head dimension"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_seq)

    def _build(self, max_seq: int):
        self.max_seq = max_seq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(max_seq, device=self.device).float()
        freqs = torch.outer(t, inv_freq)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def get(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_seq:
            self._build(max(need, self.max_seq * 2))
        return self.cos[positions], self.sin[positions]

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

@dataclass
class StateCache:
    """KV cache for generation."""
    k: torch.Tensor
    v: torch.Tensor

def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    m = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, seq_len, seq_len)

class MLP(nn.Module):
    """Feed-forward with GELU."""
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Linear(mult * dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GatedFFN(nn.Module):
    """SwiGLU feed-forward."""
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w_gate = nn.Linear(dim, inner, bias=False)
        self.w_up = nn.Linear(dim, inner, bias=False)
        self.w_down = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.act(self.w_up(x))
        return self.drop(self.w_down(gate * up))

class ScalarAttention(nn.Module):
    """Single head attention."""
    def __init__(self, dim: int, head_dim: int, drop: float = 0.0):
        super().__init__()
        self.wq = nn.Linear(dim, head_dim, bias=False)
        self.wk = nn.Linear(dim, head_dim, bias=False)
        self.wv = nn.Linear(dim, head_dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = causal_mask(T, device=x.device)
        scores = scores.masked_fill(mask.squeeze(1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        return torch.matmul(weights, v), weights

class GroupedAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0, trace: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)
        self.trace = trace

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.drop(w)
        
        ctx = torch.matmul(w, v)
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), w

class AutoregressiveAttn(nn.Module):
    """Causal attention with SDPA."""
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class ModernAttn(nn.Module):
    """Attention with RoPE + GQA + KV cache."""
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0,
                 use_rope: bool = True, max_seq: int = 4096,
                 window: Optional[int] = None, sink: int = 0,
                 n_kv_heads: Optional[int] = None):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.group_size = n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

        self.use_rope = use_rope
        self.rope_cache: Optional[RotaryCache] = None
        self.max_seq = max_seq
        self.window = window
        self.sink = sink

    def _init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RotaryCache(self.head_dim, self.max_seq, device=device)

    def forward(self, x: torch.Tensor, cache: Optional[StateCache] = None, 
                start_pos: int = 0) -> Tuple[torch.Tensor, StateCache]:
        B, T, C = x.shape
        self._init_rope(x.device)

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        if cache is not None:
            k_all = torch.cat([cache.k, k], dim=2)
            v_all = torch.cat([cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # sliding window
        if self.window and k_all.size(2) > (self.window + self.sink):
            s = self.sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.window:, :]], dim=2)

        # gqa
        if self.n_kv_heads != self.n_heads:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)
        else:
            k_attn, v_attn = k_all, v_all

        y = F.scaled_dot_product_attention(
            q, k_attn, v_attn, attn_mask=None,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=(cache is None)
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        if cache is not None:
            k_new = torch.cat([cache.k, k], dim=2)
            v_new = torch.cat([cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = StateCache(k_new, v_new)
        
        return y, new_cache

class EncoderLayer(nn.Module):
    """Transformer block."""
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = GroupedAttention(dim, n_heads, drop)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, mult=4, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    """GPT decoder block."""
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = AutoregressiveAttn(dim, n_heads, drop)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, mult=4, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ModernLayer(nn.Module):
    """Modern transformer block."""
    def __init__(self, dim: int, n_heads: int, drop: float = 0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True,
                 use_rope: bool = True, max_seq: int = 4096,
                 window: Optional[int] = None, sink: int = 0,
                 n_kv_heads: Optional[int] = None):
        super().__init__()
        Norm = RootMeanNorm if use_rmsnorm else nn.LayerNorm
        self.ln1 = Norm(dim)
        self.attn = ModernAttn(dim, n_heads, drop, use_rope, max_seq, window, sink, n_kv_heads)
        self.ln2 = Norm(dim)
        self.ffn = GatedFFN(dim, mult=4, drop=drop) if use_swiglu else MLP(dim, mult=4, drop=drop)

    def forward(self, x: torch.Tensor, cache: Optional[StateCache] = None, 
                start_pos: int = 0) -> Tuple[torch.Tensor, StateCache]:
        a, cache = self.attn(self.ln1(x), cache=cache, start_pos=start_pos)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, cache

class CharTokenizer:
    """Byte-level tokenizer."""
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256

class CharDataLoader:
    """Loads text for training."""
    def __init__(self, path: str, ctx_len: int = 256, train_split: float = 0.9):
        data = Path(path).read_bytes()
        data = torch.tensor(list(data), dtype=torch.long)
        n = int(len(data) * train_split)
        self.train = data[:n]
        self.val = data[n:]
        self.ctx_len = ctx_len

    def get_batch(self, split: str, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.train if split == 'train' else self.val
        assert len(buf) > self.ctx_len + 1
        ix = torch.randint(0, len(buf) - self.ctx_len - 1, (batch_size,))
        x = torch.stack([buf[i:i+self.ctx_len] for i in ix])
        y = torch.stack([buf[i+1:i+1+self.ctx_len] for i in ix])
        return x.to(device), y.to(device)

def top_k_top_p_filter(logits: torch.Tensor, top_k: Optional[int] = None,
                       top_p: Optional[float] = None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        cutoff = logits.topk(min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = float('-inf')
        logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))
    return logits

class TinyLM(nn.Module):
    """Small GPT model."""
    def __init__(self, vocab: int, ctx_len: int, n_layers: int = 4,
                 n_heads: int = 4, dim: int = 256, drop: float = 0.0):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(ctx_len, dim)
        self.drop = nn.Dropout(drop)
        self.layers = nn.ModuleList([DecoderLayer(dim, n_heads, drop) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.ctx_len
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_tokens: int = 200,
                 temp: float = 1.0, top_k: Optional[int] = 50,
                 top_p: Optional[float] = None) -> torch.Tensor:
        self.eval()
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_tokens):
            ctx = idx[:, -self.ctx_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temp, 1e-6)
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

class ModernLM(nn.Module):
    """Modern GPT with RoPE and caching."""
    def __init__(self, vocab: int = 256, ctx_len: int = 256,
                 n_layers: int = 4, n_heads: int = 4, dim: int = 256,
                 drop: float = 0.0, use_rmsnorm: bool = True,
                 use_swiglu: bool = True, use_rope: bool = True,
                 max_seq: int = 4096, window: Optional[int] = None,
                 sink: int = 0, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(drop)
        self.layers = nn.ModuleList([
            ModernLayer(dim, n_heads, drop, use_rmsnorm, use_swiglu,
                       use_rope, max_seq, window, sink, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.Identity() if use_rmsnorm else nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache_list: Optional[list] = None, start_pos: int = 0):
        B, T = idx.shape
        assert T <= self.ctx_len
        x = self.tok_emb(idx)
        x = self.drop(x)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = None if cache_list is None else cache_list[i]
            x, cache = layer(x, cache=cache, start_pos=start_pos)
            new_caches.append(cache)

        x = self.ln_out(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_tokens: int = 200,
                 temp: float = 1.0, top_k: int = 50, top_p: Optional[float] = None,
                 eos_id: Optional[int] = 1) -> torch.Tensor:
        self.eval()
        idx = prompt
        kvs = [None] * len(self.layers)

        for _ in range(max_tokens):
            ctx = idx[:, -self.ctx_len:] if kvs[0] is None else idx[:, -1:]
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

            logits, _, kvs = self(ctx, cache_list=kvs, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temp, 1e-6)
            next_logits = top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temp == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and (next_id == eos_id).all():
                break

        return idx

def _model_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = model(x, y)
    if isinstance(out, tuple):
        return out[1]
    return out

def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> float:
    optimizer.zero_grad()
    loss = _model_loss(model, x, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def estimate_loss(model: nn.Module, data: CharDataLoader, batch_size: int,
                  eval_iters: int, device) -> dict:
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = []
            for _ in range(eval_iters):
                x, y = data.get_batch(split, batch_size, device)
                loss = _model_loss(model, x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()
    return out

if __name__ == '__main__':
    print("Foundation module loaded successfully.")
    print("\nAvailable components:")
    print("  - Positional: LearnedPosEmbed, SinPosEmbed, RotaryCache")
    print("  - Attention: ScalarAttention, GroupedAttention, AutoregressiveAttn, ModernAttn")
    print("  - FFN: MLP, GatedFFN")
    print("  - Normalization: RootMeanNorm")
    print("  - Blocks: EncoderLayer, DecoderLayer, ModernLayer")
    print("  - Models: TinyLM, ModernLM")
    print("  - Utilities: CharTokenizer, CharDataLoader, StateCache")
    
    x = torch.randn(2, 16, 64)
    attn = ModernAttn(64, n_heads=4, use_rope=True)
    out, cache = attn(x)
    print(f"\nShape test: input {x.shape} -> output {out.shape}")
