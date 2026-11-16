"""Decoder-only Transformer language model and scoring helpers."""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, ctx=512, layers=8, d=512, heads=8):
        super().__init__()
        self.cfg = dict(vocab=vocab, ctx=ctx, layers=layers, d=d, heads=heads)
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def n_params(self, with_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        return n if with_embeddings else n - self.tok.weight.numel() - self.pos.weight.numel()


def load_model(run_dir, device="cuda"):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    model = GPT(**cfg["model"])
    sd = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    return model.to(device).eval(), cfg


@torch.no_grad()
def token_nll(model, ids, batch=64, device="cuda"):
    """Negative log-likelihood (nats) of every token of a long id sequence except the
    first one.  Windows of `ctx` tokens with stride ctx/2: each token is scored once,
    with at least ctx/2 tokens of left context whenever that much exists."""
    ids = np.asarray(ids, dtype=np.int64)
    n, ctx = len(ids), model.ctx
    half = ctx // 2
    out = np.zeros(n, dtype=np.float32)
    starts = [0] + list(range(half, max(n - ctx, 0) + half, half)) if n > ctx else [0]
    starts = sorted(set(min(s, max(n - ctx, 0)) for s in starts))
    done = 1          # tokens [0, done) are already scored (token 0 is never scored)
    jobs = []
    for s in starts:
        e = min(s + ctx, n)
        if e <= done:
            continue
        jobs.append((s, e, done))
        done = e
    for i in range(0, len(jobs), batch):
        part = jobs[i:i + batch]
        x = torch.zeros(len(part), ctx, dtype=torch.long)
        for j, (s, e, _) in enumerate(part):
            x[j, :e - s] = torch.from_numpy(ids[s:e])
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        lp = F.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1).cpu().numpy()
        for j, (s, e, d0) in enumerate(part):
            out[d0:e] = -lp[j, d0 - s - 1:e - s - 1]
    return out


@torch.no_grad()
def sequence_logprob(model, seqs, batch=256, device="cuda"):
    """log P of every sequence (each must start with <bos>; scored tokens: all but the first)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res = [0.0] * len(seqs)
    for i in range(0, len(order), batch):
        part = [seqs[j][:model.ctx] for j in order[i:i + batch]]
        m = max(len(p) for p in part)
        x = torch.zeros(len(part), m, dtype=torch.long)
        for j, p in enumerate(part):
            x[j, :len(p)] = torch.tensor(p)
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        lp = F.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        for j, p in enumerate(part):
            res[order[i + j]] = float(lp[j, :len(p) - 1].sum())
    return res
