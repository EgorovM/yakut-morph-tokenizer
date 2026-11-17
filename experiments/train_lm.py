"""Train a decoder-only Transformer LM on an encoded corpus.

    python experiments/train_lm.py TOK_DIR RUN_DIR [--epochs 10] [--ctx 512] [--seed 0]

All runs share the architecture, the optimiser and the number of passes over the
corpus; only the tokenizer (hence the embedding matrix and the number of steps)
differs.  The best checkpoint by dev loss is kept.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from lm import GPT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tok")
ap.add_argument("run")
ap.add_argument("--epochs", type=float, default=10)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--evals", type=int, default=40, help="number of dev evaluations over the run")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--dropout", type=float, default=0.0)
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.makedirs(a.run, exist_ok=True)
stats = json.load(open(os.path.join(a.tok, "stats.json")))
train = np.load(os.path.join(a.tok, "train.npy"), mmap_mode="r")
dev = np.load(os.path.join(a.tok, "dev.npy")).astype(np.int64)
model = GPT(stats["vocab"], a.ctx, a.layers, a.d, a.heads, a.dropout).cuda()
steps = a.max_steps or int(a.epochs * len(train) / (a.batch * a.ctx))
cfg = {"model": model.cfg, "args": vars(a), "steps": steps, "params": model.n_params(),
       "params_no_emb": model.n_params(False), "train_tokens": int(len(train))}
json.dump(cfg, open(os.path.join(a.run, "config.json"), "w"), indent=1)
print(json.dumps(cfg), flush=True)

decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
cmodel = torch.compile(model)
train_t = torch.from_numpy(np.ascontiguousarray(train).astype(np.int64)).cuda()
n_dev = (len(dev) - 1) // a.ctx
dev_x = torch.from_numpy(dev[:n_dev * a.ctx].reshape(n_dev, a.ctx)).cuda()
dev_y = torch.from_numpy(dev[1:n_dev * a.ctx + 1].reshape(n_dev, a.ctx)).cuda()
offs = torch.arange(a.ctx + 1, device="cuda")


def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    return 1e-5 + 0.5 * (a.lr - 1e-5) * (1 + math.cos(math.pi * (s - a.warmup) / max(steps - a.warmup, 1)))


@torch.no_grad()
def evaluate():
    model.eval()
    tot = 0.0
    for i in range(0, n_dev, 256):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = cmodel(dev_x[i:i + 256])
        tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]), dev_y[i:i + 256].reshape(-1),
                               reduction="sum").item()
    model.train()
    return tot / (n_dev * a.ctx)


gen = torch.Generator(device="cuda").manual_seed(a.seed)
every = max(steps // a.evals, 1)
best, t0, log = 1e9, time.time(), open(os.path.join(a.run, "log.jsonl"), "w")
chars_per_tok = stats["dev"]["chars"] / (stats["dev"]["tokens"] - 1)
for s in range(steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(s)
    start = torch.randint(0, len(train_t) - a.ctx - 1, (a.batch,), device="cuda", generator=gen)
    xy = train_t[start[:, None] + offs]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = cmodel(xy[:, :-1])
    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), xy[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (s + 1) % every == 0 or s + 1 == steps:
        d = evaluate()
        rec = {"step": s + 1, "epoch": (s + 1) / steps * a.epochs, "train_loss": loss.item(), "dev_loss": d,
               "dev_bpc": d / math.log(2) / chars_per_tok, "min": (time.time() - t0) / 60,
               "tok_per_s": (s + 1) * a.batch * a.ctx / (time.time() - t0)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if d < best:
            best = d
            torch.save(model.state_dict(), os.path.join(a.run, "best.pt"))
print("TRAINING_DONE best_dev_loss", best, flush=True)
