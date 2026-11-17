"""Translation with a decoder-only Transformer: <bos> source <sep> target <eos>.

    python experiments/mt.py PAIRS_DIR SAH_TOK_DIR RUN_DIR --dir ru2sah [--init LM_RUN_DIR] [--seed 0]

The Sakha side uses the tokenizer under test, the Russian side a fixed BPE tokenizer
(PAIRS_DIR/tok_ru).  The loss is taken on the target only.  With --init the weights
of a language model trained with the same Sakha tokenizer are loaded and the embedding
matrix is extended with rows for the Russian tokens.  The best epoch by dev loss is
decoded with beam search and scored with chrF++ and BLEU (sacrebleu) on normalised,
lower-cased text.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F

from common import load_tok
from lm import GPT
from yakut_morph_tokenizer.tokenizer import BOS

ap = argparse.ArgumentParser()
ap.add_argument("pairs")
ap.add_argument("tok")
ap.add_argument("run")
ap.add_argument("--dir", required=True, choices=["ru2sah", "sah2ru"])
ap.add_argument("--init", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--beam", type=int, default=4)
ap.add_argument("--max_len", type=int, default=120)
ap.add_argument("--test", default="test", help="name of the test file in PAIRS_DIR")
ap.add_argument("--decode_only", action="store_true", help="reuse RUN_DIR/best.pt")
a = ap.parse_args()
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(a.run, exist_ok=True)

sah_tok, ru_tok = load_tok(a.tok), load_tok(os.path.join(a.pairs, "tok_ru"))
VS, VR = len(sah_tok), len(ru_tok)
SEP, EOS, V = VS + VR, VS + VR + 1, VS + VR + 2
CTX = 2 * a.max_len + 4
src_l, tgt_l = ("ru", "sah") if a.dir == "ru2sah" else ("sah", "ru")


def enc(text, lang):
    return sah_tok.encode(text, True) if lang == "sah" else [i + VS for i in ru_tok.encode(text, True)]


def dec(ids, lang):
    return sah_tok.decode(ids) if lang == "sah" else ru_tok.decode([i - VS for i in ids])


def load(part):
    rows = [json.loads(l) for l in open(os.path.join(a.pairs, part + ".jsonl"), encoding="utf-8")]
    return [(enc(r[src_l], src_l)[:a.max_len], enc(r[tgt_l], tgt_l)[:a.max_len], r[tgt_l]) for r in rows]


train, dev, test = load("train"), load("dev"), load(a.test)
model = GPT(V, CTX, 8, 512, 8, a.dropout)
if a.init:
    sd = torch.load(os.path.join(a.init, "best.pt"), map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in ("tok.weight", "head.weight"):
            own[k][:VS] = v
        elif k == "pos.weight":
            own[k][:min(CTX, len(v))] = v[:CTX]
        else:
            own[k] = v
    model.load_state_dict(own)
model.cuda()
steps = a.epochs * math.ceil(len(train) / a.batch)
cfg = {"args": vars(a), "vocab": V, "sah_vocab": VS, "steps": steps, "pairs": [len(train), len(dev), len(test)],
       "tgt_tokens_per_sent": float(np.mean([len(t) for _, t, _ in test]))}
if not a.decode_only:
    json.dump(cfg, open(os.path.join(a.run, "config.json"), "w"), indent=1)
print(json.dumps(cfg), flush=True)
decay = [p for p in model.parameters() if p.dim() >= 2]
other = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95))


def batchify(items):
    seqs = [[BOS] + s + [SEP] + t + [EOS] for s, t, _ in items]
    m = max(len(q) for q in seqs)
    x = torch.zeros(len(seqs), m, dtype=torch.long)
    y = torch.full((len(seqs), m), -100, dtype=torch.long)
    for j, (q, (s, _, _)) in enumerate(zip(seqs, items)):
        x[j, :len(q)] = torch.tensor(q)
        y[j, len(s) + 1:len(q) - 1] = torch.tensor(q[len(s) + 2:])     # predict target tokens and <eos>
    return x.cuda(), y.cuda()


@torch.no_grad()
def dev_loss():
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(dev), 256):
        x, y = batchify(dev[i:i + 256])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, V), y.view(-1), ignore_index=-100, reduction="sum").item()
        n += int((y >= 0).sum())
    model.train()
    return tot / n


best, step, t0 = 1e9, 0, time.time()
log = open(os.path.join(a.run, "log.jsonl"), "a" if a.decode_only else "w")
for ep in range(0 if a.decode_only else a.epochs):
    order = sorted(range(len(train)), key=lambda i: (len(train[i][0]) + len(train[i][1])) // 8 + random.random())
    batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
    random.shuffle(batches)
    for b in batches:
        lr = a.lr * (step + 1) / 300 if step < 300 else 1e-5 + 0.5 * (a.lr - 1e-5) * (1 + math.cos(math.pi * (step - 300) / max(steps - 300, 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batchify([train[i] for i in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, V), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    d = dev_loss()
    rec = {"epoch": ep + 1, "train_loss": loss.item(), "dev_loss": d, "min": (time.time() - t0) / 60}
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print(json.dumps(rec), flush=True)
    if d < best:
        best = d
        torch.save(model.state_dict(), os.path.join(a.run, "best.pt"))

# ------------------------------------------------------------------ decoding
model.load_state_dict(torch.load(os.path.join(a.run, "best.pt")))
model.eval()
allowed = torch.zeros(V, dtype=torch.bool)
if tgt_l == "sah":
    allowed[3:VS] = True
else:
    allowed[VS + 3:VS + VR] = True
allowed[EOS] = True
allowed = allowed.cuda()


@torch.no_grad()
def next_logp(seqs):
    m = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), m, dtype=torch.long)
    for j, s in enumerate(seqs):
        x[j, :len(s)] = torch.tensor(s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x.cuda())
    last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
    lp = logits[torch.arange(len(seqs), device="cuda"), last].float().masked_fill(~allowed, -1e9)
    return F.log_softmax(lp, -1)


def norm(score, n):
    return score / ((5 + n) / 6) ** 0.6


def translate(items):
    pre = [[BOS] + s + [SEP] for s, _, _ in items]
    lim = [min(a.max_len, int(1.5 * len(s)) + 10) for s, _, _ in items]
    beams = [[((), 0.0)] for _ in items]
    done = [[] for _ in items]
    for t in range(a.max_len + 1):
        flat = [(k, h) for k, bs in enumerate(beams) for h in bs]
        if not flat:
            break
        lp = next_logp([pre[k] + list(h[0]) for k, h in flat])
        top = lp.topk(a.beam, -1)
        tv, ti = top.values.tolist(), top.indices.tolist()
        new = [[] for _ in items]
        for j, (k, (toks, sc)) in enumerate(flat):
            for v, i in zip(tv[j], ti[j]):
                if i == EOS or len(toks) + 1 >= lim[k]:
                    done[k].append((toks if i == EOS else toks + (i,), norm(sc + v, len(toks) + 1)))
                else:
                    new[k].append((toks + (i,), sc + v))
        for k in range(len(items)):
            cand = sorted(new[k], key=lambda h: -h[1])[:a.beam]
            if len(done[k]) >= a.beam:
                bd = max(s for _, s in done[k])
                cand = [h for h in cand if norm(h[1], len(h[0])) > bd]
            beams[k] = cand
    return [max(d, key=lambda h: h[1])[0] if d else () for d in done]


order = sorted(range(len(test)), key=lambda i: len(test[i][0]))
hyps = [""] * len(test)
for i in range(0, len(order), 96):
    idx = order[i:i + 96]
    for j, toks in zip(idx, translate([test[k] for k in idx])):
        hyps[j] = dec(list(toks), tgt_l)
refs = [r for _, _, r in test]
chrf = sacrebleu.metrics.CHRF(word_order=2)
bleu = sacrebleu.metrics.BLEU(tokenize="13a")
sent = np.asarray([chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)])
res = {"chrf": chrf.corpus_score(hyps, [refs]).score, "bleu": bleu.corpus_score(hyps, [refs]).score,
       "sent_chrf": float(sent.mean()), "dev_loss": best, "minutes": (time.time() - t0) / 60}
sfx = "" if a.test == "test" else "_" + a.test
np.save(os.path.join(a.run, f"sent_chrf{sfx}.npy"), sent)
json.dump(hyps, open(os.path.join(a.run, f"hyps{sfx}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
json.dump(res, open(os.path.join(a.run, f"mt{sfx}.json"), "w"), indent=1)
print("MT_DONE", json.dumps(res), flush=True)
