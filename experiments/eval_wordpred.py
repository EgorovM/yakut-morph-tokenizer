"""Decoding task 1: next-word prediction with beam search.

    python experiments/eval_wordpred.py DATA_DIR TOK_DIR RUN_DIR [--n 4000] [--beam 10]

For a position inside a test document the model must produce the next word.
Hypotheses are token sequences that spell a word; a hypothesis is closed with
the probability that the following token starts a new piece (a blank, a
punctuation mark, a digit or the end of the document).  Probabilities of
different token paths that spell the same word are added.  Reported: the share
of positions where the true word is the first / among the first five.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from common import load_tok, read_docs
from lm import load_model
from yakut_morph_tokenizer import BOUNDARY, SPACE, is_word
from yakut_morph_tokenizer.tokenizer import BOS

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("tok")
ap.add_argument("run")
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--beam", type=int, default=10)
ap.add_argument("--batch", type=int, default=48)
a = ap.parse_args()

tok = load_tok(a.tok)
model, _ = load_model(a.run)
V = len(tok)
is_char = tok.kind == "char"
starts = torch.zeros(V, dtype=torch.bool)       # may open a word
cont = torch.zeros(V, dtype=torch.bool)         # may continue a word
for t, i in tok.vocab.items():
    if i < 3:
        continue
    body = t.replace(SPACE, "").replace(BOUNDARY, "")
    if is_char:
        starts[i] = t == SPACE
        cont[i] = is_word(t)
    elif body and is_word(body):
        if t.startswith(SPACE):
            starts[i] = True
        else:
            cont[i] = True
    elif not body and t and set(t) == {BOUNDARY}:
        cont[i] = True                           # a bare morph-boundary token stays inside the word
ends = ~cont                                     # anything that does not continue the letter run
ends[0] = False
starts, cont, ends = starts.cuda(), cont.cuda(), ends.cuda()
max_steps = 28 if is_char else 14
keep = model.ctx - max_steps - 2

rng = random.Random(0)
docs = read_docs(a.data, "test")
items = []
for di in rng.sample(range(len(docs)), len(docs)):
    ch = [c for c in docs[di].split(" ") if c]
    pos = [i for i in range(10, len(ch)) if is_word(ch[i])]
    for i in rng.sample(pos, min(2, len(pos))):
        items.append((ch[max(0, i - 150):i], ch[i]))
    if len(items) >= a.n:
        break
items = items[:a.n]


@torch.no_grad()
def next_logp(seqs):
    m = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), m, dtype=torch.long)
    for j, s in enumerate(seqs):
        x[j, :len(s)] = torch.tensor(s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x.cuda())
    last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
    return F.log_softmax(logits[torch.arange(len(seqs), device="cuda"), last].float(), -1)


def logaddexp(x, y):
    return max(x, y) + math.log1p(math.exp(-abs(x - y)))


ranks = []
for b0 in range(0, len(items), a.batch):
    part = items[b0:b0 + a.batch]
    ctxs = []
    for left, _ in part:
        ids = [BOS]
        for c in left:
            ids.extend(tok.encode_chunk(c))
        ctxs.append(ids[-keep:])
    beams = [[((), 0.0)] for _ in part]           # (word tokens, log-prob)
    done = [dict() for _ in part]                 # word string -> log-prob
    for step in range(max_steps):
        flat = [(k, h) for k, bs in enumerate(beams) for h in bs]
        if not flat:
            break
        lp = next_logp([ctxs[k] + list(h[0]) for k, h in flat])
        new = [[] for _ in part]
        if step == 0:
            masked = lp.masked_fill(~starts, -1e9)
        else:
            end_lp = torch.logsumexp(lp.masked_fill(~ends, -1e9), -1).tolist()
            masked = lp.masked_fill(~cont, -1e9)
        top = masked.topk(a.beam, -1)
        tv, ti = top.values.tolist(), top.indices.tolist()
        for j, (k, (toks, sc)) in enumerate(flat):
            if step > 0:
                w = tok.decode(toks)
                s = sc + end_lp[j]
                done[k][w] = logaddexp(done[k][w], s) if w in done[k] else s
            for v, i in zip(tv[j], ti[j]):
                if v > -1e8:
                    new[k].append((toks + (i,), sc + v))
        for k in range(len(part)):
            cand = sorted(new[k], key=lambda h: -h[1])[:a.beam]
            # a hypothesis cannot improve: prune those below the 5th finished word
            if len(done[k]) >= 5:
                thr = sorted(done[k].values(), reverse=True)[4]
                cand = [h for h in cand if h[1] > thr]
            beams[k] = cand
    for k, (_, target) in enumerate(part):
        order = sorted(done[k], key=lambda w: -done[k][w])
        ranks.append(order.index(target) + 1 if target in order else 0)
    if (b0 // a.batch) % 20 == 0:
        r = np.asarray(ranks)
        print(len(r), "acc@1", float((r == 1).mean()), flush=True)
r = np.asarray(ranks)
np.save(os.path.join(a.run, "wordpred_ranks.npy"), r)
res = {"n": len(r), "acc1": float((r == 1).mean()), "acc5": float(((r >= 1) & (r <= 5)).mean()),
       "mrr5": float(np.where((r >= 1) & (r <= 5), 1.0 / np.maximum(r, 1), 0.0).mean())}
json.dump(res, open(os.path.join(a.run, "wordpred.json"), "w"), indent=1)
print(json.dumps(res), flush=True)
