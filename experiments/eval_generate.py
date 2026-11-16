"""Decoding task 2: free generation with nucleus sampling.

    python experiments/eval_generate.py DATA_DIR SEG_DIR TOK_DIR RUN_DIR [--n 1000] [--top_p 0.9]

Each model continues the first 30 chunks of a test document by about 300
characters.  The continuation is cut into Sakha words and every word is put
into one of three classes:
    known       attested in the training part
    derivable   not attested, but analysed by the segmenter as a known stem
                (a free word of the training part) plus a licensed suffix chain
    non-word    neither
The same statistics are computed for the human continuation of the same
prompts (pass --human), which gives the level a perfect model should reach.
"""
import argparse
import collections
import gzip
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import load_tok, read_docs, read_vocab
from yakut_morph_tokenizer import BOUNDARY, Segmenter, is_word, pieces
from yakut_morph_tokenizer.tokenizer import BOS

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("seg")
ap.add_argument("tok", nargs="?")
ap.add_argument("run", nargs="?")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--chars", type=int, default=300)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--human", action="store_true")
ap.add_argument("--out", default="")
a = ap.parse_args()

vocab = read_vocab(a.data)
seg = Segmenter.load(os.path.join(a.seg, "segmenter.json.gz"))
docs = [d.split(" ") for d in read_docs(a.data, "test")]
docs = [d for d in docs if len(d) >= 150][:a.n]
prompts = [d[:30] for d in docs]


def classify(texts):
    c = collections.Counter()
    types, bad = set(), collections.Counter()
    for t in texts:
        for ch in t.split(" "):
            for w in pieces(ch):
                if not is_word(w):
                    continue
                c["words"] += 1
                types.add(w)
                if w in vocab:
                    c["known"] += 1
                    continue
                stem_len, chain = seg._best(w)
                lemma = seg.lemma(w[:stem_len], w[stem_len] if chain else "")
                if chain and lemma in seg.free:
                    c["derivable"] += 1
                else:
                    c["nonword"] += 1
                    bad[w] += 1
    n = max(c["words"], 1)
    return {"words": c["words"], "known": c["known"] / n, "derivable": c["derivable"] / n,
            "nonword": c["nonword"] / n, "novel": 1 - c["known"] / n, "type_token": len(types) / n,
            "nonword_examples": [w for w, _ in bad.most_common(40)]}


def cut(text):
    """First a.chars characters, without the (possibly unfinished) last word."""
    text = text[:a.chars]
    return text[:text.rfind(" ")] if " " in text else text


if a.human:
    texts = [cut(" ".join(d[30:])) for d in docs]
    res = classify(texts)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "nonword_examples"}))
    raise SystemExit

from lm import load_model  # noqa: E402

tok = load_tok(a.tok)
model, _ = load_model(a.run)
stats = json.load(open(os.path.join(a.tok, "stats.json")))
steps = int(a.chars / stats["test"]["chars_per_token"] * 1.35) + 8
torch.manual_seed(a.seed)
texts = []
B = 250
for b0 in range(0, len(prompts), B):
    part = prompts[b0:b0 + B]
    seqs = []
    for p in part:
        ids = [BOS]
        for c in p:
            ids.extend(tok.encode_chunk(c))
        seqs.append(ids)
    plen = [len(s) for s in seqs]
    for _ in range(steps):
        m = max(len(s) for s in seqs)
        lo = max(0, m - model.ctx)
        x = torch.zeros(len(seqs), m - lo, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s) - lo] = torch.tensor(s[lo:])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda())
        last = torch.tensor([len(s) - lo - 1 for s in seqs], device="cuda")
        p = F.softmax(logits[torch.arange(len(seqs), device="cuda"), last].float(), -1)
        p[:, :3] = 0                                  # no special tokens inside a continuation
        sp, si = p.sort(-1, descending=True)
        cum = sp.cumsum(-1)
        sp[cum - sp > a.top_p] = 0
        nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(-1).tolist()
        for s, t in zip(seqs, nxt):
            s.append(t)
    for s, n0, pr in zip(seqs, plen, part):
        full = tok.decode(s)
        texts.append(cut(full[len(" ".join(pr)):].lstrip()))
res = classify(texts)
res["samples"] = [{"prompt": " ".join(p), "continuation": t} for p, t in zip(prompts[:20], texts[:20])]
json.dump(res, open(os.path.join(a.run, "generate.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("nonword_examples", "samples")}), flush=True)
