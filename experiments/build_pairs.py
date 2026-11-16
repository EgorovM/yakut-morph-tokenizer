"""Minimal pairs for the word-form test (tokenizer independent).

    python experiments/build_pairs.py DATA_DIR OUT.jsonl [--n 6000]

A test sentence is copied with one word replaced by a *sibling*: another word
form attested in the training part that shares all but the last few letters and
has the closest frequency.  A language model should give the original sentence
the higher probability.
"""
import argparse
import collections
import json
import math
import random
import re

from common import read_docs, read_vocab
from yakut_morph_tokenizer import is_word

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--min_freq", type=int, default=5)
a = ap.parse_args()
rng = random.Random(0)
vocab = {w: c for w, c in read_vocab(a.data).items() if c >= a.min_freq and is_word(w)}
index = collections.defaultdict(list)
for w in vocab:
    for k in range(3, len(w) + 1):
        index[w[:k]].append(w)


def sibling(w):
    k = max(3, len(w) - 4)
    cands = []
    for s in index.get(w[:k], ()):
        if s == w or abs(len(s) - len(w)) > 4:
            continue
        cp = 0
        while cp < min(len(s), len(w)) and s[cp] == w[cp]:
            cp += 1
        if cp < max(3, len(s) - 4) or cp == len(s) == len(w):
            continue
        cands.append((abs(math.log(vocab[s]) - math.log(vocab[w])), s))
    if not cands:
        return None
    cands.sort()
    return rng.choice(cands[:3])[1]


sents = []
for d in read_docs(a.data, "test"):
    for s in re.split(r"(?<=[.!?…])\s+", d):
        ch = s.split(" ")
        if 6 <= len(ch) <= 40:
            sents.append(ch)
rng.shuffle(sents)
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for ch in sents:
        pos = [i for i, c in enumerate(ch) if i > 0 and len(c) >= 5 and c in vocab]
        rng.shuffle(pos)
        for i in pos:
            s = sibling(ch[i])
            if s:
                f.write(json.dumps({"good": " ".join(ch), "bad": " ".join(ch[:i] + [s] + ch[i + 1:]),
                                    "orig": ch[i], "alt": s, "pos": i,
                                    "f_orig": vocab[ch[i]], "f_alt": vocab[s]}, ensure_ascii=False) + "\n")
                n += 1
                break
        if n >= a.n:
            break
print("pairs", n)
