"""Intrinsic evaluation: bits per character on the test part.

    python experiments/eval_bpc.py DATA_DIR TOK_DIR RUN_DIR

Bits per character do not depend on the tokenization: the total negative
log-likelihood of the test stream is divided by the number of characters
(one extra character per document stands for the document boundary).
Also reported: the cost of Sakha words by their training frequency.
Writes RUN_DIR/bpc.json and RUN_DIR/bpc_docs.npz (per-document sums for the bootstrap).
"""
import json
import math
import os
import sys

import numpy as np

from common import load_tok, read_docs, read_vocab
from lm import load_model, token_nll
from yakut_morph_tokenizer import is_word
from yakut_morph_tokenizer.tokenizer import BOS

BANDS = [(0, 0, "unseen"), (1, 9, "1-9"), (10, 99, "10-99"), (100, 999, "100-999"), (1000, 10 ** 12, ">=1000")]


def main(data, tok_dir, run):
    tok = load_tok(tok_dir)
    vocab = read_vocab(data)
    docs = read_docs(data, "test")
    model, _ = load_model(run)
    ids, doc_of, chunk_spans = [], [], []
    for di, d in enumerate(docs):
        ids.append(BOS)
        doc_of.append(di)             # the <bos> opening document di closes document di-1
        for c in d.split(" "):
            if not c:
                continue
            e = tok.encode_chunk(c)
            chunk_spans.append((len(ids), len(ids) + len(e), c))
            ids.extend(e)
            doc_of.extend([di] * len(e))
    ids.append(BOS)
    doc_of.append(len(docs))
    assert np.array_equal(np.asarray(ids, dtype=np.uint16), np.load(os.path.join(tok_dir, "test.npy")))
    nll = token_nll(model, ids) / math.log(2)          # bits; nll[0] = 0
    doc_of = np.asarray(doc_of)
    # the boundary token after document i is charged to document i
    owner = doc_of.copy()
    bos_pos = np.where(np.asarray(ids) == BOS)[0]
    owner[bos_pos] = doc_of[bos_pos] - 1
    doc_bits = np.bincount(owner[1:], weights=nll[1:], minlength=len(docs))[:len(docs)]
    doc_chars = np.asarray([len(d) + 1 for d in docs], dtype=np.float64)
    doc_tokens = np.bincount(owner[1:], minlength=len(docs))[:len(docs)]
    csum = np.concatenate([[0.0], np.cumsum(nll, dtype=np.float64)])
    band = {name: [0.0, 0, 0, 0] for _, _, name in BANDS}     # bits, chars, words, tokens
    for s, e, c in chunk_spans:
        if not is_word(c):
            continue
        f = vocab.get(c, 0)
        name = next(n for lo, hi, n in BANDS if lo <= f <= hi)
        b = band[name]
        b[0] += csum[e] - csum[s]
        b[1] += len(c) + 1
        b[2] += 1
        b[3] += e - s
    res = {"bpc": float(doc_bits.sum() / doc_chars.sum()),
           "bits_per_token": float(doc_bits.sum() / doc_tokens.sum()),
           "ppl_token": float(2 ** (doc_bits.sum() / doc_tokens.sum())),
           "tokens": int(doc_tokens.sum()), "chars": int(doc_chars.sum()),
           "bands": {n: {"bits_per_char": b[0] / max(b[1], 1), "bits_per_word": b[0] / max(b[2], 1),
                         "tokens_per_word": b[3] / max(b[2], 1), "words": b[2]} for n, b in band.items()}}
    json.dump(res, open(os.path.join(run, "bpc.json"), "w"), indent=1)
    np.savez(os.path.join(run, "bpc_docs.npz"), bits=doc_bits, chars=doc_chars)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
