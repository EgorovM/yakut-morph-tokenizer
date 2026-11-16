"""Downstream decoding: rescoring of speech-recognition n-best lists.

    python experiments/eval_asr.py NBEST.json.gz TOK_DIR RUN_DIR

NBEST is a list of {"ref": str, "nbest": [[hypothesis, acoustic_score], ...]}.
A hypothesis is ranked by  acoustic + w * log P_LM + b * (number of words).
(w, b) are chosen on one half of the utterances and applied to the other half
(two-fold cross-validation), so the reported WER is not tuned on the test data.
"""
import gzip
import json
import os
import sys

import numpy as np

from common import load_tok
from lm import load_model, sequence_logprob
from yakut_morph_tokenizer import normalize
from yakut_morph_tokenizer.tokenizer import BOS


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]


def main(nbest, tok_dir, run):
    rows = json.load(gzip.open(nbest, "rt", encoding="utf-8"))
    tok = load_tok(tok_dir)
    model, _ = load_model(run)
    hyps, owner = [], []
    for i, r in enumerate(rows):
        for t, _ in r["nbest"]:
            hyps.append(normalize(t))
            owner.append(i)
    lm = sequence_logprob(model, [[BOS] + tok.encode(h, normalized=True) for h in hyps])
    feats, errs, nref = [], [], []
    k = 0
    for r in rows:
        ref = normalize(r["ref"]).split()
        n = len(r["nbest"])
        hs = hyps[k:k + n]
        feats.append(np.asarray([[s, lm[k + j], len(hs[j].split())] for j, (_, s) in enumerate(r["nbest"])]))
        errs.append(np.asarray([edit(ref, h.split()) for h in hs]))
        nref.append(len(ref))
        k += n
    nref = np.asarray(nref)
    ws = np.linspace(0, 1.5, 31)
    bs = np.linspace(-2, 8, 41)

    def pick(idx, w, b):
        return np.asarray([errs[i][np.argmax(feats[i] @ np.asarray([1.0, w, b]))] for i in idx])

    folds = [np.arange(0, len(rows), 2), np.arange(1, len(rows), 2)]
    chosen = np.zeros(len(rows), dtype=np.int64)
    params = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        best = min(((pick(tr, w, b).sum(), w, b) for w in ws for b in bs))
        params.append([float(best[1]), float(best[2])])
        chosen[te] = pick(te, best[1], best[2])
    first = np.asarray([e[0] for e in errs])
    oracle = np.asarray([e.min() for e in errs])
    res = {"utterances": len(rows), "words": int(nref.sum()), "wer_1best": float(first.sum() / nref.sum()),
           "wer_rescored": float(chosen.sum() / nref.sum()), "wer_oracle": float(oracle.sum() / nref.sum()),
           "params": params}
    np.savez(os.path.join(run, "asr_utt.npz"), errors=chosen, first=first, nref=nref)
    json.dump(res, open(os.path.join(run, "asr.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
