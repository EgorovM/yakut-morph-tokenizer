"""Word-form minimal pairs: share of pairs where the original sentence is more probable.

    python experiments/eval_pairs.py PAIRS.jsonl TOK_DIR RUN_DIR
"""
import json
import os
import sys

import numpy as np

from common import load_tok
from lm import load_model, sequence_logprob
from yakut_morph_tokenizer.tokenizer import BOS


def main(pairs, tok_dir, run):
    tok = load_tok(tok_dir)
    model, _ = load_model(run)
    rows = [json.loads(l) for l in open(pairs, encoding="utf-8")]
    good = sequence_logprob(model, [[BOS] + tok.encode(r["good"], normalized=True) for r in rows])
    bad = sequence_logprob(model, [[BOS] + tok.encode(r["bad"], normalized=True) for r in rows])
    ok = np.asarray([g > b for g, b in zip(good, bad)])
    np.save(os.path.join(run, "pairs_ok.npy"), ok)
    res = {"n": len(rows), "accuracy": float(ok.mean())}
    json.dump(res, open(os.path.join(run, "pairs.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
