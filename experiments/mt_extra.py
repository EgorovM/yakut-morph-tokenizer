"""Add an extra test set to the translation experiment.

    python experiments/mt_extra.py RAW.jsonl PAIRS_DIR NAME

RAW.jsonl has one {"sah": ..., "ru": ...} per line.  Pairs are normalised; duplicates,
one-word sentences and pairs sharing a side with the training part are dropped.
The result is PAIRS_DIR/NAME.jsonl; decode it with  mt.py ... --decode_only --test NAME.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import normalize  # noqa: E402

raw, pairs, name = sys.argv[1:4]
train = [json.loads(l) for l in open(os.path.join(pairs, "train.jsonl"), encoding="utf-8")]
ts, tr = {p["sah"] for p in train}, {p["ru"] for p in train}
seen, out = set(), []
for line in open(raw, encoding="utf-8"):
    p = json.loads(line)
    s, r = normalize(p["sah"]), normalize(p["ru"])
    if len(s.split()) < 2 or len(r.split()) < 2 or s in seen or s in ts or r in tr:
        continue
    seen.add(s)
    out.append({"sah": s, "ru": r})
with open(os.path.join(pairs, name + ".jsonl"), "w", encoding="utf-8") as f:
    for p in out:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(name, len(out), "pairs;", sum(len(p["sah"].split()) for p in out), "Sakha words")
