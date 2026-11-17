"""Prepare the Sakha-Russian parallel corpus for the translation experiment.

    python experiments/mt_prep.py PARALLEL.parquet OUT_DIR

Input: the corpus of the Olonkho Institute (NEFU), columns `sah` and `ru`
(https://huggingface.co/datasets/averoo/sakha-russian-parallel-corpora-all, CC BY 4.0).
The rows follow the order of the source books, so the split is made by blocks of
500 consecutive rows; dev / test pairs that share a side with a training pair are
dropped.  A Russian BPE tokenizer (8000 units) is trained on the training part.
"""
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import normalize  # noqa: E402
from yakut_morph_tokenizer.tokenizer import SakhaTokenizer, mark_chunk, train_subword  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.read_parquet(src)
parts = {"train": [], "dev": [], "test": []}
seen = set()
for i, (s, r) in enumerate(zip(df["sah"], df["ru"])):
    s, r = normalize(str(s)), normalize(str(r))
    ns, nr = len(s.split()), len(r.split())
    if not (1 <= ns <= 60 and 1 <= nr <= 60) or (s, r) in seen:
        continue
    seen.add((s, r))
    h = int(hashlib.md5(str(i // 500).encode()).hexdigest(), 16) % 100
    parts["test" if h < 7 else "dev" if h < 10 else "train"].append({"sah": s, "ru": r})
ts, tr = {p["sah"] for p in parts["train"]}, {p["ru"] for p in parts["train"]}
for name in ("dev", "test"):
    parts[name] = [p for p in parts[name] if p["sah"] not in ts and p["ru"] not in tr and len(p["sah"].split()) >= 3]
for name, ps in parts.items():
    with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8") as f:
        for p in ps:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(name, len(ps), "pairs;", sum(len(p["sah"].split()) for p in ps), "Sakha words", flush=True)

lines = (" ".join(x for c in p["ru"].split(" ") if c for x in mark_chunk(c)) for p in parts["train"])
model = train_subword(lines, 8000, "bpe")
d = os.path.join(out, "tok_ru")
SakhaTokenizer("bpe", model.get_vocab(), model).save(d)
json.dump({"kind": "bpe", "vocab": 8000}, open(os.path.join(d, "build_args.json"), "w"))
print("Russian tokenizer:", len(model.get_vocab()), flush=True)
