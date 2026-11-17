"""Data-size ablation: keep every N-th training document.

    python experiments/subset_corpus.py DATA_DIR OUT_DIR --every 10

dev.txt and test.txt are copied unchanged; vocab.tsv is recounted on the subset.
"""
import argparse
import collections
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import is_word, pieces  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--every", type=int, required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
vocab, n, words = collections.Counter(), 0, 0
with open(os.path.join(a.out, "train.txt"), "w", encoding="utf-8") as f:
    for i, line in enumerate(open(os.path.join(a.data, "train.txt"), encoding="utf-8")):
        if i % a.every:
            continue
        f.write(line)
        ws = [p for ch in line.rstrip("\n").split(" ") for p in pieces(ch) if is_word(p)]
        vocab.update(ws)
        n += 1
        words += len(ws)
for p in ("dev.txt", "test.txt"):
    shutil.copy(os.path.join(a.data, p), os.path.join(a.out, p))
with open(os.path.join(a.out, "vocab.tsv"), "w", encoding="utf-8") as f:
    for w, c in vocab.most_common():
        f.write(f"{w}\t{c}\n")
print(f"train docs {n} words {words} types {len(vocab)}", flush=True)
