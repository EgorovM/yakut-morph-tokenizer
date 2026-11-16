"""Fit the morphological segmenter on the training vocabulary.

    python experiments/train_segmenter.py DATA_DIR OUT_DIR [--char_p 0.0278] [--bound_penalty 4]

Reads DATA_DIR/vocab.tsv (word <tab> frequency, training part only) and writes
OUT_DIR/segmenter.json.gz (model) and OUT_DIR/table.tsv.gz (word -> morphs).
"""
import argparse
import gzip
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import BOUNDARY, Segmenter, is_word  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--char_p", type=float, default=1 / 36)
ap.add_argument("--bound_penalty", type=float, default=6.0)
ap.add_argument("--depth_bonus", type=float, default=2.0)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--min_free", type=int, default=2)
a = ap.parse_args()

vocab = {}
for line in open(os.path.join(a.data, "vocab.tsv"), encoding="utf-8"):
    w, c = line.rstrip("\n").split("\t")
    if is_word(w):
        vocab[w] = int(c)
print("word types", len(vocab), flush=True)
seg = Segmenter(alpha=a.alpha, char_p=a.char_p, bound_penalty=a.bound_penalty, depth_bonus=a.depth_bonus)
t = time.time()
table = seg.fit(vocab, iters=a.iters, min_free=a.min_free, verbose=True)
print(f"fit in {time.time() - t:.0f}s", flush=True)
os.makedirs(a.out, exist_ok=True)
seg.save(os.path.join(a.out, "segmenter.json.gz"))
with gzip.open(os.path.join(a.out, "table.tsv.gz"), "wt", encoding="utf-8") as f:
    for w, m in table.items():
        f.write(w + "\t" + BOUNDARY.join(m) + "\n")
tok = sum(vocab.values())
seg_types = sum(len(m) > 1 for m in table.values())
seg_tok = sum(vocab[w] for w, m in table.items() if len(m) > 1)
morphs_tok = sum(vocab[w] * len(m) for w, m in table.items())
print(f"segmented: {seg_types / len(table):.3f} of types, {seg_tok / tok:.3f} of tokens; "
      f"morphs per token {morphs_tok / tok:.3f}; stems {len(seg.stem_counts)}", flush=True)
