"""Unsupervised baseline segmentation: Morfessor Baseline trained on word types.

    python experiments/train_morfessor.py DATA_DIR OUT_DIR [--min_count 2]

Writes OUT_DIR/table.tsv.gz (word -> morphs) and OUT_DIR/table.tsv.gz.model.
"""
import argparse
import gzip
import os
import sys
import time

import morfessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import BOUNDARY, is_word  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--min_count", type=int, default=2)
a = ap.parse_args()
vocab = {}
for line in open(os.path.join(a.data, "vocab.tsv"), encoding="utf-8"):
    w, c = line.rstrip("\n").split("\t")
    if is_word(w):
        vocab[w] = int(c)
train = [(1, w) for w, c in vocab.items() if c >= a.min_count]      # type-based training
print("types", len(vocab), "training types", len(train), flush=True)
model = morfessor.BaselineModel()
model.load_data(train)
t = time.time()
model.train_batch()
print(f"trained in {time.time() - t:.0f}s", flush=True)
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "table.tsv.gz")
with gzip.open(path, "wt", encoding="utf-8") as f:
    for w in vocab:
        f.write(w + "\t" + BOUNDARY.join(model.viterbi_segment(w)[0]) + "\n")
morfessor.MorfessorIO().write_binary_model_file(path + ".model", model)
print("done", flush=True)
