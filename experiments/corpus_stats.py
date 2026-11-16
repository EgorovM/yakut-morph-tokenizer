"""Corpus and segmenter statistics quoted in the paper.

    python experiments/corpus_stats.py ROOT OUT.json
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from yakut_morph_tokenizer import BOUNDARY, Segmenter, is_word, pieces  # noqa: E402
from yakut_morph_tokenizer.morphs import MORPHS, build_surface_index  # noqa: E402
from yakut_morph_tokenizer.segmenter import NAMES  # noqa: E402

root, out = sys.argv[1], sys.argv[2]
res = {"parts": {}}
for part in ("train", "dev", "test"):
    docs = words = chars = 0
    for line in open(os.path.join(root, "data", part + ".txt"), encoding="utf-8"):
        line = line.rstrip("\n")
        docs += 1
        chars += len(line)
        words += sum(1 for c in line.split(" ") for p in pieces(c) if is_word(p))
    res["parts"][part] = {"docs": docs, "words": words, "chars": chars}
vocab = {}
for line in open(os.path.join(root, "data", "vocab.tsv"), encoding="utf-8"):
    w, c = line.rstrip("\n").split("\t")
    if is_word(w):
        vocab[w] = int(c)
res["types"] = len(vocab)
res["singletons"] = sum(c == 1 for c in vocab.values()) / len(vocab)
seg = Segmenter()
n = [len(seg.candidates(w)) for w in vocab]
res["cands_mean"], res["cands_max"] = sum(n) / len(n), max(n)
res["templates"], res["morphemes"], res["surface"] = len(MORPHS), len(NAMES), len(build_surface_index())
tok = sum(vocab.values())
seg_types = seg_tok = morphs = 0
for line in gzip.open(os.path.join(root, "seg", "table.tsv.gz"), "rt", encoding="utf-8"):
    w, m = line.rstrip("\n").split("\t")
    k = len(m.split(BOUNDARY))
    seg_types += k > 1
    seg_tok += vocab[w] * (k > 1)
    morphs += vocab[w] * k
res["seg_types"], res["seg_tokens"], res["morphs_per_token"] = seg_types / len(vocab), seg_tok / tok, morphs / tok
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
