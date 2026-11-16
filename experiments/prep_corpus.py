"""Build the train/dev/test split of the Sakha news corpus.

    python experiments/prep_corpus.py RAW_DIR OUT_DIR

RAW_DIR holds one sub-directory per source with one plain-text document per file.
Every document is normalised to a single line.  Exact and near duplicates are
removed, documents that are mostly Russian are dropped, the split is a
deterministic function of the document text, and dev/test documents that share
long sentences with the training part are discarded.
"""
import collections
import glob
import hashlib
import multiprocessing as mp
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import is_word, normalize, pieces  # noqa: E402

SOURCES = ["kyym", "eder_saas", "sakha_sire", "uluus_media"]
MIN_CHARS = 300
MIN_SAKHA = 0.02          # share of the letters ҕ ҥ ө һ ү among Cyrillic letters
SENT = re.compile(r"[^.!?…]{60,}[.!?…]")
_LET = re.compile(r"[а-яёҕҥөһү]")
_SAH = re.compile(r"[ҕҥөһү]")


def load(path):
    try:
        t = normalize(open(path, encoding="utf-8", errors="replace").read())
    except OSError:
        return None
    if len(t) < MIN_CHARS:
        return None
    n = len(_LET.findall(t))
    if n == 0 or len(_SAH.findall(t)) / n < MIN_SAKHA:
        return None
    return os.path.basename(os.path.dirname(path)), t


def h(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main(raw, out):
    os.makedirs(out, exist_ok=True)
    files = []
    for s in SOURCES:
        files += sorted(glob.glob(os.path.join(raw, s, "*.txt")))
    print("files", len(files), flush=True)
    with mp.Pool(64) as pool:
        docs = [d for d in pool.imap(load, files, chunksize=500) if d]
    print("kept after length/language filter", len(docs), flush=True)
    seen, uniq = set(), []
    for src, t in docs:
        k1, k2 = h(t), h(t[:200])
        if k1 in seen or k2 in seen:
            continue
        seen.update((k1, k2))
        uniq.append((src, t))
    print("after deduplication", len(uniq), flush=True)
    parts = {"train": [], "dev": [], "test": []}
    for src, t in uniq:
        b = int(h(t)[:8], 16) % 100
        parts["test" if b < 2 else "dev" if b < 4 else "train"].append((src, t))
    train_sents = set()
    for _, t in parts["train"]:
        for s in SENT.findall(t):
            train_sents.add(h(s.strip())[:16])
    for name in ("dev", "test"):
        keep = []
        for src, t in parts[name]:
            ss = [h(s.strip())[:16] for s in SENT.findall(t)]
            if ss and sum(x in train_sents for x in ss) / len(ss) > 0.1:
                continue
            keep.append((src, t))
        print(name, "dropped for overlap with train:", len(parts[name]) - len(keep), flush=True)
        parts[name] = keep
    vocab = collections.Counter()
    for name, ds in parts.items():
        by = collections.Counter(s for s, _ in ds)
        chars = sum(len(t) for _, t in ds)
        words = 0
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as f:
            for _, t in ds:
                f.write(t + "\n")
                ws = [p for ch in t.split(" ") for p in pieces(ch) if is_word(p)]
                words += len(ws)
                if name == "train":
                    vocab.update(ws)
        print(f"{name}: docs {len(ds)} chars {chars} words {words} by source {dict(by)}", flush=True)
    with open(os.path.join(out, "vocab.tsv"), "w", encoding="utf-8") as f:
        for w, c in vocab.most_common():
            f.write(f"{w}\t{c}\n")
    print("train word types", len(vocab), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
