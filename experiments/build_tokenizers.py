"""Train one tokenizer and encode the corpus with it.

    python experiments/build_tokenizers.py DATA_DIR OUT_DIR --kind bpe --vocab 8000
    python experiments/build_tokenizers.py DATA_DIR OUT_DIR --kind bpe --vocab 8000 \
        --seg_table SEG/table.tsv.gz --seg_model SEG/segmenter.json.gz      # morphological BPE

kind: char | bpe | unigram.  With --seg_table the words are cut at morph
boundaries before the subword model is trained (merges cannot cross them).
OUT_DIR receives the tokenizer, {train,dev,test}.npy (uint16 ids, every document
is preceded by <bos>) and stats.json.
"""
import argparse
import collections
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import BOUNDARY, SPACE, Segmenter, is_word  # noqa: E402
from yakut_morph_tokenizer.tokenizer import (BOS, SPECIALS, UNK, SakhaTokenizer, TableSegmenter,  # noqa: E402
                                             mark_chunk, train_subword)


def load_segmenter(table_path, model_path=None):
    table = {}
    for line in gzip.open(table_path, "rt", encoding="utf-8"):
        w, m = line.rstrip("\n").split("\t")
        table[w] = tuple(m.split(BOUNDARY))
    fallback = None
    if model_path:
        fallback = Segmenter.load(model_path).segment
    elif os.path.exists(table_path + ".model"):
        import morfessor
        mm = morfessor.MorfessorIO().read_binary_model_file(table_path + ".model")
        fallback = lambda w: tuple(mm.viterbi_segment(w)[0])  # noqa: E731
    return TableSegmenter(table, fallback)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--kind", required=True, choices=["char", "bpe", "unigram"])
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--seg_table")
    ap.add_argument("--seg_model")
    ap.add_argument("--encode_only", action="store_true", help="reuse the tokenizer in OUT_DIR")
    ap.add_argument("--force", action="store_true", help="with --encode_only: rewrite train.npy too")
    a = ap.parse_args()
    t0 = time.time()
    docs = {p: [l.rstrip("\n") for l in open(os.path.join(a.data, p + ".txt"), encoding="utf-8")]
            for p in ("train", "dev", "test")}
    seg = load_segmenter(a.seg_table, a.seg_model) if a.seg_table else None
    counts = collections.Counter()
    for d in docs["train"]:
        counts.update(d.split(" "))
    counts.pop("", None)
    print(f"train chunks: {sum(counts.values())} tokens, {len(counts)} types ({time.time() - t0:.0f}s)", flush=True)

    if a.encode_only:
        tok = SakhaTokenizer.load(a.out, seg)
    elif a.kind == "char":
        ch = collections.Counter()
        for c, n in counts.items():
            for x in c:
                ch[x] += n
        vocab = {t: i for i, t in enumerate(SPECIALS)}
        vocab[SPACE] = len(vocab)
        for x, n in sorted(ch.items()):
            if n >= 5:
                vocab[x] = len(vocab)
        tok = SakhaTokenizer("char", vocab)
    else:
        marked = {c: " ".join(mark_chunk(c, seg)) for c in counts}

        def lines():
            for d in docs["train"]:
                yield " ".join(marked[c] for c in d.split(" ") if c)
        model = train_subword(lines(), a.vocab, a.kind)
        tok = SakhaTokenizer(a.kind + ("+morph" if seg else ""), model.get_vocab(), model, seg)
    print(f"tokenizer ready: |V| = {len(tok)} ({time.time() - t0:.0f}s)", flush=True)
    if not a.encode_only:
        tok.save(a.out)
        if tok.kind != "char":
            tok = SakhaTokenizer.load(a.out, seg)
    json.dump({k: (os.path.abspath(v) if k in ('seg_table', 'seg_model') and v else v)
               for k, v in vars(a).items() if k not in ("encode_only", "force")},
              open(os.path.join(a.out, 'build_args.json'), 'w'), indent=1)

    stats = {"kind": tok.kind, "vocab": len(tok)}
    for part, ds in docs.items():
        chunks = set()
        for d in ds:
            chunks.update(d.split(" "))
        chunks.discard("")
        tok.warm(sorted(chunks))
        ids = []
        for d in ds:
            ids.append(BOS)
            ids.extend(tok.encode(d, normalized=True))
        ids.append(BOS)
        arr = np.asarray(ids, dtype=np.uint16)
        path = os.path.join(a.out, part + ".npy")
        if a.encode_only and part == "train" and not a.force:
            assert np.array_equal(arr, np.load(path)), "training stream changed"
        else:
            np.save(path, arr)
        chars = sum(len(d) for d in ds)
        words = sum(len(d.split(" ")) for d in ds)
        stats[part] = {"docs": len(ds), "tokens": int(len(arr)), "chars": chars, "chunks": words,
                       "tokens_per_chunk": (len(arr) - len(ds) - 1) / words,
                       "chars_per_token": chars / (len(arr) - len(ds) - 1),
                       "unk_rate": float((arr == UNK).mean())}
        print(part, stats[part], f"({time.time() - t0:.0f}s)", flush=True)
    # share of Sakha word tokens (test) that are a single vocabulary item
    cnt = collections.Counter()
    for d in docs["test"]:
        cnt.update(c for c in d.split(" ") if is_word(c))
    one = sum(n for c, n in cnt.items() if len(tok.encode_chunk(c)) == (2 if tok.kind == "char" else 1))
    stats["test_single_token_words"] = one / max(sum(cnt.values()), 1)
    # round trip
    sample = docs["test"][:200]
    stats["roundtrip_ok"] = sum(tok.decode(tok.encode(d, normalized=True)) == d for d in sample) / len(sample)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print("round trip", stats["roundtrip_ok"], "done", flush=True)


if __name__ == "__main__":
    main()
