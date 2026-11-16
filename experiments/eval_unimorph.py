"""Stem-boundary accuracy against UniMorph (sah).

    python experiments/eval_unimorph.py SEG_DIR DATA_DIR [--part test|dev] [--tok DIR ...]

UniMorph gives (lemma, form, features).  For forms that start with their lemma the
end of the lemma is a gold morph boundary.  Lemmas are split in two halves by a
fixed hash: `dev` was used to choose the hyper-parameters, `test` is reported.
Metrics (forms with a non-empty ending):
    exact   the predicted stem equals the lemma
    recall  the lemma end is one of the predicted boundaries
    over    some predicted boundary lies inside the lemma
and for bare forms (form == lemma): `bare` = share left unsegmented.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import BOUNDARY, SPACE, Segmenter  # noqa: E402
from yakut_morph_tokenizer.tokenizer import SakhaTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seg")
ap.add_argument("data")
ap.add_argument("--part", default="test")
ap.add_argument("--unimorph", default=os.path.join(os.path.dirname(__file__), "..", "data", "external", "unimorph_sah.tsv"))
ap.add_argument("--tok", nargs="*", default=[])
ap.add_argument("--table", nargs="*", default=[], help="name=path of extra word->morphs tables")
ap.add_argument("--per_lemma", type=int, default=30)
ap.add_argument("--out", default="")
a = ap.parse_args()

vocab = {}
for line in open(os.path.join(a.data, "vocab.tsv"), encoding="utf-8"):
    w, c = line.rstrip("\n").split("\t")
    vocab[w] = int(c)

rows = collections.defaultdict(list)
for line in open(a.unimorph, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 3 and " " not in p[1] and p[1].startswith(p[0]):
        part = "dev" if int(hashlib.md5(p[0].encode()).hexdigest()[:8], 16) % 2 == 0 else "test"
        if part == a.part:
            rows[p[0]].append((p[0], p[1], p[2].split(";")[0]))
rng = random.Random(0)
sample = []
for lemma in sorted(rows):
    forms = sorted(set(rows[lemma]))
    attested = [f for f in forms if f[1] in vocab]
    rest = [f for f in forms if f[1] not in vocab]
    rng.shuffle(rest)
    sample += attested + rest[:max(0, a.per_lemma - len(attested))]
print(f"part={a.part} lemmas={len(rows)} forms={len(sample)} attested in corpus={sum(f[1] in vocab for f in sample)}")

systems = {}
seg = Segmenter.load(os.path.join(a.seg, "segmenter.json.gz"))
table = {}
for line in gzip.open(os.path.join(a.seg, "table.tsv.gz"), "rt", encoding="utf-8"):
    w, m = line.rstrip("\n").split("\t")
    table[w] = tuple(m.split(BOUNDARY))


def proposed(w):
    return table[w] if w in table else seg.segment(w)


systems["proposed"] = proposed
for spec in a.table:
    name, path = spec.split("=")
    t = {}
    for line in gzip.open(path, "rt", encoding="utf-8"):
        w, m = line.rstrip("\n").split("\t")
        t[w] = tuple(m.split(BOUNDARY))
    fb = None
    if os.path.exists(path + ".model"):
        import morfessor
        mm = morfessor.MorfessorIO().read_binary_model_file(path + ".model")
        fb = lambda w, mm=mm: tuple(mm.viterbi_segment(w)[0])  # noqa: E731
    systems[name] = (lambda w, t=t, fb=fb: t[w] if w in t else (fb(w) if fb else (w,)))
for d in a.tok:
    tk = SakhaTokenizer.load(d)
    if tk.kind == "char":
        continue

    def bpe(w, tk=tk):
        return tuple(tk.itos[i].replace(SPACE, "").replace(BOUNDARY, "") for i in tk.encode_chunk(w))
    systems[os.path.basename(d.rstrip("/"))] = bpe

res = {}
for name, fn in systems.items():
    for subset in ("all", "attested"):
        st = collections.Counter()
        for lemma, form, pos in sample:
            if subset == "attested" and form not in vocab:
                continue
            m = fn(form)
            bounds, k = set(), 0
            for x in m[:-1]:
                k += len(x)
                bounds.add(k)
            for key in ("ALL", pos):
                if form == lemma:
                    st[key, "n_bare"] += 1
                    st[key, "bare"] += not bounds
                else:
                    st[key, "n"] += 1
                    st[key, "exact"] += bool(bounds) and min(bounds) == len(lemma)
                    st[key, "recall"] += len(lemma) in bounds
                    st[key, "over"] += any(b < len(lemma) for b in bounds)
                    st[key, "nmorph"] += len(m)
        for key in ("ALL", "N", "V", "V.CVB"):
            n = st[key, "n"]
            if not n:
                continue
            r = {"n": n, "exact": st[key, "exact"] / n, "recall": st[key, "recall"] / n,
                 "over": st[key, "over"] / n, "morphs": st[key, "nmorph"] / n,
                 "bare": st[key, "bare"] / max(st[key, "n_bare"], 1), "n_bare": st[key, "n_bare"]}
            res[f"{name}/{subset}/{key}"] = r
            print(f"{name:14s} {subset:9s} {key:6s} n={n:6d} exact={r['exact']:.3f} recall={r['recall']:.3f} "
                  f"over={r['over']:.3f} morphs/form={r['morphs']:.2f} bare_ok={r['bare']:.3f} (n={r['n_bare']})")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
