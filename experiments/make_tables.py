"""Collect every result into results/summary.json and draw the figures.

    python experiments/make_tables.py ROOT OUT_DIR

ROOT has the layout of run_all.sh (data/, seg/, tok/, runs/, eval/).
Paired bootstrap (10 000 resamples) is used for all differences between tokenizers.
"""
import collections
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from yakut_morph_tokenizer import BOUNDARY, Segmenter  # noqa: E402

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
NB = 10000


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


runs = collections.defaultdict(dict)
for d in sorted(glob.glob(os.path.join(root, "runs", "*_s*"))):
    name, seed = os.path.basename(d).rsplit("_s", 1)
    if not os.path.exists(os.path.join(d, "bpc.json")):
        continue
    r = {"dir": d, "config": jload(os.path.join(d, "config.json"))}
    for k in ("bpc", "pairs", "wordpred", "generate", "asr"):
        r[k] = jload(os.path.join(d, k + ".json"))
    log = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    r["log"] = [(x["epoch"], x["dev_bpc"]) for x in log]
    r["best_dev_bpc"] = min(x["dev_bpc"] for x in log)
    r["minutes"] = log[-1]["min"]
    runs[name][int(seed)] = r

summary = {"tokenizers": {}, "runs": {}, "tests": {}}
for name in sorted(glob.glob(os.path.join(root, "tok", "*"))):
    s = jload(os.path.join(name, "stats.json"))
    if s:
        summary["tokenizers"][os.path.basename(name)] = s


def metric(r, key):
    try:
        return {"bpc": r["bpc"]["bpc"], "pairs": r["pairs"]["accuracy"], "acc1": r["wordpred"]["acc1"],
                "acc5": r["wordpred"]["acc5"], "nonword": r["generate"]["nonword"],
                "novel": r["generate"]["novel"], "derivable": r["generate"]["derivable"],
                "wer": r["asr"]["wer_rescored"] if r["asr"] else None, "dev_bpc": r["best_dev_bpc"],
                "rare_bpc": r["bpc"]["bands"]["unseen"]["bits_per_char"]}[key]
    except (TypeError, KeyError):
        return None


KEYS = ["bpc", "dev_bpc", "pairs", "acc1", "acc5", "nonword", "novel", "derivable", "wer", "rare_bpc"]
for name, seeds in runs.items():
    row = {"seeds": sorted(seeds), "params": seeds[min(seeds)]["config"]["params"],
           "steps": seeds[min(seeds)]["config"]["steps"], "minutes": float(np.mean([r["minutes"] for r in seeds.values()]))}
    for k in KEYS:
        v = [metric(r, k) for r in seeds.values()]
        v = [x for x in v if x is not None]
        if v:
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                      "values": [float(x) for x in v]}
    row["bands"] = {b: float(np.mean([r["bpc"]["bands"][b]["bits_per_char"] for r in seeds.values()]))
                    for b in seeds[min(seeds)]["bpc"]["bands"]}
    row["band_words"] = {b: seeds[min(seeds)]["bpc"]["bands"][b]["words"] for b in row["bands"]}
    row["band_tokens"] = {b: seeds[min(seeds)]["bpc"]["bands"][b]["tokens_per_word"] for b in row["bands"]}
    row["curve"] = seeds[min(seeds)]["log"]
    row["asr_1best"] = seeds[min(seeds)]["asr"]["wer_1best"] if seeds[min(seeds)]["asr"] else None
    row["asr_oracle"] = seeds[min(seeds)]["asr"]["wer_oracle"] if seeds[min(seeds)]["asr"] else None
    summary["runs"][name] = row


# ------------------------------------------------------------ paired bootstrap
def seed_mean(name, fn):
    return np.mean([fn(r["dir"]) for r in runs[name].values()], axis=0)


def boot(stat):
    obs = stat(None)
    n = stat("n")
    vals = np.asarray([stat(rng.integers(0, n, n)) for _ in range(NB)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta": float(obs), "lo": float(lo), "hi": float(hi), "p_better": float((vals < 0).mean())}


def compare(a, b):
    """Differences a - b (negative BPC / WER and positive accuracy favour a)."""
    res = {}
    ba, bb = (seed_mean(x, lambda d: np.load(os.path.join(d, "bpc_docs.npz"))["bits"]) for x in (a, b))
    ch = np.load(os.path.join(runs[a][min(runs[a])]["dir"], "bpc_docs.npz"))["chars"]

    def s_bpc(i):
        if isinstance(i, str):
            return len(ch)
        i = slice(None) if i is None else i
        return ba[i].sum() / ch[i].sum() - bb[i].sum() / ch[i].sum()
    res["bpc"] = boot(s_bpc)
    for key, fname, conv in (("pairs", "pairs_ok.npy", lambda x: x.astype(float)),
                             ("acc1", "wordpred_ranks.npy", lambda x: (x == 1).astype(float)),
                             ("acc5", "wordpred_ranks.npy", lambda x: ((x >= 1) & (x <= 5)).astype(float))):
        xa, xb = (seed_mean(x, lambda d: conv(np.load(os.path.join(d, fname)))) for x in (a, b))

        def s(i, xa=xa, xb=xb):
            if isinstance(i, str):
                return len(xa)
            i = slice(None) if i is None else i
            return -(xa[i].mean() - xb[i].mean())       # sign flipped so that "< 0" means a is better
        r = boot(s)
        res[key] = {"delta": -r["delta"], "lo": -r["hi"], "hi": -r["lo"], "p_better": r["p_better"]}
    if all(os.path.exists(os.path.join(r["dir"], "asr_utt.npz")) for x in (a, b) for r in runs[x].values()):
        ea, eb = (seed_mean(x, lambda d: np.load(os.path.join(d, "asr_utt.npz"))["errors"].astype(float)) for x in (a, b))
        nref = np.load(os.path.join(runs[a][min(runs[a])]["dir"], "asr_utt.npz"))["nref"]

        def s_wer(i):
            if isinstance(i, str):
                return len(nref)
            i = slice(None) if i is None else i
            return (ea[i].sum() - eb[i].sum()) / nref[i].sum()
        res["wer"] = boot(s_wer)
    return res


for a, b in (("morph8k", "bpe8k"), ("mfs8k", "bpe8k"), ("morph8k", "mfs8k"), ("morph8k", "char"),
             ("bpe8k", "char"), ("unimorph8k", "uni8k"), ("morph2k", "bpe2k"), ("morph4k", "bpe4k"),
             ("morph16k", "bpe16k"), ("morph32k", "bpe32k")):
    if a in runs and b in runs:
        summary["tests"][f"{a}-{b}"] = compare(a, b)

# ---------------------------------------------- minimal pairs by type of contrast
pairs_path = os.path.join(root, "eval", "pairs.jsonl")
if os.path.exists(pairs_path):
    seg = Segmenter.load(os.path.join(root, "seg", "segmenter.json.gz"))
    table = {}
    for line in gzip.open(os.path.join(root, "seg", "table.tsv.gz"), "rt", encoding="utf-8"):
        w, m = line.rstrip("\n").split("\t")
        table[w] = m.split(BOUNDARY)[0]
    rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    same = np.asarray([table.get(r["orig"]) == table.get(r["alt"]) for r in rows])
    summary["pairs_same_stem"] = int(same.sum())
    for name, seeds in runs.items():
        oks = [np.load(os.path.join(r["dir"], "pairs_ok.npy")) for r in seeds.values()
               if os.path.exists(os.path.join(r["dir"], "pairs_ok.npy"))]
        if oks:
            ok = np.mean(oks, axis=0)
            summary["runs"][name]["pairs_same_stem"] = float(ok[same].mean())
            summary["runs"][name]["pairs_other"] = float(ok[~same].mean())
    for a, b in (("morph8k", "bpe8k"), ("morph8k", "mfs8k"), ("mfs8k", "bpe8k")):
        if a in runs and b in runs:
            xa, xb = (seed_mean(x, lambda d: np.load(os.path.join(d, "pairs_ok.npy")).astype(float))[same] for x in (a, b))
            r = boot(lambda i: len(xa) if isinstance(i, str) else -(xa[slice(None) if i is None else i].mean()
                                                                 - xb[slice(None) if i is None else i].mean()))
            summary["tests"][f"{a}-{b}"]["pairs_same_stem"] = {"delta": -r["delta"], "lo": -r["hi"], "hi": -r["lo"],
                                                              "p_better": r["p_better"]}

summary["human_generate"] = jload(os.path.join(root, "eval", "human_generate.json"))
summary["unimorph"] = jload(os.path.join(root, "eval", "unimorph_test.json"))
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)

# --------------------------------------------------------------------- figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#52514e", "pdf.fonttype": 42})
C = {"bpe": "#2a78d6", "morph": "#d9622b", "char": "#2a9d6a", "mfs": "#8a5cc7"}
R = summary["runs"]
sizes = [2, 4, 8, 16, 32]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, key, lab in ((axes[0], "bpc", "бит на символ (тест)"), (axes[1], "acc1", "точность предсказания слова, top-1")):
    for fam, name, mk, ls in (("bpe", "BPE", "o", "-"), ("morph", "морфемный BPE", "s", "--")):
        xs = [v for v in sizes if f"{fam}{v}k" in R and key in R[f"{fam}{v}k"]]
        ys = [R[f"{fam}{v}k"][key]["mean"] for v in xs]
        ax.plot(xs, ys, ls, marker=mk, color=C[fam], lw=2, ms=6, label=name)
    if "char" in R and key in R["char"]:
        ax.axhline(R["char"][key]["mean"], color=C["char"], lw=1.5, ls=":", label="символы")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{v}k" for v in sizes])
    ax.set_xlabel("размер словаря")
    ax.set_ylabel(lab)
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_vocab.pdf"))

bands = ["unseen", "1-9", "10-99", "100-999", ">=1000"]
labels = ["0", "1–9", "10–99", "100–999", "≥1000"]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
shown = [(n, l, c, h) for n, l, c, h in (("char", "символы", C["char"], ""), ("bpe8k", "BPE", C["bpe"], ""),
                                         ("mfs8k", "Morfessor + BPE", C["mfs"], "//"),
                                         ("morph8k", "морфемный BPE", C["morph"], "xx")) if n in R]
w = 0.8 / max(len(shown), 1)
for k, (n, l, c, h) in enumerate(shown):
    ax.bar(np.arange(len(bands)) + (k - (len(shown) - 1) / 2) * w, [R[n]["bands"][b] for b in bands], w * 0.9,
           color=c, hatch=h, edgecolor="#fcfcfb", linewidth=0.8, label=l)
ax.set_xticks(np.arange(len(bands)))
ax.set_xticklabels(labels)
ax.set_xlabel("частота словоформы в обучающей части")
ax.set_ylabel("бит на символ")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_bands.pdf"))
print("summary and figures written to", out)
