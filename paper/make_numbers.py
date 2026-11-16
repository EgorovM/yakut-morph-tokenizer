"""Generate paper/numbers.tex (every number and table body of the paper) from results/.

    python paper/make_numbers.py
"""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "..", "results", "summary.json"), encoding="utf-8"))
C = json.load(open(os.path.join(H, "..", "results", "corpus_stats.json"), encoding="utf-8"))
R, T, U = S["runs"], S["tokenizers"], S["unimorph"]


def ru(x, d=2):
    return f"{x:.{d}f}".replace(".", "{,}")


def pct(x, d=1):
    return ru(100 * x, d)


def ms(row, key, scale=1.0, d=2):
    v = row.get(key)
    if not v:
        return "---"
    s = ru(scale * v["mean"], d)
    if v["std"] is not None:
        s += r"\,$\pm$\," + ru(scale * v["std"], d)
    return s


m = {}
m["CorpusWords"] = ru(C["parts"]["train"]["words"] / 1e6, 1)
m["CorpusTypes"] = f"{C['types'] / 1e3:.0f}"
m["SingletonShare"] = f"{100 * C['singletons']:.0f}"
m["RawDocs"] = "120\\,158"
m["DroppedDocs"] = "379"
m["NTemplates"], m["NMorphemes"], m["NSurface"] = str(C["templates"]), str(C["morphemes"]), str(C["surface"])
m["MaxCands"] = str(C["cands_max"])
m["SegTypes"], m["SegTokens"] = pct(C["seg_types"], 0), pct(C["seg_tokens"], 0)
m["MorphsPerToken"] = ru(C["morphs_per_token"], 2)
names = {"train": "Обучающая", "dev": "Настроечная", "test": "Тестовая"}
m["CorpusTable"] = "\n".join(
    f"{names[p]} & {v['docs']:,}".replace(",", "\\,") + f" & {ru(v['words'] / 1e6, 2)} & {ru(v['chars'] / 1e6, 1)} \\\\"
    for p, v in C["parts"].items())

# ---- UniMorph
ua = lambda n, k="ALL": U[f"{n}/attested/{k}"]  # noqa: E731
m["UMrecall"], m["UMrecallN"], m["UMrecallV"] = pct(ua("proposed")["recall"], 0), pct(ua("proposed", "N")["recall"], 0), pct(ua("proposed", "V")["recall"], 0)
m["UMrecallMfs"], m["UMrecallBpe"], m["UMoverBpe"] = pct(ua("morfessor")["recall"], 0), pct(ua("bpe8k")["recall"], 0), pct(ua("bpe8k")["over"], 0)
m["UMforms"] = f"{ua('proposed')['n']:,}".replace(",", "\\,")
m["UMlemmas"] = "2878"
rows = [("proposed", "Предложенный сегментатор"), ("morfessor", "Morfessor Baseline"), ("uni8k", "Unigram, 8 тыс."),
        ("bpe8k", "BPE, 8 тыс."), ("bpe32k", "BPE, 32 тыс.")]
m["UMtable"] = "\n".join(f"{lab} & {pct(ua(n)['recall'])} & {pct(ua(n)['exact'])} & {pct(ua(n)['over'])} & {ru(ua(n)['morphs'])} \\\\"
                         for n, lab in rows)

# ---- tokenizers
order = [("char", "Символы"), ("bpe2k", "BPE"), ("bpe4k", "BPE"), ("bpe8k", "BPE"), ("bpe16k", "BPE"), ("bpe32k", "BPE"),
         ("morph2k", "Морфемный BPE"), ("morph4k", "Морфемный BPE"), ("morph8k", "Морфемный BPE"),
         ("morph16k", "Морфемный BPE"), ("morph32k", "Морфемный BPE"), ("mfs8k", "Morfessor + BPE"),
         ("uni8k", "Unigram"), ("unimorph8k", "Морфемный Unigram")]
lines = []
for n, lab in order:
    if n not in T:
        continue
    t = T[n]
    lines.append(f"{lab} & {t['vocab']:,}".replace(",", "\\,") + f" & {ru(t['train']['tokens'] / 1e6, 1)} & "
                 f"{ru(t['test']['chars_per_token'])} & {pct(t['test_single_token_words'])} \\\\")
    if n in ("char", "bpe32k", "morph32k"):
        lines.append("\\addlinespace")
m["TokTable"] = "\n".join(lines)
m["TokMoreRel"] = f"{100 * (T['morph8k']['train']['tokens'] / T['bpe8k']['train']['tokens'] - 1):.0f}"
m["TokSingleMorph"], m["TokSingleBpe"] = pct(T["morph8k"]["test_single_token_words"], 0), pct(T["bpe8k"]["test_single_token_words"], 0)

# ---- models
body = R["bpe8k"]["params"] - T["bpe8k"]["vocab"] * 512 - 512 * 512
m["ParamsBody"] = ru(body / 1e6, 1)
m["ParamsMin"] = ru(min(r["params"] for r in R.values()) / 1e6, 1)
m["ParamsMax"] = ru(max(r["params"] for r in R.values()) / 1e6, 1)
m["NRuns"] = str(sum(len(r["seeds"]) for r in R.values()))
sub = [r["minutes"] for n, r in R.items() if n != "char"]
m["TrainMinLo"], m["TrainMinHi"] = f"{min(sub):.0f}", f"{max(sub):.0f}"
m["PairsSame"] = str(S.get("pairs_same_stem", 0))
m["BpcBpe"], m["BpcMorph"] = ru(R["bpe8k"]["bpc"]["mean"], 3), ru(R["morph8k"]["bpc"]["mean"], 3)
m["BpcDeltaRel"] = ru(100 * (1 - R["morph8k"]["bpc"]["mean"] / R["bpe8k"]["bpc"]["mean"]), 1)
m["NonwordBpe"], m["NonwordMorph"] = pct(R["bpe8k"]["nonword"]["mean"], 2), pct(R["morph8k"]["nonword"]["mean"], 2)
m["NonwordHuman"] = pct(S["human_generate"]["nonword"], 2)
m["WerFirst"], m["WerOracle"] = pct(R["bpe8k"]["asr_1best"], 2), pct(R["bpe8k"]["asr_oracle"], 2)
main = [("char", "Символы"), ("bpe8k", "BPE"), ("mfs8k", "Morfessor + BPE"), ("morph8k", "Морфемный BPE"),
        ("uni8k", "Unigram"), ("unimorph8k", "Морфемный Unigram")]
m["MainTable"] = "\n".join(
    f"{lab} & {ms(R[n], 'bpc', 1, 3)} & {ms(R[n], 'pairs', 100, 1)} & {ms(R[n], 'acc1', 100, 1)} & "
    f"{ms(R[n], 'acc5', 100, 1)} & {ms(R[n], 'nonword', 100, 2)} & {ms(R[n], 'wer', 100, 2)} \\\\"
    for n, lab in main if n in R)
t = S["tests"].get("morph8k-bpe8k", {})
lab = [("bpc", "Бит на символ", 1, 4), ("pairs", "Минимальные пары, п.\\,п.", 100, 2),
       ("pairs_same_stem", "\\quad в т.\\,ч. с общей основой, п.\\,п.", 100, 2),
       ("acc1", "Слово, top-1, п.\\,п.", 100, 2), ("acc5", "Слово, top-5, п.\\,п.", 100, 2), ("wer", "WER, п.\\,п.", 100, 2)]


def sgn(x, sc, d):
    if round(abs(x) * sc, d) == 0:
        return ru(0, d)
    return ("$+$" if x > 0 else "$-$") + ru(abs(x) * sc, d)


def better(k, v):
    # p_better is P(delta < 0) for bpc/wer and P(a better) for accuracies (sign already flipped)
    return v["p_better"]


m["TestTable"] = "\n".join(
    f"{name} & {sgn(t[k]['delta'], sc, d)} & [{sgn(t[k]['lo'], sc, d)}; {sgn(t[k]['hi'], sc, d)}] & {ru(better(k, t[k]), 3)} \\\\"
    for k, name, sc, d in lab if k in t)

# ---- numbers quoted in the running text
for tag, n in (("Bpe", "bpe8k"), ("Morph", "morph8k"), ("Mfs", "mfs8k"), ("Char", "char"), ("Uni", "uni8k"),
               ("UniMorph", "unimorph8k")):
    r = R[n]
    m["Bpc" + tag] = ru(r["bpc"]["mean"], 3)
    m["Pairs" + tag] = pct(r["pairs"]["mean"], 1)
    m["PairsSame" + tag] = pct(r["pairs_same_stem"], 1)
    m["AccOne" + tag], m["AccFive" + tag] = pct(r["acc1"]["mean"], 1), pct(r["acc5"]["mean"], 1)
    m["Nonword" + tag] = pct(r["nonword"]["mean"], 2)
    m["Novel" + tag] = pct(r["novel"]["mean"], 2)
    m["Derivable" + tag] = pct(r["derivable"]["mean"], 2)
    m["Wer" + tag] = pct(r["wer"]["mean"], 2)
    for b, bt in (("unseen", "Zero"), ("1-9", "Low"), ("10-99", "Mid"), ("100-999", "High"), (">=1000", "Top")):
        m["Band" + bt + tag] = ru(r["bands"][b], 3)
m["BpcStdBpe"], m["BpcStdMorph"] = ru(R["bpe8k"]["bpc"]["std"], 3), ru(R["morph8k"]["bpc"]["std"], 3)
hg = S["human_generate"]
m["NovelHuman"], m["DerivableHuman"] = pct(hg["novel"], 2), pct(hg["derivable"], 2)
m["NonwordSeedsBpe"] = "; ".join(pct(x, 2) for x in R["bpe8k"]["nonword"]["values"])
m["NonwordSeedsMorph"] = "; ".join(pct(x, 2) for x in R["morph8k"]["nonword"]["values"])
m["NonwordGap"] = f"{100 * (R['bpe8k']['nonword']['mean'] - R['morph8k']['nonword']['mean']) / (R['bpe8k']['nonword']['mean'] - hg['nonword']):.0f}"
bw = R["bpe8k"]["band_words"]
m["BandZeroWords"] = f"{bw['unseen']:,}".replace(",", "\\,")
m["BandZeroShare"] = pct(bw["unseen"] / sum(bw.values()), 1)
m["WerMin"], m["WerMax"] = pct(min(r["wer"]["mean"] for r in R.values()), 1), pct(max(r["wer"]["mean"] for r in R.values()), 1)
w1, wo = R["bpe8k"]["asr_1best"], R["bpe8k"]["asr_oracle"]
m["WerRecovered"] = f"{100 * (w1 - R['morph8k']['wer']['mean']) / (w1 - wo):.0f}"
m["StepsBpe"], m["StepsMorph"], m["StepsChar"] = (f"{R[n]['steps']:,}".replace(",", "\\,") for n in ("bpe8k", "morph8k", "char"))
m["StepsCharRel"] = ru(R["char"]["steps"] / R["bpe8k"]["steps"], 1)
m["TokMoreRelMfs"] = f"{100 * (T['mfs8k']['train']['tokens'] / T['bpe8k']['train']['tokens'] - 1):.0f}"
m["TokMoreRelUniMorph"] = f"{100 * (T['unimorph8k']['train']['tokens'] / T['morph8k']['train']['tokens'] - 1):.0f}"
m["TokSingleBpeBig"], m["TokSingleMorphBig"] = pct(T["bpe32k"]["test_single_token_words"], 0), pct(T["morph32k"]["test_single_token_words"], 0)
rel = {v: 100 * (1 - R[f"morph{v}k"]["bpc"]["mean"] / R[f"bpe{v}k"]["bpc"]["mean"]) for v in (2, 4, 8, 16, 32)}
m["BpcRelMin"], m["BpcRelMax"] = ru(min(rel.values()), 1), ru(max(rel.values()), 1)
for v, tag in ((2, "Two"), (4, "Four"), (16, "Sixteen"), (32, "ThirtyTwo")):
    m["BpcBpe" + tag], m["BpcMorph" + tag] = ru(R[f"bpe{v}k"]["bpc"]["mean"], 3), ru(R[f"morph{v}k"]["bpc"]["mean"], 3)


def star(t, sc, d):
    mark = "*" if t["lo"] * t["hi"] > 0 else ""
    return sgn(t["delta"], sc, d) + mark


rows = []
for v in (2, 4, 8, 16, 32):
    t = S["tests"][f"morph{v}k-bpe{v}k"]
    rows.append(f"{v} & {ru(R[f'bpe{v}k']['bpc']['mean'], 3)} & {ru(R[f'morph{v}k']['bpc']['mean'], 3)} & "
                f"{star(t['bpc'], 1, 3)} & {star(t['pairs'], 100, 2)} & {star(t['acc1'], 100, 2)} & "
                f"{star(t['acc5'], 100, 2)} & {star(t['wer'], 100, 2)} \\\\")
m["VocabTable"] = "\n".join(rows)
for key, tag in (("morph8k-mfs8k", "MorphMfs"), ("mfs8k-bpe8k", "MfsBpe"), ("unimorph8k-uni8k", "UniMorphUni"),
                 ("morph8k-bpe8k", "MorphBpe")):
    t = S["tests"][key]
    for k, kt, sc, d in (("bpc", "Bpc", 1, 3), ("pairs", "Pairs", 100, 2), ("pairs_same_stem", "PairsSame", 100, 2),
                         ("acc1", "AccOne", 100, 2), ("acc5", "AccFive", 100, 2), ("wer", "Wer", 100, 2)):
        if k in t:
            m["D" + kt + tag] = sgn(t[k]["delta"], sc, d)
            m["CI" + kt + tag] = f"[{sgn(t[k]['lo'], sc, d)}; {sgn(t[k]['hi'], sc, d)}]"

with open(os.path.join(H, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by make_numbers.py -- do not edit\n")
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("numbers.tex:", len(m), "macros")
