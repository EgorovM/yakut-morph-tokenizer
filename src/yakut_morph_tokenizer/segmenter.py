"""Rule-constrained, corpus-trained morphological segmenter for Sakha.

Rules propose every suffix chain licensed by vowel harmony, consonant
assimilation and morphotactics; a two-factor model fitted on an unannotated
word list with leave-one-out EM picks one of them:

    P(stem, m_1..m_k) = P_stem(lemma(stem)) * prod_i P_suf(m_i) * P_suf(END)

Suffix probabilities are tied over the allomorphs of a morpheme (the allomorph
is fully determined by the left context).  `P_stem` is a Dirichlet-process-style
estimate over *canonical* stems with a penalty for stems that are not attested
as free words.
"""
from __future__ import annotations

import gzip
import json
import math

from .morphs import MORPHS, NEG_OK, NONFINAL, PREV, build_surface_index
from .phonology import VOWELS, harmony_state, realize_state, V_LOW

MAX_SUFFIXES = 6
MIN_STEM = 2
NAMES = sorted({m.name for m in MORPHS})
NAME_ID = {n: i for i, n in enumerate(NAMES)}
MID2NAME = [NAME_ID[m.name] for m in MORPHS]
END = len(NAMES)
# stem-final alternations: surface consonant before a vowel-initial suffix -> underlying
_VOICED = {"г": "к", "ҕ": "х", "б": "п", "һ": "с"}
# surface consonant produced by assimilation to an identical suffix onset -> underlying
_ASSIM = {"к": "т", "п": "т", "м": "н", "ҥ": "н"}


class Segmenter:
    def __init__(self, stem_counts=None, suf_counts=None, free=None, alpha=1.0,
                 char_p=1 / 36.0, bound_penalty=6.0, depth_bonus=2.0, min_stem=MIN_STEM):
        self.index = build_surface_index()
        self.maxlen = max(len(s) for s in self.index)
        self.stem_counts = stem_counts or {}
        self.suf_counts = suf_counts or [1.0] * (len(NAMES) + 1)
        self.free = free if free is not None else set()
        self.alpha, self.char_p, self.min_stem = alpha, char_p, min_stem
        self.bound_penalty, self.depth_bonus = bound_penalty, depth_bonus
        self._refresh()
        self._cache = {}
        self._split_memo = {}

    # ------------------------------------------------------------- candidates
    def candidates(self, word: str):
        """All licensed analyses: list of (stem_len, ((start, morph_id), ...))."""
        out = [(len(word), ())]
        self._strip(word, len(word), (), None, None, out)
        return out

    def _strip(self, word, end, chain, right_morph, right_first, out):
        if len(chain) >= MAX_SUFFIXES:
            return
        lo = max(self.min_stem, end - self.maxlen)
        for start in range(end - 1, lo - 1, -1):
            hits = self.index.get(word[start:end])
            if not hits:
                continue
            left = word[:start]
            ctx = self._ctx(left)
            if ctx is None:
                continue
            for m, ok in hits:
                if ctx not in ok:
                    continue
                if right_morph is None:
                    if m.need_next is not None or m.cat in NONFINAL:
                        continue
                else:
                    if m.cat not in PREV[right_morph.cat]:
                        continue
                    if m.cat == "NEG" and right_morph.name not in NEG_OK:
                        continue
                    if m.need_next is not None and right_first not in m.need_next:
                        continue
                new = ((start, m.id),) + chain
                if "ROOT" in PREV[m.cat] and self._stem_ok(left):
                    out.append((start, new))
                self._strip(word, start, new, m, word[start], out)

    @staticmethod
    def _ctx(left):
        from .phonology import left_class
        st, cl = harmony_state(left), left_class(left)
        return None if st is None or cl is None else (st, cl)

    def _stem_ok(self, stem):
        return len(stem) >= self.min_stem and any(c in VOWELS for c in stem)

    # ------------------------------------------------------- canonical stems
    def lemma(self, stem: str, nxt: str = "") -> str:
        """Underlying form of a surface stem standing before a suffix starting with `nxt`:
        undoes voicing (балыг-ы ~ балык), assimilation (аппыт ~ ат) and the vowel
        contraction of long-vowel verbs (кэпс-иир ~ кэпсээ) when that form is a free word."""
        if not nxt or stem in self.free:
            return stem
        last = stem[-1]
        if nxt in VOWELS:
            under = _VOICED.get(last)
            if under and stem[:-1] + under in self.free:
                return stem[:-1] + under
            st = harmony_state(stem)
            if st is not None and last not in VOWELS:
                long = stem + realize_state("AA", st, V_LOW)
                if long in self.free:
                    return long
        elif last == nxt:
            under = _ASSIM.get(last)
            if under and stem[:-1] + under in self.free:
                return stem[:-1] + under
        return stem

    def _lemma_of(self, word, cand):
        stem_len, chain = cand
        return self.lemma(word[:stem_len], word[stem_len] if chain else "")

    # ------------------------------------------------------------------ model
    def _refresh(self):
        self._n_stem = sum(self.stem_counts.values())
        tot = sum(self.suf_counts)
        # depth_bonus offsets the cost of one more suffix: without it the likelihood
        # prefers to store every frequent inflected form as a stem of its own
        self._logsuf = [(math.log(c / tot) if c > 0 else -1e9) + (self.depth_bonus if i != END else 0.0)
                        for i, c in enumerate(self.suf_counts)]

    def log_stem(self, lemma, own=0.0, is_whole=False):
        """log P_stem; `own` is the mass the scored word itself put on this stem
        (leave-one-out: a word is not evidence for its own stem)."""
        p0 = (self.char_p ** len(lemma)) * 0.5
        c = max(self.stem_counts.get(lemma, 0.0) - own, 0.0)
        lp = math.log((c + self.alpha * p0) / (self._n_stem + self.alpha))
        if not is_whole and lemma not in self.free:
            lp -= self.bound_penalty
        return lp

    def score(self, word, cand, own=0.0):
        s = self.log_stem(self._lemma_of(word, cand), own, not cand[1]) + self._logsuf[END]
        for _, mid in cand[1]:
            s += self._logsuf[MID2NAME[mid]]
        return s

    def fit(self, vocab, iters=15, min_free=2, verbose=False):
        """Leave-one-out EM over word types.

        `vocab` maps lower-cased word -> corpus frequency.  A word is *free* (may
        stand as a stem without penalty) if its frequency is at least `min_free`.
        Returns the table word -> segmentation for the whole vocabulary."""
        self.free = {w for w, c in vocab.items() if c >= min_free}
        data = [(w, self.candidates(w)) for w in vocab]
        lem = [[self._lemma_of(w, c) for c in cands] for w, cands in data]
        # start from the deepest analyses whose stem is a free word (EM then prunes
        # the splits the data do not support); the uniform start converges to the
        # opposite fixed point, where every inflected stem is its own lexical entry
        post = []
        for (w, cands), ls in zip(data, lem):
            ok = [i for i, l in enumerate(ls) if l in self.free] or [0]
            deep = max(len(cands[i][1]) for i in ok)
            top = [i for i in ok if len(cands[i][1]) == deep]
            post.append([1.0 / len(top) if i in top else 0.0 for i in range(len(cands))])
        split = set()          # words whose current best analysis has suffixes
        best_i = [0] * len(data)
        for it in range(iters):
            stem_counts, suf_counts = {}, [0.01] * (len(NAMES) + 1)
            for (w, cands), ls, ps in zip(data, lem, post):
                for (_, chain), l, p in zip(cands, ls, ps):
                    if p < 1e-6:
                        continue
                    stem_counts[l] = stem_counts.get(l, 0.0) + p
                    suf_counts[END] += p
                    for _, mid in chain:
                        suf_counts[MID2NAME[mid]] += p
            self.stem_counts, self.suf_counts = stem_counts, suf_counts
            self._refresh()
            ll, new_post, new_split = 0.0, [], set()
            for k, ((w, cands), ls, ps) in enumerate(zip(data, lem, post)):
                own = {}
                for l, p in zip(ls, ps):
                    own[l] = own.get(l, 0.0) + p
                sc = []
                for c, l in zip(cands, ls):
                    # a stem must be atomic: a word that is itself analysed cannot be a stem
                    if c[1] and (w[:c[0]] in split or l in split):
                        sc.append(-1e9)
                        continue
                    s = self.log_stem(l, own[l], not c[1]) + self._logsuf[END]
                    for _, mid in c[1]:
                        s += self._logsuf[MID2NAME[mid]]
                    sc.append(s)
                mx = max(sc)
                z = mx + math.log(sum(math.exp(s - mx) for s in sc))
                ll += z
                ps = [math.exp(s - z) for s in sc]
                new_post.append(ps)
                i = max(range(len(ps)), key=ps.__getitem__)
                best_i[k] = i
                if cands[i][1]:
                    new_split.add(w)
            post, split = new_post, new_split
            if verbose:
                print(f"EM iter {it + 1}: loo-loglik/type = {ll / len(data):.4f}, "
                      f"segmented types = {len(split) / len(data):.3f}", flush=True)
        best = {w: cands[i] for (w, cands), i in zip(data, best_i)}
        # final hard counts
        stem_counts, suf_counts = {}, [0.01] * (len(NAMES) + 1)
        for w, c in best.items():
            l = self._lemma_of(w, c)
            stem_counts[l] = stem_counts.get(l, 0.0) + 1.0
            suf_counts[END] += 1
            for _, mid in c[1]:
                suf_counts[MID2NAME[mid]] += 1
        self.stem_counts, self.suf_counts = stem_counts, suf_counts
        self.free = {w for w in self.free if w in stem_counts}
        self._refresh()
        table = {w: self._surface(w, c) for w, c in best.items()}
        self._cache = dict(table)
        self._split_memo = {}
        return table

    @staticmethod
    def _surface(word, cand):
        stem_len, chain = cand
        bounds = [s for s, _ in chain] + [len(word)]
        return (word[:stem_len],) + tuple(word[bounds[i]:bounds[i + 1]] for i in range(len(chain)))

    # ------------------------------------------------------------- inference
    def _best(self, word, depth=0):
        cands = self.candidates(word)
        if depth < 4:
            # atomic-stem constraint, as in training
            ok = [c for c in cands if not c[1] or not self._is_split(word[:c[0]], depth)]
        else:
            ok = cands
        return max(ok, key=lambda c: self.score(word, c))

    def _is_split(self, stem, depth):
        # a pure function of the model (never of the call history), so that the
        # segmentation of a word does not depend on what was segmented before it
        if stem not in self.free:
            return False
        hit = self._split_memo.get(stem)
        if hit is None:
            hit = self._split_memo[stem] = bool(self._best(stem, depth + 1)[1])
        return hit

    def analyse(self, word: str):
        """Best analysis: (stem, [(surface, morpheme name), ...])."""
        stem_len, chain = self._best(word)
        bounds = [s for s, _ in chain] + [len(word)]
        return word[:stem_len], [(word[bounds[i]:bounds[i + 1]], MORPHS[mid].name)
                                 for i, (_, mid) in enumerate(chain)]

    def segment(self, word: str):
        """Surface morphs of a lower-cased word, stem first."""
        hit = self._cache.get(word)
        if hit is None:
            hit = self._cache[word] = self._surface(word, self._best(word))
        return hit

    # ------------------------------------------------------------------- i/o
    def save(self, path):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"alpha": self.alpha, "char_p": self.char_p, "min_stem": self.min_stem,
                       "bound_penalty": self.bound_penalty, "depth_bonus": self.depth_bonus,
                       "suffixes": {n: round(self.suf_counts[i], 3) for n, i in NAME_ID.items()},
                       "end": round(self.suf_counts[END], 3),
                       "stems": {s: [c, int(s in self.free)] for s, c in self.stem_counts.items()}},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        suf = [d["suffixes"][n] for n in NAMES] + [d["end"]]
        return cls({s: v[0] for s, v in d["stems"].items()}, suf,
                   {s for s, v in d["stems"].items() if v[1]},
                   d["alpha"], d["char_p"], d["bound_penalty"], d["depth_bonus"], d["min_stem"])
