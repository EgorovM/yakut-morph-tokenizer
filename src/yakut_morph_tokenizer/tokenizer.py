"""Subword tokenizers of the study behind one interface.

Every tokenizer sees the same normalised text and the same pre-tokenisation
(whitespace chunks cut into letter runs, digit runs and single symbols).  The
first piece of a chunk carries the SPACE marker.  The morphological variants
additionally cut every Sakha word at morph boundaries; pieces that continue a
word carry the BOUNDARY marker, so a suffix never shares a vocabulary entry
with a word-initial string.  A subword model (BPE or Unigram) is then learned
*inside* the pieces: merges never cross a piece boundary.

    kind      pieces of "дьиэлэрбитигэр,"
    bpe       ▁дьиэлэрбитигэр ,
    morph     ▁дьиэ ¦лэр ¦бит ¦игэр ,
    char      every character is a token

Decoding is the inverse for all kinds: concatenate, drop BOUNDARY, turn SPACE
into a blank.
"""
from __future__ import annotations

import json
import os

from .normalize import BOUNDARY, SPACE, is_word, normalize, pieces

SPECIALS = ["<pad>", "<bos>", "<unk>"]
PAD, BOS, UNK = 0, 1, 2


class TableSegmenter:
    """word -> morphs from a fixed table, with a fallback for unseen words."""

    def __init__(self, table, fallback=None):
        self.table, self.fallback = table, fallback

    def segment(self, word):
        hit = self.table.get(word)
        if hit is None:
            hit = self.fallback(word) if self.fallback else (word,)
            self.table[word] = hit
        return hit


def mark_chunk(chunk: str, segmenter=None):
    """Marked pieces of one whitespace-free chunk."""
    out = []
    for i, p in enumerate(pieces(chunk)):
        lead = SPACE if i == 0 else ""
        if segmenter is not None and is_word(p):
            morphs = segmenter.segment(p)
            out.append(lead + morphs[0])
            out.extend(BOUNDARY + m for m in morphs[1:])
        else:
            out.append(lead + p)
    return out


class SakhaTokenizer:
    def __init__(self, kind, vocab, model=None, segmenter=None):
        self.kind, self.vocab, self.model, self.segmenter = kind, vocab, model, segmenter
        self.itos = [None] * len(vocab)
        for t, i in vocab.items():
            self.itos[i] = t
        self._cache = {}

    def __len__(self):
        return len(self.vocab)

    # ---------------------------------------------------------------- encode
    def encode_chunk(self, chunk: str):
        ids = self._cache.get(chunk)
        if ids is None:
            ids = self._cache[chunk] = self._encode_chunk(chunk)
        return ids

    def _encode_chunk(self, chunk):
        if self.kind == "char":
            return tuple(self.vocab.get(c, UNK) for c in SPACE + chunk)
        enc = self.model.encode(mark_chunk(chunk, self.segmenter), is_pretokenized=True,
                                add_special_tokens=False)
        return tuple(enc.ids)

    def warm(self, chunks, batch=20000):
        """Fill the chunk cache for an iterable of distinct chunks (batched, fast)."""
        todo = [c for c in chunks if c not in self._cache]
        if self.kind == "char":
            for c in todo:
                self._cache[c] = self._encode_chunk(c)
            return
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            encs = self.model.encode_batch([mark_chunk(c, self.segmenter) for c in part],
                                           is_pretokenized=True, add_special_tokens=False)
            for c, e in zip(part, encs):
                self._cache[c] = tuple(e.ids)

    def encode(self, text: str, normalized=False):
        if not normalized:
            text = normalize(text)
        out = []
        for chunk in text.split(" "):
            if chunk:
                out.extend(self.encode_chunk(chunk))
        return out

    def tokens(self, text: str, normalized=False):
        return [self.itos[i] for i in self.encode(text, normalized)]

    # ---------------------------------------------------------------- decode
    def decode(self, ids):
        s = "".join(self.itos[i] for i in ids if i >= len(SPECIALS))
        return s.replace(BOUNDARY, "").replace(SPACE, " ").strip()

    # ------------------------------------------------------------------- i/o
    def save(self, d):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"kind": self.kind, "vocab": self.vocab}, f, ensure_ascii=False)
        if self.model is not None:
            self.model.save(os.path.join(d, "tokenizer.json"))

    @classmethod
    def load(cls, d, segmenter=None):
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        model = None
        p = os.path.join(d, "tokenizer.json")
        if os.path.exists(p):
            from tokenizers import Tokenizer
            model = Tokenizer.from_file(p)
        return cls(meta["kind"], meta["vocab"], model, segmenter)


def train_subword(marked_lines, vocab_size, algo="bpe"):
    """Learn a BPE / Unigram model over marked pieces (one line = pieces joined by blanks)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    if algo == "bpe":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False)
    else:
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                          unk_token="<unk>", show_progress=False)
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train_from_iterator(marked_lines, trainer)
    return tok
