"""Text normalisation and pre-tokenisation shared by every tokenizer in the study."""
from __future__ import annotations

import re
import unicodedata

SPACE = "▁"      # marks "preceded by a space"
BOUNDARY = "¦"   # morph boundary inside a word (never produced by normalisation)

_MAP = str.maketrans({"ғ": "ҕ", "ң": "ҥ", "ɵ": "ө", "ѳ": "ө", "ӊ": "ҥ", "ҳ": "һ",
                      " ": " ", "­": None, "​": None, "﻿": None,
                      SPACE: None, BOUNDARY: None})
_H = re.compile(r"(?<=[а-яёҕҥөһү])h|h(?=[а-яёҕҥөһү])")
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SAKHA = "а-яёҕҥөһү"
# a chunk between spaces is cut into letter runs, digit runs and single other characters
_PIECE = re.compile(rf"[{SAKHA}]+|[a-z]+|[0-9]+|[^\s]")
WORD = re.compile(rf"^[{SAKHA}]+$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(_MAP)
    text = _H.sub("һ", text)
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def pieces(chunk: str):
    """Letter runs / digit runs / single symbols of a whitespace-free chunk."""
    return _PIECE.findall(chunk)


def is_word(piece: str) -> bool:
    return WORD.match(piece) is not None
