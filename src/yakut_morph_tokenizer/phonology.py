"""Sakha (Yakut) phonology needed for suffix allomorphy: vowel harmony and
suffix-onset consonant assimilation.

A suffix is written as a *template* over Cyrillic literals and archiphonemes:

    A  low harmonic vowel       а / э / о / ө
    I  high harmonic vowel      ы / и / у / ү
    D  harmonic diphthong       ыа / иэ / уо / үө
    L  onset  л / т / д / н     (plural -LAр, proprietive -LAAх ...)
    T  onset  т / л / д / н     (partitive -TA, past -TI ...)
    G  onset  ҕ / г / к / х / ҥ (dative -GA, 2nd person -GIн ...)
    B  onset  б / п / м         (1st person -BIт, negation -BA ...)

`realize(template, left)` returns the only surface form the template can take
after the string `left`.
"""
from __future__ import annotations

LOW = "аэоө"
HIGH = "ыиуү"
RU_VOWELS = {"я": "а", "е": "э", "ё": "о", "ю": "у"}
VOWELS = set(LOW + HIGH) | set(RU_VOWELS)

# harmony states
BACK, FRONT, O, OE, U, UE = range(6)
STATES = (BACK, FRONT, O, OE, U, UE)
_A = {BACK: "а", FRONT: "э", O: "о", OE: "ө", U: "а", UE: "э"}
_I = {BACK: "ы", FRONT: "и", O: "у", OE: "ү", U: "у", UE: "ү"}
_D = {BACK: "ыа", FRONT: "иэ", O: "уо", OE: "үө", U: "уо", UE: "үө"}

# left-context classes for consonant assimilation
V_LOW, V_HIGH, VL, KH, LAT, RY, NAS, VD = range(8)
CLASSES = (V_LOW, V_HIGH, VL, KH, LAT, RY, NAS, VD)
_CONS_CLASS = {}
for _c in "кпстчшщцф":
    _CONS_CLASS[_c] = VL
_CONS_CLASS["х"] = KH
_CONS_CLASS["л"] = LAT
for _c in "рй":
    _CONS_CLASS[_c] = RY
for _c in "мнҥ":
    _CONS_CLASS[_c] = NAS
for _c in "бвгджзҕһ":
    _CONS_CLASS[_c] = VD

_ONSET = {
    "L": {V_LOW: "л", V_HIGH: "л", LAT: "л", VL: "т", KH: "т", RY: "д", VD: "д", NAS: "н"},
    "T": {V_LOW: "т", V_HIGH: "т", LAT: "л", VL: "т", KH: "т", RY: "д", VD: "д", NAS: "н"},
    "G": {V_LOW: "ҕ", V_HIGH: "г", LAT: "г", VL: "к", KH: "х", RY: "г", VD: "г", NAS: "ҥ"},
    "B": {V_LOW: "б", V_HIGH: "б", LAT: "б", VL: "п", KH: "п", RY: "б", VD: "б", NAS: "м"},
}


def is_vowel(ch: str) -> bool:
    return ch in VOWELS


def harmony_state(s: str):
    """Harmony state set by the last vowel of `s` (None if there is no vowel)."""
    for i in range(len(s) - 1, -1, -1):
        ch = RU_VOWELS.get(s[i], s[i])
        if ch in "аы":
            return BACK
        if ch in "эи":
            return FRONT
        if ch == "у":
            return U
        if ch == "ү":
            return UE
        if ch == "о":
            return U if i > 0 and s[i - 1] == "у" else O
        if ch == "ө":
            return UE if i > 0 and s[i - 1] == "ү" else OE
    return None


def left_class(s: str):
    """Assimilation class of the final segment of `s` (None if undefined)."""
    for i in range(len(s) - 1, -1, -1):
        ch = s[i]
        if ch in "ьъ":
            continue
        if ch in VOWELS:
            return V_LOW if RU_VOWELS.get(ch, ch) in LOW else V_HIGH
        return _CONS_CLASS.get(ch)
    return None


def _after(ch: str, prev: str):
    if ch in "аы":
        return BACK
    if ch in "эи":
        return FRONT
    if ch == "у":
        return U
    if ch == "ү":
        return UE
    if ch == "о":
        return U if prev == "у" else O
    return UE if prev == "ү" else OE


def realize_state(template: str, state: int, cls: int) -> str:
    """Surface form of `template` after a context with harmony `state` and class `cls`."""
    out = []
    for k, sym in enumerate(template):
        if sym == "A":
            v = _A[state]
        elif sym == "I":
            v = _I[state]
        elif sym == "D":
            v = _D[state]
        elif sym in _ONSET:
            if k != 0:
                raise ValueError(f"onset archiphoneme inside template {template!r}")
            out.append(_ONSET[sym][cls])
            continue
        else:
            out.append(sym)
            continue
        out.append(v)
        state = _after(v[-1], v[-2] if len(v) > 1 else "")
    return "".join(out)


def realize(template: str, left: str):
    state, cls = harmony_state(left), left_class(left)
    if state is None or cls is None:
        return None
    return realize_state(template, state, cls)
