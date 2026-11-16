"""Suffix inventory and morphotactics of Sakha, written as templates
(see `phonology`).  Each entry is (name, template, left condition, category,
need_next):

    left condition  V = only after a vowel, C = only after a consonant, * = any
    need_next       a set of characters the *following* suffix must start with
                    (for stem-final alternants such as -быт ~ -быппын), or None
"""
from __future__ import annotations

from .phonology import CLASSES, STATES, V_HIGH, V_LOW, VOWELS, realize_state

ANYV = frozenset(VOWELS)

# name, template, left, category, need_next
_RAW = [
    # ---- number
    ("PL", "LAр", "*", "PL", None),
    # ---- possession, nominative
    ("P1SG", "м", "V", "POSS", None), ("P1SG", "Iм", "C", "POSS", None),
    ("P2SG", "ҥ", "V", "POSS", None), ("P2SG", "Iҥ", "C", "POSS", None),
    ("P3SG", "тA", "V", "POSS", None), ("P3SG", "A", "C", "POSS", None),
    ("P1PL", "BIт", "*", "POSSPL", None), ("P2PL", "GIт", "*", "POSSPL", None),
    # ---- case, simple declension
    ("ACC", "нI", "V", "CASE", None), ("ACC", "I", "C", "CASE", None),
    ("DAT", "GA", "*", "CASE", None),
    ("PART", "TA", "*", "CASE", None),
    ("ABL", "ттAн", "V", "CASE", None), ("ABL", "тAн", "C", "CASE", None),
    ("INS", "нAн", "V", "CASE", None), ("INS", "IнAн", "C", "CASE", None),
    ("COM", "LIIн", "*", "CASE", None),
    ("CMP", "TAAҕAр", "*", "CASE", None),
    # ---- case fused with 1sg / 2sg possession
    ("P1SG.ACC", "BIн", "*", "FUS", None), ("P1SG.DAT", "BAр", "*", "FUS", None),
    ("P1SG.PART", "BIнA", "*", "FUS", None), ("P1SG.ABL", "BIттAн", "*", "FUS", None),
    ("P1SG.INS", "BIнAн", "*", "FUS", None), ("P1SG.COM", "BIнIIн", "*", "FUS", None),
    ("P1SG.CMP", "BIнAAҕAр", "*", "FUS", None),
    ("P2SG.ACC", "GIн", "*", "FUS", None), ("P2SG.DAT", "GAр", "*", "FUS", None),
    ("P2SG.PART", "GIнA", "*", "FUS", None), ("P2SG.ABL", "GIттAн", "*", "FUS", None),
    ("P2SG.INS", "GIнAн", "*", "FUS", None), ("P2SG.COM", "GIнIIн", "*", "FUS", None),
    ("P2SG.CMP", "GIнAAҕAр", "*", "FUS", None),
    # ---- possessive declension (3sg possession; also after -BIт / -GIт / plural)
    ("PX.ACC", "тIн", "V", "OBL", None), ("PX.ACC", "Iн", "C", "OBL", None),
    ("PX.DAT", "тIгAр", "V", "OBL", None), ("PX.DAT", "IгAр", "C", "OBL", None),
    ("PX.PART", "тIнA", "V", "OBL", None), ("PX.PART", "IнA", "C", "OBL", None),
    ("PX.ABL", "тIттAн", "V", "OBL", None), ("PX.ABL", "IттAн", "C", "OBL", None),
    ("PX.INS", "тIнAн", "V", "OBL", None), ("PX.INS", "IнAн", "C", "OBL", None),
    ("PX.COM", "тIнIIн", "V", "OBL", None), ("PX.COM", "IнIIн", "C", "OBL", None),
    ("PX.CMP", "тIнAAҕAр", "V", "OBL", None), ("PX.CMP", "IнAAҕAр", "C", "OBL", None),
    # ---- predicative person
    ("PRED1SG", "BIн", "*", "PRED", None), ("PRED2SG", "GIн", "*", "PRED", None),
    # ---- nominal derivation
    ("PROP", "LAAх", "*", "DERN", None), ("PROP", "LAAҕ", "*", "DERN", ANYV),
    ("LOCADJ", "TAAҕI", "*", "DERN", None),
    ("ADVZ", "LII", "*", "DERN", None),
    ("AGENT", "сIт", "C", "DERN", None), ("AGENT", "һIт", "V", "DERN", None),
    # ---- verbal derivation and voice
    ("VBLZ", "LAA", "*", "VDER", None),
    ("CAUS", "TAр", "*", "VOICE", None),
    ("PASS", "IлIн", "C", "VOICE", None), ("PASS", "лIн", "V", "VOICE", None),
    ("PASS", "IлI", "C", "VOICE", frozenset("нмҥл")), ("PASS", "лI", "V", "VOICE", frozenset("нмҥл")),
    # ---- negation
    ("NEG", "BA", "*", "NEG", None),
    # ---- participles (nominal bases)
    ("PTCP.PST", "BIт", "*", "PTCP", None),
    ("PTCP.PST", "BIп", "*", "PTCP", frozenset("п")), ("PTCP.PST", "BIк", "*", "PTCP", frozenset("к")),
    ("PTCP.RES", "TAх", "*", "PTCP", None), ("PTCP.RES", "TAҕ", "*", "PTCP", ANYV),
    ("PTCP.FUT", "Dх", "C", "PTCP", None), ("PTCP.FUT", "Dҕ", "C", "PTCP", ANYV),
    ("PTCP.AOR", "Aр", "C", "PTCP", None), ("PTCP.AOR", "IIр", "C", "PTCP", None),
    ("PTCP.NEG", "т", "V", "NEGT", None),
    ("PTCP.NEG", "п", "V", "NEGT", frozenset("п")), ("PTCP.NEG", "к", "V", "NEGT", frozenset("к")),
    ("PTCP.HAB", "AAччI", "C", "PTCP", None),
    ("NMLZ", "II", "C", "PTCP", None),
    ("OBLIG", "DхтAAх", "C", "PTCP", None), ("OBLIG", "DхтAAҕ", "C", "PTCP", ANYV),
    # ---- finite forms
    ("PST", "TI", "*", "PAST", None),
    ("PST.3SG", "TA", "*", "VFIN", None),
    ("COND", "TAр", "*", "COND", None),
    ("PRS", "A", "C", "PRES", None), ("PRS", "II", "C", "PRES", None),
    ("PRS.3PL", "AллAр", "C", "VFIN", None), ("PRS.3PL", "IIллAр", "C", "VFIN", None),
    ("FUT.3SG", "D", "C", "VFIN", None),
    ("IMP.3SG", "TIн", "*", "VFIN", None), ("IMP.3PL", "TIннAр", "*", "VFIN", None),
    ("IMP.2PL", "Iҥ", "C", "VFIN", None), ("IMP.2PL", "ҥ", "V", "VFIN", None),
    ("IMP.1PL", "DҕIҥ", "C", "VFIN", None),
    # ---- converbs
    ("CVB.SEQ", "Aн", "C", "CVB", None), ("CVB.SEQ", "н", "V", "CVB", None),
    ("CVB.SIM", "A", "C", "CVB", None), ("CVB.SIM", "II", "C", "CVB", None),
    ("CVB.NEG", "ккA", "V", "NEGCVB", None),
    ("CVB.PURP", "AAрI", "C", "CVB", None),
    ("CVB.IMM", "AAт", "C", "CVB", None),
    ("CVB.PRIOR", "BIччA", "*", "CVB", None),
]

NOMINAL_BASE = {"ROOT", "PL", "DERN", "PTCP", "NEGT"}
VERBAL_BASE = {"ROOT", "VDER", "VOICE"}
# category -> categories that may stand immediately to its left
PREV = {
    "PL": {"ROOT", "DERN", "PTCP", "NEGT", "PAST", "COND"},
    "POSS": NOMINAL_BASE | {"PAST"},
    "POSSPL": NOMINAL_BASE | {"PAST", "PRES", "COND"},
    "CASE": NOMINAL_BASE,
    "FUS": NOMINAL_BASE,
    "OBL": NOMINAL_BASE | {"POSSPL"},
    "PRED": {"ROOT", "DERN", "PTCP", "NEGT", "PRES", "COND"},
    "DERN": {"ROOT", "DERN", "PTCP", "NEGT"},
    "VDER": {"ROOT"},
    "VOICE": VERBAL_BASE,
    "NEG": VERBAL_BASE,
    "PTCP": VERBAL_BASE,
    "NEGT": {"NEG"},
    "NEGCVB": {"NEG"},
    "PAST": VERBAL_BASE | {"NEG"},
    "VFIN": VERBAL_BASE | {"NEG"},
    "COND": VERBAL_BASE | {"NEG"},
    "PRES": VERBAL_BASE,
    "CVB": VERBAL_BASE,
}
# forms that are possible after negation -BA only in these names
NEG_OK = {"PTCP.RES", "PST", "PST.3SG", "COND", "IMP.3SG", "IMP.3PL", "PTCP.NEG", "CVB.NEG"}
NONFINAL = {"PAST", "PRES"}


class Morph:
    __slots__ = ("id", "name", "template", "left", "cat", "need_next")

    def __init__(self, i, name, template, left, cat, need_next):
        self.id, self.name, self.template = i, name, template
        self.left, self.cat, self.need_next = left, cat, need_next

    def __repr__(self):
        return f"<{self.name} -{self.template}>"


MORPHS = [Morph(i, *r) for i, r in enumerate(_RAW)]


def build_surface_index():
    """surface -> list of (morph, frozenset of (state, class) contexts that yield it)."""
    idx = {}
    for m in MORPHS:
        per = {}
        for st in STATES:
            for cl in CLASSES:
                is_v = cl in (V_LOW, V_HIGH)
                if (m.left == "V" and not is_v) or (m.left == "C" and is_v):
                    continue
                per.setdefault(realize_state(m.template, st, cl), set()).add((st, cl))
        for surf, ctx in per.items():
            idx.setdefault(surf, []).append((m, frozenset(ctx)))
    return idx
