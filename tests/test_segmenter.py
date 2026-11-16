import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yakut_morph_tokenizer import Segmenter, normalize, pieces  # noqa: E402
from yakut_morph_tokenizer.phonology import realize  # noqa: E402
from yakut_morph_tokenizer.tokenizer import SakhaTokenizer, mark_chunk  # noqa: E402


def test_harmony_and_assimilation():
    # plural -LAр
    assert realize("LAр", "оҕо") == "лор"
    assert realize("LAр", "ат") == "тар"
    assert realize("LAр", "күн") == "нэр"
    assert realize("LAр", "сир") == "дэр"
    assert realize("LAр", "суол") == "лар"       # after the diphthong уо the low vowel is а
    assert realize("LAр", "күөл") == "лэр"
    # dative -GA
    assert realize("GA", "оскуола") == "ҕа"
    assert realize("GA", "сылгы") == "га"
    assert realize("GA", "ат") == "ка"
    assert realize("GA", "күн") == "ҥэ"
    # 1pl -BIт, ablative, comparative
    assert realize("BIт", "оҕолор") == "бут"
    assert realize("BIт", "ат") == "пыт"
    assert realize("ттAн", "оҕо") == "ттон"
    assert realize("TAAҕAр", "сир") == "дээҕэр"


def test_candidates_are_licensed():
    seg = Segmenter()
    cands = {tuple(s for s, _ in chain) for _, chain in seg.candidates("оҕолорбутуттан")}
    assert (3, 6, 9) in cands                      # оҕо|лор|бут|уттан
    # a suffix that breaks vowel harmony is never proposed: *оҕо-лар
    assert all(3 not in c for c in (tuple(s for s, _ in ch) for _, ch in seg.candidates("оҕолар")))


def test_fit_on_toy_vocabulary():
    vocab = {"оҕо": 50, "оҕолор": 20, "оҕолору": 5, "оҕолорго": 4, "оҕону": 6, "оҕоҕо": 5,
             "дьиэ": 40, "дьиэлэр": 9, "дьиэҕэ": 12, "дьиэни": 4, "дьиэлэргэ": 3,
             "киһи": 60, "киһилэр": 10, "киһини": 8, "киһиттэн": 3, "ат": 30, "аттар": 6, "акка": 3}
    seg = Segmenter()
    table = seg.fit(vocab, iters=8)
    assert table["оҕолорго"] == ("оҕо", "лор", "го")
    assert table["дьиэлэргэ"] == ("дьиэ", "лэр", "гэ")
    assert table["киһи"] == ("киһи",)
    assert seg.segment("киһилэргэ") == ("киһи", "лэр", "гэ")     # unseen form of a known stem
    assert seg.lemma("ак", "к") == "ат"                           # акка = ат + ка


def test_normalize_and_pieces():
    assert normalize("  Оҕолор,\n  2024 сылга  ") == "оҕолор, 2024 сылга"
    assert normalize("саhа ғ ң") == "саһа ҕ ҥ"
    assert pieces("«дьиэҕэ»,") == ["«", "дьиэҕэ", "»", ","]
    assert mark_chunk("дьиэҕэ,") == ["▁дьиэҕэ", ","]


def test_char_tokenizer_round_trip():
    text = normalize("Оҕолор дьиэҕэ бардылар.")
    vocab = {t: i for i, t in enumerate(["<pad>", "<bos>", "<unk>", "▁"] + sorted(set(text.replace(" ", ""))))}
    tok = SakhaTokenizer("char", vocab)
    assert tok.decode(tok.encode(text, normalized=True)) == text
