"""Helpers shared by the evaluation scripts."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from yakut_morph_tokenizer.tokenizer import SakhaTokenizer  # noqa: E402


def load_tok(tok_dir):
    """Tokenizer of a directory made by build_tokenizers.py (with its segmenter, if any)."""
    args = json.load(open(os.path.join(tok_dir, "build_args.json")))
    seg = None
    if args.get("seg_table"):
        from build_tokenizers import load_segmenter
        seg = load_segmenter(args["seg_table"], args.get("seg_model"), args.get("stem_only", False))
    return SakhaTokenizer.load(tok_dir, seg)


def read_vocab(data_dir):
    v = {}
    for line in open(os.path.join(data_dir, "vocab.tsv"), encoding="utf-8"):
        w, c = line.rstrip("\n").split("\t")
        v[w] = int(c)
    return v


def read_docs(data_dir, part):
    return [l.rstrip("\n") for l in open(os.path.join(data_dir, part + ".txt"), encoding="utf-8")]
