# yakut-morph-tokenizer

A morphological segmenter and a morphology-aware subword tokenizer for Sakha (Yakut),
with the code that reproduces the paper

> М. П. Егоров. Влияние морфемной токенизации на качество декодирования нейросетевых
> языковых моделей якутского языка.

The segmenter needs no annotated data and no stem dictionary. Hand-written rules
(vowel harmony, consonant assimilation, suffix order) list every licensed analysis of a
word form; a probabilistic model fitted with EM on a plain word list picks one.
The tokenizer is BPE whose merges never cross a morph boundary.

```
оҕолорбутуттан   ->  оҕо | лор | бут | уттан        (child-PL-1PL-ABL)
дьиэлэрбитигэр   ->  дьиэ | лэр | бит | игэр        (house-PL-1PL-DAT)
```

## Install

```bash
pip install -e .            # the package: needs only `tokenizers`
pip install -e ".[test]" && pytest -q
```

## Use

```python
from yakut_morph_tokenizer import Segmenter

seg = Segmenter()
table = seg.fit({"оҕо": 50, "оҕолор": 20, "оҕолорго": 4, ...})   # word -> corpus frequency
seg.segment("оҕолорго")        # ('оҕо', 'лор', 'го')
seg.analyse("оҕолорго")        # ('оҕо', [('лор', 'PL'), ('го', 'DAT')])
seg.save("segmenter.json.gz"); seg = Segmenter.load("segmenter.json.gz")
```

A segmenter is only as good as the word list it was fitted on: fit it on the vocabulary
of your own corpus (`experiments/train_segmenter.py`, two minutes for 670k word types).

Building a tokenizer and encoding text:

```bash
python experiments/build_tokenizers.py DATA TOK/morph8k --kind bpe --vocab 8000 \
    --seg_table SEG/table.tsv.gz --seg_model SEG/segmenter.json.gz
```

```python
from experiments.common import load_tok
tok = load_tok("TOK/morph8k")
tok.tokens("Оҕолор дьиэлэригэр бардылар.")
tok.decode(tok.encode("Оҕолор дьиэлэригэр бардылар."))   # lossless up to normalisation
```

## Layout

| path | content |
|---|---|
| `src/yakut_morph_tokenizer/phonology.py` | harmony states, assimilation classes, realisation of suffix templates |
| `src/yakut_morph_tokenizer/morphs.py` | 99 suffix templates for 68 morphemes, suffix order |
| `src/yakut_morph_tokenizer/segmenter.py` | candidate analyses, leave-one-out EM, inference |
| `src/yakut_morph_tokenizer/tokenizer.py` | char / BPE / Unigram tokenizers with optional morph boundaries |
| `experiments/` | corpus preparation, training, the five evaluations, tables and figures |
| `paper/` | `make_numbers.py` turns `results/` into `numbers.tex`: every number and table body of the paper |
| `results/` | `summary.json`, `corpus_stats.json`, figures |

## Reproducing the paper

```bash
pip install -r requirements-experiments.txt
bash experiments/run_all.sh RAW_DIR WORK_DIR     # about 10 hours on one NVIDIA H200
python paper/make_numbers.py                     # numbers and tables of the paper
```

`RAW_DIR` holds one sub-directory per source (`kyym`, `eder_saas`, `sakha_sire`,
`uluus_media`) with one UTF-8 text file per news article. The articles belong to their
publishers and are not redistributed here. `prep_corpus.py` is deterministic: the split
is a function of the document text, so the same raw files give the same train / dev / test parts.

Steps of `run_all.sh`:

1. `prep_corpus.py` cleans, de-duplicates and splits the corpus.
2. `train_segmenter.py` fits the segmenter; `train_morfessor.py` fits the unsupervised baseline.
3. `build_tokenizers.py` trains 14 tokenizers and encodes the corpus.
4. `train_lm.py` trains a decoder-only Transformer (8 layers, d = 512) for 10 epochs per tokenizer.
5. `eval_all.sh` runs the evaluations of one model:
   - `eval_bpc.py`: bits per character on the test part, also by word frequency;
   - `eval_pairs.py`: word-form minimal pairs (`build_pairs.py` makes them);
   - `eval_wordpred.py`: next-word prediction with beam search;
   - `eval_generate.py`: share of non-words in sampled continuations;
   - `eval_asr.py`: rescoring of speech-recognition n-best lists (optional).
6. `eval_unimorph.py` scores segmentations against UniMorph stem boundaries.
7. Ablations: one boundary per word (`--stem_only`), no depth bonus, 10 % and 1 % of the
   training corpus (`subset_corpus.py`), a smaller and a larger model.
8. `mt_prep.py` and `mt.py`: Russian-Sakha translation in both directions with the same
   decoder-only model, from scratch and from the language-model weights (optional).
9. `make_tables.py` collects everything and runs the paired bootstrap tests.

The translation experiment needs `WORK_DIR/mt/raw/parallel.parquet`: the
[parallel corpus of the Olonkho Institute, NEFU](https://huggingface.co/datasets/averoo/sakha-russian-parallel-corpora-all)
(CC BY 4.0), file `data/train-00000-of-00001.parquet`. It is skipped when the file is absent.

The speech-recognition experiment needs `WORK_DIR/eval/nbest.json.gz`: a list of
`{"ref": str, "nbest": [[hypothesis, acoustic_score], ...]}`. The lists used in the paper
come from a held-out test set of spontaneous speech and are not public; the script
accepts any lists in this format and is skipped when the file is absent.

## Data credits

Segmentation is evaluated on [UniMorph `sah`](https://github.com/unimorph/sah), derived
from [apertium-sah](https://github.com/apertium/apertium-sah). It is downloaded by
`experiments/get_unimorph.sh` and is used for evaluation only: no rule, parameter or
dictionary of the segmenter comes from it, apart from two hyper-parameters chosen on
one half of its lemmas.

## License

MIT (code).
