#!/bin/bash
# Reproduce the whole study.
#
#   bash experiments/run_all.sh RAW_DIR WORK_DIR
#
# RAW_DIR   one sub-directory per source (kyym, eder_saas, sakha_sire, uluus_media),
#           one UTF-8 text file per document
# WORK_DIR  everything is written here: data/ seg/ morfessor/ tok/ runs/ eval/ results/
#
# One NVIDIA H200: about 4 hours.  Set JOBS to the number of models trained at once.
set -e
X=$(cd "$(dirname "$0")" && pwd); RAW=$(cd "$1" && pwd); mkdir -p "$2"; cd "$2"
export PYTHON=${PYTHON:-python}; P=$PYTHON; JOBS=${JOBS:-4}
mkdir -p logs tok runs eval results
[ -f "$X/../data/external/unimorph_sah.tsv" ] || bash "$X/get_unimorph.sh"

# 1. corpus, segmenters
$P $X/prep_corpus.py "$RAW" data
$P $X/train_segmenter.py data seg
$P $X/train_morfessor.py data morfessor

# 2. tokenizers
B=$X/build_tokenizers.py; S="--seg_table seg/table.tsv.gz --seg_model seg/segmenter.json.gz"
$P $B data tok/char --kind char
for v in 2 4 8 16 32; do
  $P $B data tok/bpe${v}k --kind bpe --vocab ${v}000
  $P $B data tok/morph${v}k --kind bpe --vocab ${v}000 $S
done
$P $B data tok/uni8k --kind unigram --vocab 8000
$P $B data tok/unimorph8k --kind unigram --vocab 8000 $S
$P $B data tok/mfs8k --kind bpe --vocab 8000 --seg_table morfessor/table.tsv.gz

# 3. evaluation material that does not depend on a model
$P $X/build_pairs.py data eval/pairs.jsonl
$P $X/eval_generate.py data seg --human --out eval/human_generate.json
$P $X/eval_unimorph.py seg data --part test --table morfessor=morfessor/table.tsv.gz \
   --tok tok/bpe8k tok/uni8k tok/bpe32k --out eval/unimorph_test.json
# optional: put speech-recognition n-best lists into eval/nbest.json.gz (see README)

# 4. language models and their evaluation
job() {
  t=$1; s=$2; shift 2; r=${t}_s$s
  $PYTHON $X/train_lm.py tok/$t runs/$r --seed $s "$@" > logs/run_$r.log 2>&1
  bash $X/eval_all.sh $t $r > logs/eval_$r.log 2>&1
}
export -f job; export X
{ echo "char 0 --ctx 1024 --batch 64"
  for t in bpe2k morph2k bpe4k morph4k bpe16k morph16k bpe32k morph32k uni8k unimorph8k; do echo "$t 0"; done
  for s in 0 1 2; do for t in bpe8k morph8k mfs8k; do echo "$t $s"; done; done
} | xargs -P $JOBS -L 1 bash -c 'job "$@"' _

# 5. tables and figures
$P $X/make_tables.py . results
$P $X/corpus_stats.py . results/corpus_stats.json
