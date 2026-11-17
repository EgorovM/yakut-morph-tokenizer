#!/bin/bash
# Reproduce the whole study.
#
#   bash experiments/run_all.sh RAW_DIR WORK_DIR
#
# RAW_DIR   one sub-directory per source (kyym, eder_saas, sakha_sire, uluus_media),
#           one UTF-8 text file per document
# WORK_DIR  everything is written here: data/ seg/ morfessor/ tok/ runs/ eval/ results/
#
# One NVIDIA H200: about 10 hours.  Set JOBS to the number of models trained at once.
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

# 5. ablations: segmentation depth, training-set size, model size
$P $B data tok/stem8k --kind bpe --vocab 8000 $S --stem_only
$P $X/train_segmenter.py data seg_g0 --depth_bonus 0
$P $B data tok/nog8k --kind bpe --vocab 8000 --seg_table seg_g0/table.tsv.gz --seg_model seg_g0/segmenter.json.gz
for n in 10 100; do
  d=$((100 / n)); v=$([ $n = 10 ] && echo 8 || echo 4)
  $P $X/subset_corpus.py data data_$d --every $n
  $P $X/train_segmenter.py data_$d seg_$d
  $P $B data_$d tok/bpe${v}k-d$d --kind bpe --vocab ${v}000
  $P $B data_$d tok/morph${v}k-d$d --kind bpe --vocab ${v}000 --seg_table seg_$d/table.tsv.gz --seg_model seg_$d/segmenter.json.gz
done
abl() { t=$1; r=$2; s=$3; shift 3
  $PYTHON $X/train_lm.py tok/$t runs/${r}_s$s --seed $s "$@" > logs/run_${r}_s$s.log 2>&1
  bash $X/eval_all.sh $t ${r}_s$s > logs/eval_${r}_s$s.log 2>&1; }
export -f abl
{ for t in bpe8k morph8k; do
    echo "$t $t-S 0 --layers 4 --d 256 --heads 4"
    echo "$t $t-L 0 --layers 12 --d 768 --heads 12 --lr 6e-4"
  done
  for s in 0 1 2; do
    for t in stem8k nog8k; do echo "$t $t $s"; done
    for t in bpe8k-d10 morph8k-d10; do echo "$t $t $s --batch 32 --epochs 30 --dropout 0.1"; done
    for t in bpe4k-d1 morph4k-d1; do echo "$t $t $s --batch 16 --epochs 60 --dropout 0.1 --warmup 200"; done
  done; } | xargs -P $JOBS -L 1 bash -c 'abl "$@"' _

# 6. translation (optional): put the parallel corpus into mt/raw/parallel.parquet (see README)
if [ -f mt/raw/parallel.parquet ]; then
  $P $X/mt_prep.py mt/raw/parallel.parquet mt/pairs
  mt() { r=mt/runs/$1_$2_$3_s$4; i=""; [ $3 = init ] && i="--init runs/$2_s$5"
    $PYTHON $X/mt.py mt/pairs tok/$2 $r --dir $1 --seed $4 --epochs 30 --dropout 0.2 $i > logs/mt_$1_$2_$3_s$4.log 2>&1; }
  export -f mt
  { for d in ru2sah sah2ru; do
      for t in bpe8k mfs8k morph8k; do
        for s in 0 1 2 3 4; do echo "$d $t scratch $s 0"; done
        for s in 0 1 2; do echo "$d $t init $s $s"; done
      done
      for t in bpe2k morph2k bpe32k morph32k; do for s in 0 1 2; do echo "$d $t init $s 0"; done; done
    done; } | xargs -P $JOBS -L 1 bash -c 'mt "$@"' _
fi

# 7. tables and figures
$P $X/make_tables.py . results
$P $X/corpus_stats.py . results/corpus_stats.json
