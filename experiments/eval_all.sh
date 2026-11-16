#!/bin/bash
# Run every evaluation for one trained model:  eval_all.sh TOK_NAME RUN_NAME
# (expects the layout of run_all.sh: data/, seg/, tok/<name>, runs/<name>, eval/)
set -e
P=${PYTHON:-python}; X=$(dirname "$0"); T=tok/$1; R=runs/$2
$P $X/eval_bpc.py data $T $R
$P $X/eval_pairs.py eval/pairs.jsonl $T $R
$P $X/eval_wordpred.py data $T $R
$P $X/eval_generate.py data seg $T $R
[ -f eval/nbest.json.gz ] && $P $X/eval_asr.py eval/nbest.json.gz $T $R
echo EVAL_DONE $2
