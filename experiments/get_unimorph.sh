#!/bin/bash
# UniMorph data for Sakha (derived from apertium-sah); used only for evaluation.
set -e
mkdir -p "$(dirname "$0")/../data/external"
curl -sL https://raw.githubusercontent.com/unimorph/sah/master/sah \
     -o "$(dirname "$0")/../data/external/unimorph_sah.tsv"
wc -l "$(dirname "$0")/../data/external/unimorph_sah.tsv"
