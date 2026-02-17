#!/usr/bin/env bash

# Calculate wackiness scores for SPLADE-v2

python3 scripts/calculate_wackiness_scores.py \
  --model-name "naver/splade_v2_max" \
  --index-dir "./experiments/splade-v2-max/msmarco/index/" \
  --token-freqs-path "./data/idfs/msmarco_splade_v2_idfs.json" \
  --token-scores-save-path "./experiments/wackiness_scores/splade_v2_original_wackiness_scores.json" \
  --sample-size 10000 \
  --batch-size 100 \
  --top-k 100 \
  --seed 42 \

###########################################################################################

# Calculate wackiness scores for SPLADE-v3.

python3 scripts/calculate_wackiness_scores.py \
  --model-name "naver/splade-v3" \
  --index-dir "./experiments/splade-v3/msmarco/index/" \
  --token-freqs-path "./data/idfs/msmarco_splade_v3_idfs.json" \
  --token-scores-save-path "./experiments/wackiness_scores/splade_v3_original_wackiness_scores.json" \
  --sample-size 10000 \
  --batch-size 100 \
  --top-k 100 \
  --seed 42 \