#!/usr/bin/env bash

# Calculate document frequency values for SPLADE-v2 tokenizer on the MSMARCO collection
python scripts/calculate_dfs.py \
    --model_name naver/splade_v2_max \
    --original_collection data/msmarco/corpus.jsonl \
    --output_path data/idfs/msmarco_splade_v2_idfs_v2.json \
    --batch_size 10000 \
    --num_threads 16

# Calculate document frequency values for SPLADE-v3 tokenizer on the MSMARCO collection
python scripts/calculate_dfs.py \
    --model_name naver/splade-v3 \
    --original_collection data/msmarco/corpus.jsonl \
    --output_path data/idfs/msmarco_splade_v3_idfs_v2.json \
    --batch_size 10000 \
    --num_threads 16