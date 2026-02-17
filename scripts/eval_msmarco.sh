#!/usr/bin/env bash

###############################################################################

# Evaluate SPLADE-v2 (Reproduction)

# define config to use
export SPLADE_CONFIG_NAME="config_splade++_max.yaml"

# Build the MSMARCO index
python3 -m splade.index \
    config.checkpoint_dir=./models/splade_max/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-flops/msmarco/index \
    config.index_retrieve_batch_size=50

# Run retrieval on MSMARCO
python3 -m splade.retrieve \
    config.checkpoint_dir=./models/splade_max/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-flops/msmarco/index \
    config.out_dir=./experiments/splade-v2-max-reproduce-flops/msmarco/out

###############################################################################

# Evaluate SPLADE-v2-L1

# define config to use
export SPLADE_CONFIG_NAME="config_splade++_max_.yaml"

# Build the MSMARCO index
python3 -m splade.index \
    config.checkpoint_dir=./models/splade_max_l1/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-l1/msmarco/index \
    config.index_retrieve_batch_size=50

# Run retrieval on MSMARCO
python3 -m splade.retrieve \
    config.checkpoint_dir=./models/splade_max_l1/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-l1/msmarco/index \
    config.out_dir=./experiments/splade-v2-max-reproduce-l1/msmarco/out

###############################################################################

# Evaluate SPLADE-v2-DistilRoBERTa

# define config to use
export SPLADE_CONFIG_NAME="config_splade++_max_distilroberta.yaml"

# Build the MSMARCO index
python3 -m splade.index \
    config.checkpoint_dir=./models/splade_max_distilroberta/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-distilroberta/msmarco/index \
    config.index_retrieve_batch_size=50
    
# Run retrieval on MSMARCO
python3 -m splade.retrieve \
    config.checkpoint_dir=./models/splade_max_distilroberta/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-distilroberta/msmarco/index \
    config.out_dir=./experiments/splade-v2-max-reproduce-distilroberta/msmarco/out


###############################################################################

# Evaluate SPLADE-v2-ModernBERT

# define config to use
export SPLADE_CONFIG_NAME="config_splade++_max_modernbert.yaml"

Build the MSMARCO index
python3 -m splade.index \
    config.checkpoint_dir=./models/splade_max_modernbert/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-modernbert/msmarco/index \
    config.index_retrieve_batch_size=50

Run retrieval on MSMARCO
python3 -m splade.retrieve \
    config.checkpoint_dir=./models/splade_max_modernbert/checkpoint \
    config.index_dir=./experiments/splade-v2-max-reproduce-modernbert/msmarco/index \
    config.out_dir=./experiments/splade-v2-max-reproduce-modernbert/msmarco/out
