#!/usr/bin/env bash

# Reproducing SPLADE-v2

export SPLADE_CONFIG_NAME="config_splade++_max.yaml"
python3 -m splade.all \
    config.train_batch_size=128 \
    config.eval_batch_size=128 \
    config.index_retrieve_batch_size=128 \
    config.checkpoint_dir=models/splade_v2_max/checkpoint \
    config.index_dir=models/splade_v2_max/index \
    config.out_dir=models/splade_v2_max/out

###########################################################################################

# SPLADE-v2-L1

export SPLADE_CONFIG_NAME="config_splade++_max_l1.yaml"
python3 -m splade.all \
    config.train_batch_size=128 \
    config.eval_batch_size=128 \
    config.index_retrieve_batch_size=128 \
    config.checkpoint_dir=models/splade_v2_l1/checkpoint \
    config.index_dir=models/splade_v2_l1/index \
    config.out_dir=models/splade_v2_l1/out

###########################################################################################

# SPLADE-v2-ModernBERT

export SPLADE_CONFIG_NAME="config_splade++_max_modernbert.yaml"
python3 -m splade.all \
    config.train_batch_size=128 \
    config.eval_batch_size=128 \
    config.index_retrieve_batch_size=128 \
    config.checkpoint_dir=models/splade_v2_modernbert/checkpoint \
    config.index_dir=models/splade_v2_modernbert/index \
    config.out_dir=models/splade_v2_modernbert/out

###########################################################################################

# SPLADE-v2-DistilRoBERTa

export SPLADE_CONFIG_NAME="config_splade++_max_distilroberta.yaml"
python3 -m splade.all \
    config.train_batch_size=128 \
    config.eval_batch_size=128 \
    config.index_retrieve_batch_size=128 \
    config.checkpoint_dir=models/splade_v2_distilroberta/checkpoint \
    config.index_dir=models/splade_v2_distilroberta/index \
    config.out_dir=models/splade_v2_distilroberta/out

###########################################################################################

# SPLADE-v2-Sum

export SPLADE_CONFIG_NAME="config_splade++_sum.yaml"
python3 -m splade.all \
    config.train_batch_size=128 \
    config.eval_batch_size=128 \
    config.index_retrieve_batch_size=128 \
    config.checkpoint_dir=models/splade_v2_sum/checkpoint \
    config.index_dir=models/splade_v2_sum/index \
    config.out_dir=models/splade_v2_sum/out

###########################################################################################

# SPLADE-v2-CLS

export SPLADE_CONFIG_NAME="config_splade++_cls.yaml"
python3 -m splade.all \
    config.train_batch_size=128 \
    config.eval_batch_size=128 \
    config.index_retrieve_batch_size=128 \
    config.checkpoint_dir=models/splade_v2_cls/checkpoint \
    config.index_dir=models/splade_v2_cls/index \
    config.out_dir=models/splade_v2_cls/out