#!/usr/bin/env bash

###############################################################################

# Evaluate SPLADE-v2 (Original)

# Define config to use
export SPLADE_CONFIG_NAME="config_splade++_max.yaml"

# Build the MSMARCO index
python3 -m splade.index \
    init_dict.model_type_or_dir=naver/splade_v2_max \
    config.pretrained_no_yamlconfig=true \
    config.index_dir=experiments/splade-v2-max/msmarco/index \
    config.index_retrieve_batch_size=128

# Run retrieval on MSMARCO
python3 -m splade.retrieve \
    init_dict.model_type_or_dir=naver/splade_v2_max \
    config.pretrained_no_yamlconfig=true \
    config.index_dir=experiments/splade-v2-max/msmarco/index \
    config.out_dir=experiments/splade-v2-max/msmarco/out \

###############################################################################

# Evaluate SPLADE-v3 (Original)

# Define config to use
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil"

# Build the MSMARCO index
python3 -m splade.index \
    init_dict.model_type_or_dir=naver/splade-v3 \
    config.pretrained_no_yamlconfig=true \
    config.index_dir=experiments/splade-v3/msmarco/index \
    config.index_retrieve_batch_size=50

# Run retrieval on MSMARCO
python3 -m splade.retrieve \
    init_dict.model_type_or_dir=naver/splade-v3 \
    config.pretrained_no_yamlconfig=true \
    config.index_dir=experiments/splade-v3/msmarco/index \
    config.out_dir=experiments/splade-v3/msmarco/out_3

###############################################################################

# # Evaluate SPLADE-v2 (Reproduction)

# # Define config to use
# export SPLADE_CONFIG_NAME="config_splade++_max.yaml"

# # Build the MSMARCO index
# python3 -m splade.index \
#     config.checkpoint_dir=./models/splade_v2_repro \
#     config.index_dir=./experiments/splade_v2_repro/msmarco/index \
#     config.index_retrieve_batch_size=50

# # Run retrieval on MSMARCO
# python3 -m splade.retrieve \
#     config.checkpoint_dir=./models/splade_v2_repro \
#     config.index_dir=./experiments/splade_v2_repro/msmarco/index \
#     config.out_dir=./experiments/splade_v2_repro/msmarco/out

# ###############################################################################

# # Evaluate SPLADE-v2-L1

# # Define config to use
# export SPLADE_CONFIG_NAME="config_splade++_max_.yaml"

# # Build the MSMARCO index
# python3 -m splade.index \
#     config.checkpoint_dir=./models/splade_v2_l1 \
#     config.index_dir=./experiments/splade_v2_l1/msmarco/index \
#     config.index_retrieve_batch_size=50

# # Run retrieval on MSMARCO
# python3 -m splade.retrieve \
#     config.checkpoint_dir=./models/splade_v2_l1 \
#     config.index_dir=./experiments/splade_v2_l1/msmarco/index \
#     config.out_dir=./experiments/splade_v2_l1/msmarco/out

# ###############################################################################

# # Evaluate SPLADE-v2-DistilRoBERTa

# # Define config to use
# export SPLADE_CONFIG_NAME="config_splade++_max_distilroberta.yaml"

# # Build the MSMARCO index
# python3 -m splade.index \
#     config.checkpoint_dir=./models/splade_v2_distilroberta \
#     config.index_dir=./experiments/splade_v2_distilroberta/msmarco/index \
#     config.index_retrieve_batch_size=50
    
# # Run retrieval on MSMARCO
# python3 -m splade.retrieve \
#     config.checkpoint_dir=./models/splade_v2_distilroberta \
#     config.index_dir=./experiments/splade_v2_distilroberta/msmarco/index \
#     config.out_dir=./experiments/splade_v2_distilroberta/msmarco/out


# ###############################################################################

# # Evaluate SPLADE-v2-ModernBERT

# # Define config to use
# export SPLADE_CONFIG_NAME="config_splade++_max_modernbert.yaml"

# # Build the MSMARCO index
# python3 -m splade.index \
#     config.checkpoint_dir=./models/splade_v2_modernbert \
#     config.index_dir=./experiments/splade_v2_modernbert/msmarco/index \
#     config.index_retrieve_batch_size=50

# # Run retrieval on MSMARCO
# python3 -m splade.retrieve \
#     config.checkpoint_dir=./models/splade_v2_modernbert \
#     config.index_dir=./experiments/splade_v2_modernbert/msmarco/index \
#     config.out_dir=./experiments/splade_v2_modernbert/msmarco/out
