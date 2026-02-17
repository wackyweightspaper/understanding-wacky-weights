## Understanding Wacky Weights: A Dissection of SPLADE’s Learned Term Importance

This repository contains the official code and analysis scripts for the paper:

> **"Understanding Wacky Weights: A Dissection of SPLADE’s Learned Term Importance"**

### Requirements

This project requires **Python 3.11**.

All dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

### Repository Structure

This repository is built upon the official [SPLADE implementation](https://github.com/naver/splade).

* **`splade/`**: Contains the modified source code and core logic.
* **`conf/`**: Contains the configuration files for reproducing the training of different SPLADE variants.
* **`scripts/`**: Contains the scripts to reproduce the training and evaluation of the different SPLADE variants, as well as the calculation of Wackiness Scores.
* **`notebooks/`**: Contains Jupyter notebooks with experiments from the paper
* **`utils/`**: Contains utility functions for evaluation and Wackiness Scores Calculation.

### Reproducing SPLADE’s Training

* Script ``scripts/train.sh`` contains the commands to train the different SPLADE variants.
* Script ``scripts/eval_msmarco.sh`` contains the commands to evaluate the different SPLADE variants on MSMARCO, TREC DL 2019, and TREC DL 2020.

### Calculating Wackiness Scores

The following pipeline describes how to obtain Wackiness Scores for a given SPLADE variant:

1. **Obtain precomputed index:** First, use `scripts/eval_msmarco.sh` to index and evaluate the model on MSMARCO. The precomputed MS MARCO index is required for the next steps.
2. **Precompute document frequencies:** Compute the document frequencies of individual tokens across the MSMARCO index by running `scripts/calculate_dfs.sh`. These frequencies are used to derive inverse document frequency (IDF) values needed for the Wackiness Score calculation.
3. **Run Wackiness Score calculation:** Run `scripts/calculate_wackiness_scores.sh` to compute the Wackiness Scores using the precomputed index and document frequencies.

Sample parameters for some models such as SPLADE-v2 and SPLADE-v3 are provided in the scripts for reference. We will publish checkpoints for other reproduced SPLADE variants and precomputed wackiness scores for them upon deanonymizing this repository.