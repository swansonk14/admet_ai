# Chemprop ADMET
Training and prediction scripts for [Chemprop](https://github.com/chemprop/chemprop) models trained on ADMET datasets from the Therapeutics Data Commons ([TDC](https://tdcommons.ai/)).


## Installation

1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) if you don't already have it.
2. Create a conda environment: `conda create -y -n chemprop_admet python=3.10`
3. Activate the conda environment: `conda activate chemprop_admet`
4. Install requirements: `pip install -r requirements.txt`


## TODO: pre-trained models

TODO


## Download TDC ADMET data

Download the [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) data for evaluating models using scaffold splits in order to compare to the TDC leaderboard.

```bash
python tdc_admet_group_prepare.py \
    --save_dir data/tdc_admet_group
```

Download all TDC [ADME](https://tdcommons.ai/single_pred_tasks/adme/) and [Tox](https://tdcommons.ai/single_pred_tasks/tox/) datasets for training models.

```bash
python tdc_admet_all_prepare.py \
    --save_dir data/tdc_admet_all
```


## Compute RDKit features

Compute RDKit features in order to train Chemprop-RDKit models (i.e., Chemprop models augmented with 200 molecular features from RDKit).

Compute RDKit features for the TDC ADMET Benchmark Group data.
```bash
python rdkit_tdc_admet_group.py \
    --data_dir data/tdc_admet_group
```

Compute RDKit features for all TDC ADMET datasets.
```bash
python rdkit_tdc_admet_all.py \
    --data_dir data/tdc_admet_all
```


## Train Chemprop ADMET predictors


## Make predictions with Chemprop ADMET predictors

