# Chemprop ADMET
Training and prediction scripts for [Chemprop](https://github.com/chemprop/chemprop) models trained on ADMET datasets from the Therapeutics Data Commons ([TDC](https://tdcommons.ai/)).


## Installation

1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) if you don't already have it.
2. Create a conda environment: `conda create -y -n chemprop_admet python=3.10`
3. Activate the conda environment: `conda activate chemprop_admet`
4. Install requirements: `pip install -r requirements.txt`


## TODO: Downlading and applying trained models

TODO


## Download TDC ADMET data

Download the [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) data for evaluating models using scaffold splits in order to compare to the TDC leaderboard.

```bash
python prepare_tdc_admet_group.py \
    --save_dir data/tdc_admet_group
```

Download all TDC [ADME](https://tdcommons.ai/single_pred_tasks/adme/) and [Tox](https://tdcommons.ai/single_pred_tasks/tox/) datasets for training models. Skip datasets that are redundant or not needed.

```bash
python prepare_tdc_admet_all.py \
    --save_dir data/tdc_admet_all \
    --skip_datasets herg_central hERG_Karim ToxCast
```

## Create multitask datasets for regression and classification

Create multitask datasets for regression and classification for all the TDC ADMET datasets.

```bash
python merge_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir data/tdc_admet_all_multitask
```


## Compute RDKit features

Compute RDKit features in order to train Chemprop-RDKit models (i.e., Chemprop models augmented with 200 molecular features from RDKit).

Compute RDKit features for the TDC ADMET Benchmark Group data.

```bash
python rdkit_tdc_admet_group.py \
    --data_dir data/tdc_admet_group
```

Compute RDKit features for TDC ADMET multitask datasets.

```bash
python rdkit_tdc_admet_all.py \
    --data_dir data/tdc_admet_all_multitask
```


## Train Chemprop ADMET predictors

Train Chemprop and Chemprop-RDKit predictors on the ADMET data.

Train Chemprop ADMET predictors on the TDC ADMET Benchmark Group data.
```bash
python train_tdc_admet_group.py \
    --data_dir data/tdc_admet_group \
    --save_dir models/tdc_admet_group \
    --model_type chemprop
```

Train Chemprop-RDKit ADMET predictors on the TDC ADMET Benchmark Group data.
```bash
python train_tdc_admet_group.py \
    --data_dir data/tdc_admet_group \
    --save_dir models/tdc_admet_group \
    --model_type chemprop_rdkit
```

Train Chemprop ADMET predictors on the TDC ADMET multitask datasets.
```bash
python train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop
```

Train Chemprop-RDKit ADMET predictors on the TDC ADMET multitask datasets.
```bash
python train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop_rdkit
```


## Make predictions with Chemprop ADMET predictors

The instructions below illustrate how to make predictions with trained Chemprop ADMET predictors. The instructions assume that you have a file called `data.csv` which contains SMILES strings in a column called `smiles`.

Make predictions with Chemprop ADMET predictors trained on the TDC ADMET Benchmark Group data.
```bash
python predict_tdc_admet.py \
    --data_path data.csv \
    --save_path preds.csv \
    --model_dir models/tdc_admet_group/chemprop \
    --model_type chemprop \
    --smiles_column smiles
```

Make predictions with Chemprop-RDKit ADMET predictors trained on the TDC ADMET Benchmark Group data.
```bash
python predict_tdc_admet.py \
    --data_path data.csv \
    --save_path preds.csv \
    --model_dir models/tdc_admet_group/chemprop_rdkit \
    --model_type chemprop_rdkit \
    --smiles_column smiles
```

Make predictions with Chemprop ADMET predictors trained on all the TDC ADMET data.
```bash
python predict_tdc_admet.py \
    --data_path data.csv \
    --save_path preds.csv \
    --model_dir models/tdc_admet_all/chemprop \
    --model_type chemprop \
    --smiles_column smiles
```

Make predictions with Chemprop-RDKit ADMET predictors trained on all the TDC ADMET data.
```bash
python predict_tdc_admet.py \
    --data_path data.csv \
    --save_path preds.csv \
    --model_dir models/tdc_admet_all/chemprop_rdkit \
    --model_type chemprop_rdkit \
    --smiles_column smiles
```


## Get approved drugs from DrugBank

Get approved drugs from DrugBank to create a comparison set for Chemprop ADMET predictors.

```bash
python get_drugbank_approved.py \
    --data_path data/drugbank.xml \
    --save_path data/drugbank_approved.csv
```
