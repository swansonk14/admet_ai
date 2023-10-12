# ADMET-AI

Training and prediction scripts for [Chemprop](https://github.com/chemprop/chemprop) models trained on ADMET datasets from the Therapeutics Data Commons ([TDC](https://tdcommons.ai/)).

TODO: table of contents

TODO: black reformat everything

## Installation

ADMET-AI can be installed in a few minutes on any operating system using pip (optionally within a conda environment).

Optionally, create a conda environment.
```bash
conda create -y -n admet_ai python=3.10
conda activate admet_ai
```

Install ADMET-AI via pip.
```bash
pip install admet_ai
```

Alternatively, clone the repo and install ADMET-AI locally.
```bash
git clone https://github.com/swansonk14/SyntheMol.git
cd SyntheMol
pip install -e .
```

If there are version issues with the required packages, create a conda environment with specific working versions of the packages as follows.
```bash
pip install -r requirements.txt
pip install -e .
```

Note: If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`, run `conda install -c conda-forge xorg-libxrender`.

## TODO: Downlading and applying trained models

TODO


## Download TDC ADMET data

Download the [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) data for evaluating models using scaffold splits in order to compare to the TDC leaderboard.

```bash
python prepare_tdc_admet_group.py \
    --raw_data_dir data/tdc_admet_group_raw \
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
python compute_rdkit_features.py \
    --data_dir data/tdc_admet_group \
    --smiles_column Drug
```

Compute RDKit features for all TDC ADMET datasets.

```bash
python compute_rdkit_features.py \
    --data_dir data/tdc_admet_all \
    --smiles_column smiles
```

Compute RDKit features for TDC ADMET multitask datasets.

```bash
python compute_rdkit_features.py \
    --data_dir data/tdc_admet_all_multitask \
    --smiles_column smiles
```


## Train Chemprop ADMET predictors

Train Chemprop and Chemprop-RDKit predictors on the ADMET data. Note: A GPU is used by default if available.

Train Chemprop-RDKit ADMET predictors on the TDC ADMET Benchmark Group data.

```bash
python train_tdc_admet_group.py \
    --data_dir data/tdc_admet_group \
    --save_dir models/tdc_admet_group \
    --model_type chemprop_rdkit
```

Train Chemprop-RDKit ADMET predictors on all TDC ADMET datasets.

```bash
python train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir models/tdc_admet_all \
    --model_type chemprop_rdkit
```

Train Chemprop-RDKit ADMET predictors on the TDC ADMET multitask datasets.

```bash
python train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all_multitask \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop_rdkit
```

## Evaluate TDC ADMET Benchmark Group models

Evaluate Chemprop-RDKit ADMET predictors trained on the TDC ADMET Benchmark Group data.

```bash
python evaluate_tdc_admet_group.py \
    --data_dir data/tdc_admet_group_raw \
    --preds_dir models/tdc_admet_group/chemprop_rdkit
```


## Make predictions with Chemprop ADMET predictors

The instructions below illustrate how to make predictions with trained Chemprop-RDKit multitask ADMET predictors. The instructions assume that you have a file called `data.csv` which contains SMILES strings in a column called `smiles`. Note: A GPU is used by default if available.

```bash
admet_predict \
    --data_path data.csv \
    --save_path preds.csv \
    --model_dir models/tdc_admet_all_multitask/chemprop_rdkit \
    --smiles_column smiles
```

## Get approved drugs from DrugBank

Get approved drugs from DrugBank to create a comparison set for Chemprop ADMET predictors.

```bash
python get_drugbank_approved.py \
    --data_path data/drugbank/drugbank.xml \
    --save_path data/drugbank/drugbank_approved.csv
```

## Plot results

Plot TDC ADMET results. First, download the results from [here](https://docs.google.com/spreadsheets/d/1bh9FEHqhbfHKF-Nxjad0Cpy2p5ztH__p0pijB43yc94/edit?usp=sharing) and save them to `results/TDC ADMET Results.xlsx`. Then run the following command.

```bash
python scripts/plot_tdc_results.py \
    --results_path results/TDC\ ADMET\ Results.xlsx \
    --save_dir plots/tdc_results
```

Plot DrugBank statistics.

```bash
python scripts/plot_drugbank_approved.py \
    --data_path data/drugbank/drugbank_approved.csv \
    --save_dir plots/drugbank_approved
```
