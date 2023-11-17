# Reproducing TDC ADMET Results

This file contains instructions for reproducing the ADMET-AI data, models, and results. Note: Running the below commands requires the additional TDC dependencies (i.e., `pip install admet_ai[tdc]`).

- [Download TDC ADMET data](#download-tdc-admet-data)
- [Create multitask datasets for regression and classification](#create-multitask-datasets-for-regression-and-classification)
- [Create a single dataset with all TDC ADMET data](#create-a-single-dataset-with-all-tdc-admet-data)
- [Compute RDKit features](#compute-rdkit-features)
- [Train Chemprop-RDKit ADMET predictors](#train-chemprop-rdkit-admet-predictors)
- [Evaluate TDC ADMET Benchmark Group models](#evaluate-tdc-admet-benchmark-group-models)
- [Get approved drugs from DrugBank](#get-approved-drugs-from-drugbank)
- [Subsample approved drugs from DrugBank](#subsample-approved-drugs-from-drugbank)
- [Make predictions on DrugBank approved drugs](#make-predictions-on-drugbank-approved-drugs)
- [Plot results](#plot-results)

## Download TDC ADMET data

Download the [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) data (v0.4.1) for evaluating models using scaffold splits in order to compare to the TDC leaderboard.

```bash
python scripts/prepare_tdc_admet_group.py \
    --raw_data_dir data/tdc_admet_group_raw \
    --save_dir data/tdc_admet_group
```

Download all TDC [ADME](https://tdcommons.ai/single_pred_tasks/adme/) and [Tox](https://tdcommons.ai/single_pred_tasks/tox/) datasets for training models. Skip datasets that are redundant or not needed.

```bash
python scripts/prepare_tdc_admet_all.py \
    --save_dir data/tdc_admet_all \
    --skip_datasets herg_central hERG_Karim ToxCast
```

## Create multitask datasets for regression and classification

Create multitask datasets for regression and classification for all the TDC ADMET datasets.

```bash
python scripts/merge_tdc_admet_multitask.py \
    --data_dir data/tdc_admet_all \
    --save_dir data/tdc_admet_all_multitask
```

## Create a single dataset with all TDC ADMET data

Create a single dataset with all TDC ADMET data, primarily for the purpose of searching across the TDC data.

```bash
python scripts/merge_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_path data/tdc_admet_all.csv
```

## Compute RDKit features

Compute RDKit features in order to train Chemprop-RDKit models (i.e., Chemprop models augmented with 200 molecular features from RDKit).

Compute RDKit features for the TDC ADMET Benchmark Group data.

```bash
python scripts/compute_rdkit_features.py \
    --data_dir data/tdc_admet_group \
    --smiles_column Drug
```

Compute RDKit features for all TDC ADMET datasets.

```bash
python scripts/compute_rdkit_features.py \
    --data_dir data/tdc_admet_all \
    --smiles_column smiles
```

Compute RDKit features for TDC ADMET multitask datasets.

```bash
python scripts/compute_rdkit_features.py \
    --data_dir data/tdc_admet_all_multitask \
    --smiles_column smiles
```

## Train Chemprop-RDKit ADMET predictors

Train Chemprop-RDKit predictors on the ADMET data. Note: A GPU is used by default if available.

Train Chemprop-RDKit ADMET predictors on the TDC ADMET Benchmark Group data.

```bash
python scripts/train_tdc_admet_group.py \
    --data_dir data/tdc_admet_group \
    --save_dir models/tdc_admet_group \
    --model_type chemprop_rdkit
```

Train Chemprop-RDKit ADMET predictors on all TDC ADMET datasets.

```bash
python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all \
    --save_dir models/tdc_admet_all \
    --model_type chemprop_rdkit
```

Train Chemprop-RDKit ADMET predictors on the TDC ADMET multitask datasets.

```bash
python scripts/train_tdc_admet_all.py \
    --data_dir data/tdc_admet_all_multitask \
    --save_dir models/tdc_admet_all_multitask \
    --model_type chemprop_rdkit
```

## Evaluate TDC ADMET Benchmark Group models

Evaluate Chemprop-RDKit ADMET predictors trained on the TDC ADMET Benchmark Group data.

```bash
python scripts/evaluate_tdc_admet_group.py \
    --data_dir data/tdc_admet_group_raw \
    --preds_dir models/tdc_admet_group/chemprop_rdkit
```

## Get approved drugs from DrugBank

Get approved drugs from [DrugBank](https://go.drugbank.com/) (v5.1.10) to create a comparison set for Chemprop ADMET predictors. This assumes that the file `drugbank.xml` has been downloaded from DrugBank (license required).

```bash
python scripts/get_drugbank_approved.py \
    --data_path data/drugbank/drugbank.xml \
    --save_path data/drugbank/drugbank_approved.csv
```

## Make predictions on DrugBank approved drugs

Compute physicochemical properties on DrugBank approved drugs using RDKit.

```bash
physchem_compute \
    --data_path data/drugbank/drugbank_approved.csv \
    --save_path data/drugbank/drugbank_approved_physchem.csv \
    --smiles_column smiles
```

Make ADMET predictions on DrugBank approved drugs using Chemprop-RDKit multitask predictor.

```bash
admet_predict \
    --data_path data/drugbank/drugbank_approved_physchem.csv \
    --save_path data/drugbank/drugbank_approved_physchem_admet.csv \
    --model_dir models/tdc_admet_all_multitask/chemprop_rdkit \
    --smiles_column smiles
```

## Subsample approved drugs from DrugBank

Subsample approved drugs from DrugBank for measuring ADMET prediction speed. Limit SMILES length to 200 for compatibility with SwissADME.

```bash
for NUM_MOLECULES in 1 10 100
do
python scripts/sample_molecules.py \
    --data_path data/drugbank/drugbank_approved.csv \
    --num_molecules ${NUM_MOLECULES} \
    --max_smiles_length 200 \
    --save_path data/drugbank/drugbank_approved_${NUM_MOLECULES}.csv
done
```

Due to compatibility issues with ADMETlab2.0, four compounds in `drugbank_approved_100.csv` were replaced with other randomly sampled compounds from DrugBank. The replacements are as follows.

Helium ==> Cabazitaxel

Chromic nitrate ==> Butorphanol

Perboric acid ==> Methazolamide

Fluoride ion F-18 ==> Tetracaine

Then, `drugbank_approved_1000.csv` was constructed by repeating `drugbank_approved_100.csv` ten times (in order to maintain compatibility with ADMETlab2.0). Similarly, `drugbank_approved_1M.csv` was constructed by repeating `drugbank_approved_1000.csv` 1000 times.


## Time local ADMET-AI predictions

Time local ADMET-AI predictions on DrugBank approved drugs. Run this command on an 8-core machine with and without a GPU.

```bash
for NUM_MOLECULES in 1 10 100 1000 1M
do
for ITER in 1 2 3
do
echo "Timing ADMET-AI on ${NUM_MOLECULES} molecules, iteration ${ITER}"
time admet_predict \
    --data_path data/drugbank/drugbank_approved_${NUM_MOLECULES}.csv \
    --save_path data/drugbank/drugbank_approved_${NUM_MOLECULES}_admet_${ITER}.csv \
    --model_dir models/tdc_admet_all_multitask/chemprop_rdkit \
    --smiles_column smiles
done
done
```

## Plot results

Plot TDC ADMET results. First, download the results from [here](https://docs.google.com/spreadsheets/d/1bh9FEHqhbfHKF-Nxjad0Cpy2p5ztH__p0pijB43yc94/edit?usp=sharing) and save them to `results/TDC ADMET Results.xlsx`. Then, run the following command.

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

Plot ADMET website speed. First, download the results from [here]() and save them to `results/ADMET Tools Comparison.xlsx`. Then, run the following command.

```bash
python scripts/plot_admet_speed.py \
    --results_path results/ADMET\ Tools\ Comparison.xlsx \
    --save_path plots/admet_speed.pdf
```
