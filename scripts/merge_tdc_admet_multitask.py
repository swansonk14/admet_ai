"""Merge all the Therapeutics Data Commons (TDC) ADMET datasets into multitask datasets for regression and classification."""
from functools import reduce
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tdc_constants import DATASET_TO_TYPE, ADMET_ALL_SMILES_COLUMN


def merge_tdc_admet_all(data_dir: Path, save_dir: Path) -> None:
    """Merge all the Therapeutics Data Commons (TDC) ADMET datasets into multitask datasets for regression and classification.

    :param data_dir: A directory with all the TDC ADMET datasets as CSV files.
    :param save_dir: A directory where the two merged datasets will be saved.
    """
    # Get dataset paths
    data_paths = sorted(data_dir.glob("*.csv"))

    # Set up lists for regression and classification data
    regression_data = []
    classification_data = []

    # Load all datasets
    for data_path in tqdm(data_paths):
        # Get data name
        data_name = data_path.stem

        # Load dataset
        dataset = pd.read_csv(data_path)

        # Add dataset to appropriate list
        if DATASET_TO_TYPE[data_name] == "regression":
            regression_data.append(dataset)
        else:
            classification_data.append(dataset)

    # Merge datasets
    regression_data = reduce(
        lambda left, right: pd.merge(
            left, right, how="outer", on=ADMET_ALL_SMILES_COLUMN
        ),
        regression_data,
    )
    classification_data = reduce(
        lambda left, right: pd.merge(
            left, right, how="outer", on=ADMET_ALL_SMILES_COLUMN
        ),
        classification_data,
    )

    # Print stats
    print(f"Regression dataset size = {len(regression_data):,}")
    print(f"Number of regression tasks: {len(regression_data.columns) - 1}")
    print()
    print(f"Classification dataset size = {len(classification_data):,}")
    print(f"Number of classification tasks: {len(classification_data.columns) - 1}")

    # Save datasets
    save_dir.mkdir(parents=True, exist_ok=True)
    regression_data.to_csv(save_dir / "admet_regression.csv", index=False)
    classification_data.to_csv(save_dir / "admet_classification.csv", index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(merge_tdc_admet_all)
