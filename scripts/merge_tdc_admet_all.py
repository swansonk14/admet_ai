"""Merge all the Therapeutics Data Commons (TDC) ADMET datasets into a single dataset."""
from functools import reduce
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tdc_constants import DATASET_TO_TYPE, ADMET_ALL_SMILES_COLUMN


def merge_tdc_admet_all(data_dir: Path, save_path: Path) -> None:
    """Merge all the Therapeutics Data Commons (TDC) ADMET datasets into a single dataset.

    :param data_dir: A directory with all the TDC ADMET datasets as CSV files.
    :param save_path: Path to a CSV file where the merged dataset will be saved.
    """
    # Get dataset paths
    data_paths = sorted(data_dir.glob("*.csv"))

    # Load all dataset
    data = [pd.read_csv(data_path) for data_path in tqdm(data_paths)]

    # Merge datasets
    data = reduce(
        lambda left, right: pd.merge(
            left, right, how="outer", on=ADMET_ALL_SMILES_COLUMN
        ),
        data,
    )

    # Print stats
    print(f"Dataset size = {len(data):,}")
    print(f"Number of tasks: {len(data.columns) - 1}")

    # Save dataset
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(merge_tdc_admet_all)
