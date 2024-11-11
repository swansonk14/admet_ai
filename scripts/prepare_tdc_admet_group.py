"""Download and prepare the Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets."""

from pathlib import Path

import pandas as pd
from tdc import utils
from tdc.benchmark_group import admet_group
from tqdm import tqdm

from tdc_constants import (
    ADMET_GROUP_SEEDS,
    ADMET_GROUP_TARGET_COLUMN,
    DATASET_TO_TYPE_LOWER,
)


def prepare_tdc_admet_group(raw_data_dir: Path, save_dir: Path) -> None:
    """Download and prepare the Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets.

    :param raw_data_dir: A directory where the raw TDC ADMET Benchmark Group data will be saved.
    :param save_dir: A directory where the formatted TDC AMDET Benchmark Group data will be saved.
    """
    # Get ADMET Benchmark Group dataset names from TDC
    data_names = utils.retrieve_benchmark_names("ADMET_Group")

    # Download/access the ADMET group
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=raw_data_dir)

    # Create list of dataset stats
    dataset_stats = []
    all_data = []

    # Prepare each dataset
    for data_name in tqdm(data_names):
        # Load dataset
        benchmark = group.get(data_name)

        # Add split labels to each row in train_val and test sets
        train_val_data = benchmark["train_val"].copy()
        test_data = benchmark["test"].copy()
        train_val_data["split"] = "train"  # Start with train, adjust later for val
        test_data["split"] = "test"

        # Split train_val into 5-fold CV
        for seed in ADMET_GROUP_SEEDS:

            dataset_dir = save_dir / data_name / str(seed)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            # Get train and val split
            train, valid = group.get_train_valid_split(
                benchmark=benchmark["name"], split_type="default", seed=seed
            )

            # Assign 'train' and 'val' to corresponding rows
            train_val_data.loc[train.index, "split"] = "train"
            train_val_data.loc[valid.index, "split"] = "val"

            # Combine train_val and test into a single dataset
            combined_data = pd.concat([train_val_data, test_data]).reset_index(
                drop=True
            )

            # Add dataset name and seed for reference
            combined_data["dataset"] = data_name
            combined_data["seed"] = seed
            combined_data.to_csv(dataset_dir / "data.csv", index=False)

            # Append combined dataset to all_data
            all_data.append(combined_data)

        # Create a single dataset for computing statistics
        data = pd.concat((benchmark["train_val"], benchmark["test"]))

        # Compute class balance
        if DATASET_TO_TYPE_LOWER[data_name] == "classification":
            class_balance = data[ADMET_GROUP_TARGET_COLUMN].value_counts(
                normalize=True
            )[1]
        else:
            class_balance = None

        # Collect dataset stats
        dataset_stats.append(
            {
                "name": data_name,
                "size": len(data),
                "min": data[ADMET_GROUP_TARGET_COLUMN].min(),
                "max": data[ADMET_GROUP_TARGET_COLUMN].max(),
                "class_balance": class_balance,
            }
        )

    print(dataset_stats)


if __name__ == "__main__":
    from tap import tapify

    tapify(prepare_tdc_admet_group)
