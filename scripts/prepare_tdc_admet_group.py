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
    group = admet_group(path=raw_data_dir)

    # Create list of dataset stats
    dataset_stats = []

    # Prepare each dataset
    for data_name in tqdm(data_names):
        # Load dataset
        benchmark = group.get(data_name)

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

        # Get name
        name = benchmark["name"]

        # Make data directory
        benchmark_dir = save_dir / name
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Split data into train_val and test
        train_val, test = benchmark["train_val"], benchmark["test"]

        # Save test data
        test.to_csv(benchmark_dir / "test.csv")

        # Split train_val into 5-fold CV
        for seed in ADMET_GROUP_SEEDS:
            # Split train_val into train and val
            train, valid = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed
            )

            # Make seed directory
            seed_dir = benchmark_dir / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Save train and val data
            train.to_csv(seed_dir / "train.csv")
            valid.to_csv(seed_dir / "val.csv")

    # Print dataset stats
    dataset_stats = pd.DataFrame(dataset_stats).set_index("name")
    pd.set_option("display.max_rows", None)
    print(dataset_stats)


if __name__ == "__main__":
    from tap import tapify

    tapify(prepare_tdc_admet_group)
