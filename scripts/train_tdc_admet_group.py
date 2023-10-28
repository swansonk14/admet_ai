"""Train Chemprop models on the Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets"""
import subprocess
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from tdc_constants import (
    ADMET_GROUP_SEEDS,
    ADMET_GROUP_SMILES_COLUMN,
    ADMET_GROUP_TARGET_COLUMN,
    DATASET_TO_TYPE_LOWER,
    DATASET_TYPE_TO_METRICS_COMMAND_LINE,
)


def train_tdc_admet_all(
    data_dir: Path, save_dir: Path, model_type: Literal["chemprop", "chemprop_rdkit"]
) -> None:
    """Train Chemprop models on the Therapeutics Data Commons (TDC) ADMET Benchmark Group datasets.

    :param data_dir: A directory containing the downloaded and prepared TDC ADMET Benchmark Group data.
    :param save_dir: A directory where the models will be saved.
    :param model_type: The type of model to train (Chemprop or Chemprop-RDKit).
    """
    # Get dataset paths
    data_dirs = sorted(data_dir.iterdir())

    # Train Chemprop or Chemprop-RDKit model on each dataset
    for data_dir in tqdm(data_dirs):
        data_name = data_dir.name
        dataset_type = DATASET_TO_TYPE_LOWER[data_name]

        for seed in ADMET_GROUP_SEEDS:
            command = [
                "chemprop_train",
                "--data_path",
                str(data_dir / str(seed) / "train.csv"),
                "--separate_val_path",
                str(data_dir / str(seed) / "val.csv"),
                "--separate_test_path",
                str(data_dir / "test.csv"),
                "--dataset_type",
                dataset_type,
                "--smiles_column",
                ADMET_GROUP_SMILES_COLUMN,
                "--target_columns",
                ADMET_GROUP_TARGET_COLUMN,
                *DATASET_TYPE_TO_METRICS_COMMAND_LINE[dataset_type],
                "--save_dir",
                save_dir / model_type / data_name / str(seed),
                "--save_preds",
                "--quiet",
            ]

            if model_type == "chemprop_rdkit":
                command += [
                    "--features_path",
                    str(data_dir / str(seed) / "train.npz"),
                    "--separate_val_features_path",
                    str(data_dir / str(seed) / "val.npz"),
                    "--separate_test_features_path",
                    str(data_dir / "test.npz"),
                    "--no_features_scaling",
                ]

            subprocess.run(command)


if __name__ == "__main__":
    from tap import tapify

    tapify(train_tdc_admet_all)
