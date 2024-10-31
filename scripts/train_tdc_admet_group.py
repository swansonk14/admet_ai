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
                "chemprop",
                "train",
                "-i",
                str(data_dir / str(seed) / "data.csv"),
                "--task-type",
                dataset_type,
                "--splits-column",
                "split",
                "--smiles-columns",
                ADMET_GROUP_SMILES_COLUMN,
                "--target-columns",
                ADMET_GROUP_TARGET_COLUMN,
                *DATASET_TYPE_TO_METRICS_COMMAND_LINE[dataset_type],
                "--save-dir",
                save_dir / model_type / data_name / str(seed),
                "--checkpoint",
                save_dir / model_type / data_name / str(seed),
                # "--save_preds",
                # "--quiet",
            ]

            if model_type == "chemprop_rdkit":
                command += [
                    "--descriptors-path",
                    str(data_dir / str(seed) / "data.npz"),
                    "--no-descriptor-scaling",
                ]

            subprocess.run(command)


if __name__ == "__main__":
    from tap import tapify

    tapify(train_tdc_admet_all)
