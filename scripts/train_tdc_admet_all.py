"""Train Chemprop models on all the Therapeutics Data Commons (TDC) ADMET datasets"""

import subprocess
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from tdc_constants import (
    DATASET_TO_TYPE,
    DATASET_TYPE_TO_METRICS_COMMAND_LINE,
    ADMET_ALL_SMILES_COLUMN,
)


def train_tdc_admet_all(
    data_dir: Path,
    save_dir: Path,
    model_type: Literal["chemprop", "chemprop_rdkit"],
    num_replicates: int = 5,
) -> None:
    """Train Chemprop models on all the Therapeutics Data Commons (TDC) ADMET datasets.

    :param data_dir: A directory containing all the downloaded and prepared TDC ADMET data.
    :param save_dir: A directory where the models will be saved.
    :param model_type: The type of model to train (Chemprop or Chemprop-RDKit).
    :param num_folds: The number of folds to use for cross-validation.
    """
    # Get dataset paths
    data_paths = sorted(data_dir.glob("*.csv"))

    # Train Chemprop or Chemprop-RDKit model on each dataset
    for data_path in tqdm(data_paths):
        data_name = data_path.stem
        dataset_type = DATASET_TO_TYPE[data_name]

        command = [
            "chemprop",
            "train",
            "-i",
            str(data_path),
            "--task-type",
            dataset_type,
            "--smiles-column",
            ADMET_ALL_SMILES_COLUMN,
            *DATASET_TYPE_TO_METRICS_COMMAND_LINE[dataset_type],
            "--num-replicates",
            str(num_replicates),
            "--save-dir",
            save_dir / model_type / data_name,
            # "--save_preds",
            # "--quiet",
        ]

        if model_type == "chemprop_rdkit":
            command += [
                "--descriptors-path",
                str(data_path.with_suffix(".npz")),
                "--no-descriptor-scaling",
            ]

        subprocess.run(command)


if __name__ == "__main__":
    from tap import tapify

    tapify(train_tdc_admet_all)
