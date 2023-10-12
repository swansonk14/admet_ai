"""Make predictions on a dataset using Chemprop models trained on TDC ADMET data (all or Benchmark Data group)."""
from pathlib import Path
from typing import Literal

import pandas as pd

from admet_ai import ADMETModel


def predict_tdc_admet(
    data_path: Path,
    model_dir: Path,
    save_path: Path | None = None,
    smiles_column: str = "smiles",
    num_workers: int = 8,
    cache_molecules: bool = True,
) -> None:
    """Make predictions on a dataset using Chemprop models trained on TDC ADMET data (all or Benchmark Data group).

    Note: If using a Chemprop-RDKit model, this script first computes RDKit features.

    :param data_path: Path to a CSV file containing a dataset of molecules.
    :param model_dir: Path to a directory containing Chemprop or Chemprop-RDKit models.
    :param save_path: Path to a CSV file where predictions will be saved. If None, defaults to data_path.
    :param smiles_column: Name of the column containing SMILES strings.
    :param num_workers: Number of workers for the data loader. Zero workers (i.e., sequential data loading)
                        may be faster if not using a GPU.
    :param cache_molecules: Whether to cache molecules. Caching improves prediction speed but requires more memory.
    """
    # Load data
    data = pd.read_csv(data_path)
    smiles = list(data[smiles_column])

    # Build ADMETModel
    model = ADMETModel(
        model_dirs=sorted(model_dir.iterdir()),
        num_workers=num_workers,
        cache_molecules=cache_molecules
    )

    # Make predictions
    preds = model.predict(smiles)

    # Merge data and preds
    data_with_preds = pd.concat((data, preds), axis=1)

    # Save predictions
    if save_path is None:
        save_path = data_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    data_with_preds.to_csv(save_path, index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(predict_tdc_admet)
