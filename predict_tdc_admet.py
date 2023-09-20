"""Make predictions on a dataset using Chemprop models trained on TDC ADMET data (all or Benchmark Data group)."""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from chemfunc.molecular_fingerprints import compute_fingerprints
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDatapoint,
    MoleculeDataset,
    set_cache_graph,
    set_cache_mol,
)
from chemprop.train import predict
from chemprop.utils import load_args, load_checkpoint, load_scalers
from tqdm import tqdm


def predict_tdc_admet(
    data_path: Path,
    model_dir: Path,
    model_type: Literal["chemprop", "chemprop_rdkit"],
    save_path: Path | None = None,
    smiles_column: str = "smiles",
    num_workers: int = 8,
    no_cache: bool = False,
) -> None:
    """Make predictions on a dataset using Chemprop models trained on TDC ADMET data (all or Benchmark Data group).

    Note: If using a Chemprop-RDKit model, this script first computes RDKit features.

    :param data_path: Path to a CSV file containing a dataset of molecules.
    :param model_dir: Path to a directory containing Chemprop or Chemprop-RDKit models.
    :param model_type: The type of model to use (Chemprop or Chemprop-RDKit).
    :param save_path: Path to a CSV file where predictions will be saved. If None, defaults to data_path.
    :param smiles_column: Name of the column containing SMILES strings.
    :param num_workers: Number of workers for the data loader. Zero workers (i.e., sequential data loading)
                        may be faster if not using a GPU.
    :param no_cache: Whether to disable caching. This is suggested when making predictions on large datasets.
    """
    # Disable Chemprop caching for prediction to avoid memory issues with large datasets
    if no_cache:
        set_cache_graph(False)
        set_cache_mol(False)

    # Load SMILES
    data = pd.read_csv(data_path)
    smiles = list(data[smiles_column])

    # Compute fingerprints
    fingerprints = (
        compute_fingerprints(smiles, fingerprint_type="rdkit")
        if model_type == "chemprop_rdkit"
        else None
    )

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set up fingerprints
    if fingerprints is None:
        fingerprints = [None] * len(smiles)

    # Build data loader
    data_loader = MoleculeDataLoader(
        dataset=MoleculeDataset(
            [
                MoleculeDatapoint(
                    smiles=[smile],
                    features=fingerprint,
                )
                for smile, fingerprint in zip(smiles, fingerprints)
            ]
        ),
        num_workers=num_workers,
        shuffle=False,
    )

    # Get model directory for each ensemble
    ensemble_dirs = sorted(model_dir.iterdir())

    # Initialize dictionary to contain predictions
    task_to_preds = {}

    # Loop through each ensemble and make predictions
    for ensemble_dir in tqdm(ensemble_dirs, desc="model ensembles"):
        # Get model paths in ensemble
        model_paths = sorted(ensemble_dir.glob("**/*.pt"))

        # Get task names
        train_args = load_args(model_paths[0])
        task_names = train_args.task_names

        # Load models
        models = [
            load_checkpoint(path=str(model_path), device=device).eval()
            for model_path in model_paths
        ]

        # Load scalers
        scalers = [load_scalers(path=str(model_path))[0] for model_path in model_paths]

        # Make predictions
        preds = [
            predict(model=model, data_loader=data_loader)
            for model in tqdm(models, desc="individual models")
        ]

        # Scale predictions if needed (for regression)
        if scalers[0] is not None:
            preds = [
                scaler.inverse_transform(pred).astype(float)
                for scaler, pred in zip(scalers, preds)
            ]

        # Average ensemble predictions
        preds = np.mean(preds, axis=0)

        # Add predictions to data
        for i, task_name in enumerate(task_names):
            task_to_preds[task_name] = preds[:, i]

    # Put preds in a DataFrame
    preds = pd.DataFrame(task_to_preds)

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
