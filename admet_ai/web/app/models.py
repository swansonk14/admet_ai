"""Defines functions for ADMET-AI models."""
from typing import Any

import numpy as np
from chemfunc.molecular_fingerprints import compute_fingerprints
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.train import predict as chemprop_predict
from chemprop.utils import load_args, load_checkpoint, load_scalers, load_task_names
from tqdm import tqdm

from admet_ai.web.app import app


MODELS: list[dict[str, Any]] = []


def load_models() -> None:
    """Loads the models into memory."""
    print("--- LOADING MODELS ---")

    # Loop through the model directories and add each model ensemble to MODELS
    for model_dir in app.config["MODEL_DIR"].iterdir():
        # Get model paths for models in ensemble
        model_paths = list(model_dir.glob("**/*.pt"))

        # Load train args
        train_args = load_args(model_paths[0])

        # Add task names, models, and scalers to MODELS
        MODELS.append(
            {
                "task_names": load_task_names(model_paths[0]),
                "models": [
                    load_checkpoint(path=str(model_path)).eval()
                    for model_path in model_paths
                ],
                "scalers": [
                    load_scalers(path=str(model_path))[0] for model_path in model_paths
                ],
                "uses_fingerprints": train_args.features_path is not None
                or train_args.features_generator is not None,
            }
        )


def get_models() -> list[dict[str, Any]]:
    """Gets a list of models and their associated information (models, scalers, task names, etc.).

    :return: A list of models and their associated information (models, scalers, task names, etc.).
    """
    return MODELS


def predict_all_models(
    smiles: list[str], num_workers: int = 0
) -> tuple[list[str], list[list[float]]]:
    """Make prediction with all the loaded models.

    TODO: Support GPU prediction.
    TODO: Handle invalid SMILES.

    :param smiles: A list of SMILES.
    :param num_workers: The number of workers for parallel data loading.
    :return: A tuple containing a list of task names and a list of predictions (num_molecules, num_tasks).
    """
    # Get models dict with models and association information
    models = get_models()

    # Determine fingerprints use
    uses_fingerprints_set = {model_dict["uses_fingerprints"] for model_dict in models}
    any_fingerprints_use = any(uses_fingerprints_set)
    all_fingerprints_use = all(uses_fingerprints_set)

    # Build data loader without fingerprints
    if not all_fingerprints_use:
        data_loader_without_fingerprints = MoleculeDataLoader(
            dataset=MoleculeDataset(
                [MoleculeDatapoint(smiles=[smile],) for smile in smiles]
            ),
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        data_loader_without_fingerprints = None

    # Build dataloader with fingerprints
    if any_fingerprints_use:
        # TODO: Remove assumption of RDKit fingerprints
        fingerprints = compute_fingerprints(smiles, fingerprint_type="rdkit")

        data_loader_with_fingerprints = MoleculeDataLoader(
            dataset=MoleculeDataset(
                [
                    MoleculeDatapoint(smiles=[smile], features=fingerprint)
                    for smile, fingerprint in zip(smiles, fingerprints)
                ]
            ),
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        data_loader_with_fingerprints = None

    # Initialize lists to contain task names and predictions
    all_task_names = []
    all_preds = []

    # Loop through each ensemble and make predictions
    for model_dict in tqdm(models, desc="model ensembles"):
        # Get task names
        all_task_names += model_dict["task_names"]

        # Select data loader based on features use
        if model_dict["uses_fingerprints"]:
            data_loader = data_loader_with_fingerprints
        else:
            data_loader = data_loader_without_fingerprints

        # Make predictions
        preds = [
            chemprop_predict(model=model, data_loader=data_loader)
            for model in tqdm(model_dict["models"], desc="individual models")
        ]

        # Scale predictions if needed (for regression)
        if model_dict["scalers"][0] is not None:
            preds = [
                scaler.inverse_transform(pred).astype(float)
                for scaler, pred in zip(model_dict["scalers"], preds)
            ]

        # Average ensemble predictions
        preds = np.mean(preds, axis=0).transpose()  # (num_tasks, num_molecules)
        all_preds += preds.tolist()

    # Transpose preds
    all_preds: list[list[float]] = np.array(
        all_preds
    ).transpose().tolist()  # (num_molecules, num_tasks)

    return all_task_names, all_preds
