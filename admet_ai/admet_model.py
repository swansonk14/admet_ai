"""ADMET-AI class to contain ADMET model and prediction function."""
from pathlib import Path

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
from chemprop.models import MoleculeModel
from chemprop.train import predict
from chemprop.utils import load_args, load_checkpoint, load_scalers
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# TODO: use ADMETModel in scripts/predict_tdc_admet.py
class ADMETModel:
    """ADMET-AI model class."""

    # TODO: set defaults for model paths in constants and include model files in git repo
    def __init__(
            self,
            model_dirs: list[Path],
            num_workers: int = 8,
            cache_molecules: bool = True
    ) -> None:
        """Initialize the ADMET-AI model.

        :param model_dirs: List of paths to directories, where each directory contains
                           an ensemble of Chemprop-RDKit models.
        :param num_workers: Number of workers for the data loader.
        :param cache_molecules: Whether to cache molecules. Caching improves prediction speed but requires more memory.
        """
        # Save parameters
        self.num_workers = num_workers

        # Set caching
        set_cache_graph(cache_molecules)
        set_cache_mol(cache_molecules)

        # Set device based on GPU availability
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load each ensemble of models
        self.task_lists: list[list[str]] = []
        self.model_lists: list[list[MoleculeModel]] = []
        self.scaler_lists: list[list[StandardScaler | None]] = []

        for model_dir in model_dirs:
            # Get model paths for the ensemble in the directory
            model_paths = sorted(model_dir.glob("**/*.pt"))

            # Get task names for this ensemble
            train_args = load_args(model_paths[0])
            task_names = train_args.task_names
            self.task_lists.append(task_names)

            # Load models in the ensemble
            models = [
                load_checkpoint(path=str(model_path), device=self.device).eval()
                for model_path in model_paths
            ]
            self.model_lists.append(models)

            # Load scalers for each model
            scalers = [load_scalers(path=str(model_path))[0] for model_path in model_paths]
            self.scaler_lists.append(scalers)

    @property
    def num_ensembles(self) -> int:
        """Get the number of ensembles."""
        return len(self.model_lists)

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        """Make predictions on a list of SMILES strings.

        :param smiles: List of SMILES strings.
        :return: A DataFrame containing the predictions with SMILES strings as the index.
        """
        # Compute fingerprints
        fingerprints = compute_fingerprints(smiles, fingerprint_type="rdkit")

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
            num_workers=self.num_workers,
            shuffle=False,
        )

        # Make predictions
        task_to_preds = {}

        # Loop through each ensemble and make predictions
        for tasks, models, scalers in tqdm(zip(self.task_lists, self.model_lists, self.scaler_lists),
                                           total=self.num_ensembles, desc="model ensembles"):
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
            for i, task in enumerate(tasks):
                task_to_preds[task] = preds[:, i]

        # Put preds in a DataFrame
        preds = pd.DataFrame(task_to_preds, index=smiles)

        return preds