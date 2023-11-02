"""ADMET-AI class to contain ADMET model and prediction function."""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemfunc.molecular_fingerprints import compute_rdkit_fingerprint
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDatapoint,
    MoleculeDataset,
    set_cache_graph,
    set_cache_mol,
)
from chemprop.data.data import SMILES_TO_MOL
from chemprop.models import MoleculeModel
from chemprop.train import predict
from chemprop.utils import load_args, load_checkpoint, load_scalers
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class ADMETModel:
    """ADMET-AI model class."""

    # TODO: set defaults for model paths in constants and include model files in git repo
    def __init__(
        self,
        model_dirs: list[Path | str],
        num_workers: int = 8,
        cache_molecules: bool = True,
        fingerprint_multiprocessing_min: int = 100,
    ) -> None:
        """Initialize the ADMET-AI model.

        :param model_dirs: List of paths to directories, where each directory contains
                           an ensemble of Chemprop-RDKit models.
        :param num_workers: Number of workers for the data loader.
        :param cache_molecules: Whether to cache molecules. Caching improves prediction speed but requires more memory.
        :param fingerprint_multiprocessing_min: Minimum number of molecules for multiprocessing to be used for
                                                fingerprint computation. Otherwise, single processing is used.
        """
        # Save parameters
        self.num_workers = num_workers
        self.cache_molecules = cache_molecules
        self.fingerprint_multiprocessing_min = fingerprint_multiprocessing_min

        # Set caching
        set_cache_graph(self.cache_molecules)
        set_cache_mol(self.cache_molecules)

        # Set device based on GPU availability
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load each ensemble of models
        self.task_lists: list[list[str]] = []
        self.use_features_list: list[bool] = []
        self.model_lists: list[list[MoleculeModel]] = []
        self.scaler_lists: list[list[StandardScaler | None]] = []

        for model_dir in model_dirs:
            # Get model paths for the ensemble in the directory
            model_paths = sorted(Path(model_dir).glob("**/*.pt"))

            # Load args for this ensemble
            train_args = load_args(str(model_paths[0]))

            # Get task names for this ensemble
            task_names = train_args.task_names
            self.task_lists.append(task_names)

            # Get whether to use features for this ensemble
            use_features = train_args.use_input_features
            self.use_features_list.append(use_features)

            # Load models in the ensemble
            models = [
                load_checkpoint(path=str(model_path), device=self.device).eval()
                for model_path in model_paths
            ]
            self.model_lists.append(models)

            # Load scalers for each model
            scalers = [
                load_scalers(path=str(model_path))[0] for model_path in model_paths
            ]
            self.scaler_lists.append(scalers)

        # Ensure all models do or do not use features
        if not len(set(self.use_features_list)) == 1:
            raise ValueError("All models must either use or not use features.")

        self.use_features = self.use_features_list[0]

    @property
    def num_ensembles(self) -> int:
        """Get the number of ensembles."""
        return len(self.model_lists)

    def predict(self, smiles: str | list[str]) -> pd.DataFrame:
        """Make predictions on a list of SMILES strings.

        :param smiles: A SMILES string or a list of SMILES strings.
        :return: A DataFrame containing the predictions with SMILES strings as the index.
        """
        # Convert SMILES to list if needed
        if isinstance(smiles, str):
            smiles = [smiles]

        # Convert SMILES to RDKit molecules and cache if desired
        mols = []
        for smile in tqdm(smiles, desc="SMILES to Mol"):
            if smile in SMILES_TO_MOL:
                mol = SMILES_TO_MOL[smile]
            else:
                mol = Chem.MolFromSmiles(smile)

            mols.append(mol)

            if self.cache_molecules:
                SMILES_TO_MOL[smile] = mol

        # Remove invalid molecules
        invalid_mols = [mol is None for mol in mols]

        if any(invalid_mols):
            print(f"Warning: {sum(invalid_mols):,} invalid molecules will be removed")

            mols = [mol for mol in mols if mol is not None]
            smiles = [
                smile for smile, invalid in zip(smiles, invalid_mols) if not invalid
            ]

        # Compute fingerprints if needed
        if self.use_features:
            # Select between multiprocessing and single processing
            if len(mols) >= self.fingerprint_multiprocessing_min:
                pool = Pool()
                map_fn = pool.imap
            else:
                pool = None
                map_fn = map

            # Compute fingerprints
            fingerprints = np.array(
                list(
                    tqdm(
                        map_fn(compute_rdkit_fingerprint, mols),
                        total=len(mols),
                        desc=f"RDKit fingerprints",
                    )
                )
            )

            # Close pool if needed
            if pool is not None:
                pool.close()
        else:
            fingerprints = [None] * len(smiles)

        # Build data loader
        data_loader = MoleculeDataLoader(
            dataset=MoleculeDataset(
                [
                    MoleculeDatapoint(smiles=[smile], features=fingerprint,)
                    for smile, fingerprint in zip(smiles, fingerprints)
                ]
            ),
            num_workers=self.num_workers,
            shuffle=False,
        )

        # Make predictions
        task_to_preds = {}

        # Loop through each ensemble and make predictions
        for tasks, use_features, models, scalers in tqdm(
            zip(
                self.task_lists,
                self.use_features_list,
                self.model_lists,
                self.scaler_lists,
            ),
            total=self.num_ensembles,
            desc="model ensembles",
        ):
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
