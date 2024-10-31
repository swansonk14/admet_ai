"""ADMET-AI class to contain ADMET model and prediction function."""

from multiprocessing import Pool
from pathlib import Path

from admet_ai.drugbank import (
    create_atc_code_mapping,
    filter_drugbank_by_atc,
    read_drugbank_data,
)
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
)
from chemprop.models import load_model
from chemprop.data.dataloader import build_dataloader
from chemprop.models import MPNN
from chemprop.models.utils import load_output_columns

from rdkit import Chem
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from admet_ai.constants import (
    DEFAULT_DRUGBANK_PATH,
    DEFAULT_MODELS_DIR,
    DRUGBANK_ATC_NAME_PREFIX,
    DRUGBANK_DELIMITER,
)
from admet_ai.physchem import compute_fingerprints, compute_physicochemical_properties
from admet_ai.utils import get_drugbank_suffix


class ADMETModel:
    """ADMET-AI model class."""

    def __init__(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        include_physchem: bool = True,
        drugbank_path: Path | str | None = DEFAULT_DRUGBANK_PATH,
        atc_code: str | None = None,
        num_workers: int | None = None,
        cache_molecules: bool = True,
        fingerprint_multiprocessing_min: int = 100,
    ) -> None:
        """Initialize the ADMET-AI model.

        :param models_dir: Path to a directory containing subdirectories, each of which contains an ensemble
                           of Chemprop-RDKit models.
        :param include_physchem: Whether to include physicochemical properties in the predictions.
        :param drugbank_path: Path to a CSV file containing DrugBank approved molecules
                              with ADMET predictions and ATC codes.
        :param atc_code: The ATC code to filter the DrugBank reference set by.
                         If None, the entire DrugBank reference set will be used.
        :param num_workers: Number of workers for the data loader. Zero workers (i.e., sequential data loading)
                            may be faster if not using a GPU, while multiple workers (e.g., 8) are faster with a GPU.
                            If None, defaults to 0 if no GPU is available and 8 if a GPU is available.
        :param cache_molecules: Whether to cache molecules. Caching improves prediction speed but requires more memory.
        :param fingerprint_multiprocessing_min: Minimum number of molecules for multiprocessing to be used for
                                                fingerprint computation. Otherwise, single processing is used.
        """
        # Check parameters
        if atc_code is not None and drugbank_path is None:
            raise ValueError(
                "DrugBank reference set must be provided to filter by ATC code."
            )

        # Set default num_workers
        if num_workers is None:
            num_workers = 8 if torch.cuda.is_available() else 0

        # Save parameters
        self.include_physchem = include_physchem
        self.num_workers = num_workers
        self.cache_molecules = cache_molecules
        self.fingerprint_multiprocessing_min = fingerprint_multiprocessing_min
        self._atc_code = atc_code

        # Load DrugBank reference set if needed
        self.drugbank, self.drugbank_atc_filtered = self._load_drugbank_data(
            drugbank_path, atc_code
        )

        # Set device based on GPU availability
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Prepare lists to contain model details
        self.task_lists: list[list[str]] = []
        self.use_features_list: list[bool] = []
        self.model_lists: list[list[MPNN]] = []
        self.scaler_lists: list[list[StandardScaler | None]] = []

        self._load_model_ensembles(models_dir)

        # TODO: This is currently assuming we should always use our own fingerprints
        self.use_features = True

    def _load_drugbank_data(self, drugbank_path, atc_code):
        """Load the drugbank data and map ATC codes to each drugbank index"""
        if not drugbank_path:
            drugbank = drugbank_atc_filtered = None
            return drugbank, drugbank_atc_filtered

        drugbank = read_drugbank_data(drugbank_path)

        drugbank_atc_filtered = filter_drugbank_by_atc(atc_code, drugbank)

        return drugbank, drugbank_atc_filtered

    def _load_model_ensembles(self, models_dir):
        """Load model ensembles from the specified directory."""
        self.task_lists = []
        self.model_lists = []
        model_dirs = sorted(Path(models_dir).iterdir())

        for model_dir in model_dirs:
            model_paths = sorted(model_dir.glob("**/*.pt"))
            models = [
                load_model(model_path, multicomponent=False)
                for model_path in model_paths
            ]
            task_names = load_output_columns(model_paths[0])

            self.task_lists.append(task_names)
            self.model_lists.append(models)

    @property
    def num_ensembles(self) -> int:
        """Get the number of ensembles."""
        return len(self.model_lists)

    @property
    def atc_code(self) -> str | None:
        """Get the ATC code."""
        return self._atc_code

    @atc_code.setter
    def atc_code(self, atc_code: str | None) -> None:
        """Set the ATC code and filter DrugBank based on provided ATC code.

        :param atc_code: The ATC code to filter the DrugBank reference set by.
                         If None, the entire DrugBank reference set will be used.
        """
        # Handle case of no DrugBank
        if self.drugbank is None:
            raise ValueError(
                "Cannot set ATC code if DrugBank reference is not provided."
            )

        # Validate ATC code
        if atc_code is not None and atc_code not in create_atc_code_mapping(
            self.drugbank
        ):
            raise ValueError(f"Invalid ATC code: {atc_code}")

        # Save ATC code
        self._atc_code = atc_code

        self.drugbank_atc_filtered = self._filter_drugbank_by_atc(atc_code)

    def predict(self, smiles: str | list[str]) -> dict[str, float] | pd.DataFrame:
        """Make predictions on a list of SMILES strings.

        :param smiles: A SMILES string or a list of SMILES strings.
        :return: If smiles is a string, returns a dictionary mapping property name to prediction.
                 If smiles is a list, returns a DataFrame containing the predictions with SMILES strings as the index
                 and property names as the columns.
        """
        # Convert SMILES to list if needed
        smiles, smiles_type = self._prepare_smiles(smiles=smiles)

        mols, smiles = self._filter_valid_molecules(smiles)

        # Compute physicochemical properties
        physchem_preds = compute_physicochemical_properties(
            all_smiles=smiles, mols=mols
        )

        fingerprints = compute_fingerprints(
            mols, self.use_features, self.fingerprint_multiprocessing_min
        )

        data_loader = self._build_dataloader(mols, fingerprints)

        task_to_preds = self._make_ensemble_predictions(data_loader)

        # Put preds in a DataFrame
        admet_preds = pd.DataFrame(task_to_preds, index=smiles)

        # Combine physicochemical and ADMET properties
        assert physchem_preds.index.equals(
            admet_preds.index
        ), "Internal Error: Indices do not match."
        preds = pd.concat((physchem_preds, admet_preds), axis=1)

        final_predictions = self._add_drugbank_percentiles(preds, smiles)
        # Convert to dictionary if SMILES type is string
        if smiles_type == str:
            final_predictions = final_predictions.iloc[0].to_dict()

        return final_predictions

    def _prepare_smiles(self, smiles: list[str]):
        """Ensure SMILES is a list and identify its type."""
        if isinstance(smiles, str):
            return [smiles], str
        return smiles, list

    def _filter_valid_molecules(self, smiles: list[str]):
        """Convert SMILES to RDKit molecules and filter out invalid ones."""
        valid_mols_smiles = [
            (Chem.MolFromSmiles(smile), smile)
            for smile in tqdm(smiles, desc="SMILES to Mol")
        ]
        valid_mols_smiles = [(mol, smile) for mol, smile in valid_mols_smiles if mol]

        if len(valid_mols_smiles) < len(smiles):
            print(
                f"Warning: {len(smiles) - len(valid_mols_smiles):,} invalid molecules removed."
            )

        mols, filtered_smiles = (
            zip(*valid_mols_smiles) if valid_mols_smiles else ([], [])
        )
        return mols, filtered_smiles

    def _build_dataloader(self, mols, fingerprints):
        """Create a DataLoader for model predictions."""
        data_points = [
            MoleculeDatapoint(mol=mol, x_d=fingerprint)
            for mol, fingerprint in zip(mols, fingerprints)
        ]
        dataset = MoleculeDataset(data=data_points)
        return build_dataloader(
            dataset=dataset, num_workers=self.num_workers, shuffle=False
        )

    def _make_ensemble_predictions(self, data_loader):
        """Run predictions across model ensembles."""

        assert len(self.task_lists) == len(
            self.model_lists
        ), "Internal error: amount of retrieved models does not match the amount of tasks"

        task_to_preds = {}
        for tasks, models in tqdm(
            zip(self.task_lists, self.model_lists),
            total=self.num_ensembles,
            desc="model ensembles",
        ):
            with torch.inference_mode():
                trainer = pl.Trainer(
                    logger=None, enable_progress_bar=True, accelerator="cpu", devices=1
                )

                # Shape of preds: (models x dataloaders x molecules x tasks [ADMET predictions])
                preds = np.array(
                    [
                        trainer.predict(model=model, dataloaders=data_loader)
                        for model in models
                    ]
                )

                # We perform the mean twice to merge the predictions across models and across data loaders
                # This works under the assumption that we always have one single data loader and multiple models
                preds: np.ndarray = np.mean(np.mean(preds, axis=0), axis=0)

            for i, task in enumerate(tasks):
                task_to_preds[task] = preds[:, i]

        return task_to_preds

    def _add_drugbank_percentiles(self, preds: pd.DataFrame, smiles: list[str]):
        """Compute and add DrugBank percentiles if DrugBank data is available."""
        if self.drugbank is None:
            return preds

        drugbank_suffix = get_drugbank_suffix(self.atc_code)

        drugbank_percentiles = {
            f"{property_name}_{drugbank_suffix}": [
                percentileofscore(self.drugbank_atc_filtered[property_name], value)
                for value in preds[property_name].values
            ]
            for property_name in preds.columns
        }
        drugbank_percentiles_df = pd.DataFrame(drugbank_percentiles, index=smiles)

        return pd.concat([preds, drugbank_percentiles_df], axis=1)
