"""Compute RDKit features for the Therapeutics Data Commons (TDC) ADMET datasets"""

import shutil
import subprocess
from pathlib import Path
from typing import Literal

from chemfunc import compute_fingerprints

import numpy as np
import pandas as pd
from tqdm import tqdm


# From ChemFunc
def save_fingerprints(
    data_path: Path,
    save_path: Path,
    smiles_column: str,
    fingerprint_type: Literal["morgan", "rdkit"] = "rdkit",
) -> None:
    """Saves fingerprints for molecules in a dataset.

    :param data_path: Path to a CSV file containing molecules.
    :param save_path: Path to a NPZ file where the fingerprints are saved
    :param fingerprint_type: The type of fingerprint to compute.
    :param smiles_column: Name of column containing SMILES strings.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Get SMILES
    smiles = data[smiles_column].tolist()

    # Compute fingerprints
    fingerprints = compute_fingerprints(mols=smiles, fingerprint_type=fingerprint_type)

    # Save fingerprints
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, fingerprints)


def compute_rdkit_features(data_dir: Path, smiles_column: str) -> None:
    """Compute RDKit features for the Therapeutics Data Commons (TDC) ADMET datasets.

    :param data_dir: A directory containing CSV files with TDC ADMET data.
    :param smiles_column: The name of the column containing SMILES strings.
    """
    # Get dataset paths
    data_paths = sorted(data_dir.glob("**/*.csv"))

    # Compute features for each dataset using chemfunc
    for data_path in tqdm(data_paths):
        save_fingerprints(
            data_path=data_path,
            save_path=data_path.with_suffix(".npz"),
            smiles_column=smiles_column,
            fingerprint_type="rdkit",
        )


if __name__ == "__main__":
    from tap import tapify

    tapify(compute_rdkit_features)
