"""Utility functions for ADMET-AI."""
from pathlib import Path

import pandas as pd


def load_and_preprocess_data(
    data_path: Path, smiles_column: str = "smiles"
) -> pd.DataFrame:
    """Preprocess a dataset of molecules by removing missing SMILES and setting the SMILES as the index.

    :param data_path: Path to a CSV file containing a dataset of molecules.
    :param smiles_column: Name of the column containing SMILES strings.
    :return: A DataFrame containing the preprocessed data with SMILES strings as the index.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Remove missing SMILES
    original_num_molecules = len(data)
    data.dropna(subset=[smiles_column], inplace=True, ignore_index=True)

    # Warn if molecules were removed
    if len(data) < original_num_molecules:
        print(
            f"Warning: {original_num_molecules - len(data):,} molecules were removed "
            f"from the dataset because they were missing SMILES."
        )

    # Set SMILES as index
    data.set_index(smiles_column, inplace=True)

    return data


def get_drugbank_suffix(atc_code: str | None) -> str:
    """Gets the DrugBank percentile suffix for the given ATC code.

    :param atc_code: The ATC code.
    """
    if atc_code is None:
        return "drugbank_approved_percentile"

    return f"drugbank_approved_{atc_code}_percentile"
