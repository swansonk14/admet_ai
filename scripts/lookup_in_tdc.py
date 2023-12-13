"""Look up molecules by SMILES in the TDC to identify their ADMET properties."""
from pathlib import Path

import pandas as pd
from rdkit import Chem

from tdc_constants import ADMET_ALL_SMILES_COLUMN


def preprocess_data(data: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """Preprocess data by removing missing or invalid SMILES and setting the index as the canonical SMILES.

    :param data: A DataFrame containing molecules.
    :param smiles_column: The column in data containing SMILES.
    :return: The preprocessed DataFrame.
    """
    # Drop rows without SMILES
    original_size = len(data)
    data = data.dropna(subset=[smiles_column])

    if len(data) < original_size:
        print(f"Dropped {original_size - len(data):,} rows without SMILES.")

    # Drop rows with invalid SMILES
    original_size = len(data)
    data_mols = data[smiles_column].apply(Chem.MolFromSmiles)

    data = data[data_mols.notnull()]
    data_mols = data_mols[data_mols.notnull()]

    if len(data) < original_size:
        print(f"Dropped {original_size - len(data):,} rows with invalid SMILES.")

    # Set canonical SMILES as index
    data = data.set_index(data_mols.apply(lambda mol: Chem.MolToSmiles(mol)))

    return data


def merge_tdc_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate rows in the TDC data.

    :param data: A DataFrame containing TDC data with canonical SMILES as the index.
    """
    # Get duplicate canonical SMILES
    smiles_counts = data.index.value_counts()
    duplicate_smiles = [smiles for smiles, count in smiles_counts.items() if count > 1]

    # Ensure no conflicting values in duplicate rows
    for smiles in duplicate_smiles:
        duplicate_rows = data.loc[smiles]

        for column in duplicate_rows.columns:
            values = duplicate_rows[column].dropna().unique()

            if len(values) > 1:
                raise ValueError(
                    f"Found conflicting values for column {column} in duplicate rows with canonical SMILES {smiles}."
                )

    # Merge duplicate rows
    data = data.groupby(level=0).first()

    return data


def lookup_in_tdc(
    data_path: Path,
    tdc_all_path: Path,
    smiles_column: str = ADMET_ALL_SMILES_COLUMN,
    tdc_smiles_column: str = ADMET_ALL_SMILES_COLUMN,
    save_path: Path | None = None,
) -> None:
    """Look up molecules by SMILES in the TDC to identify their ADMET properties.

    :param data_path: Path to a CSV file containing molecules to look up.
    :param tdc_all_path: Path to a CSV file containing all TDC molecules and their ADMET properties.
    :param smiles_column: Column in data_path containing SMILES.
    :param tdc_smiles_column: Column in tdc_all_path containing SMILES.
    :param save_path: Path to a CSV file where the data will be saved. If None, overwrites data_path.
    """
    # Load data
    data = pd.read_csv(data_path)
    tdc_data = pd.read_csv(tdc_all_path)

    # Preprocess datasets to remove missing or invalid SMILES and set canonical SMILES as index
    data = preprocess_data(data=data, smiles_column=smiles_column)
    tdc_data = preprocess_data(data=tdc_data, smiles_column=tdc_smiles_column)

    # Ensure unique canonical SMILES in data
    if len(data.index) != len(set(data.index)):
        raise ValueError("Found duplicate canonical SMILES.")

    # Remove SMILES column from TDC data
    tdc_data.drop(columns=[tdc_smiles_column], inplace=True)

    # Merge duplicates in TDC data
    tdc_data = merge_tdc_duplicates(data=tdc_data)

    # Look up molecules from data in TDC and copy their ADMET properties
    data = data.merge(tdc_data, how="left", left_index=True, right_index=True)

    # Save data
    if save_path is None:
        save_path = data_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(lookup_in_tdc)
