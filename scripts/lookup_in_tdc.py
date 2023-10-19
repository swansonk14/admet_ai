"""Look up molecules by SMILES in the TDC to identify their ADMET properties."""
from pathlib import Path

import pandas as pd
from rdkit import Chem

from admet_ai.constants import ADMET_ALL_SMILES_COLUMN


def lookup_in_tdc(
    data_path: Path,
    tdc_all_path: Path,
    smiles_column: str = ADMET_ALL_SMILES_COLUMN,
    tdc_all_smiles_column: str = ADMET_ALL_SMILES_COLUMN,
    save_path: Path | None = None,
) -> None:
    """Look up molecules by SMILES in the TDC to identify their ADMET properties.

    :param data_path: Path to a CSV file containing molecules to look up.
    :param smiles_column: Column in data_path containing SMILES.
    :param tdc_all_path: Path to a CSV file containing all TDC molecules and their ADMET properties.
    :param tdc_all_smiles_column: Column in tdc_all_path containing SMILES.
    :param save_path: Path to a CSV file where the data will be saved. If None, overwrites data_path.
    """
    # Load data
    data = pd.read_csv(data_path)
    tdc_data = pd.read_csv(tdc_all_path)

    # Drop rows without SMILES
    original_size = len(data)
    data.dropna(subset=[smiles_column], inplace=True)

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
    data.set_index(data_mols.apply(
        lambda mol: Chem.MolToSmiles(mol)
    ), inplace=True)
    tdc_data.set_index(tdc_data[tdc_all_smiles_column].apply(
        lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    ), inplace=True)

    # Look up molecules from data in TDC and copy their ADMET properties
    data = data.merge(tdc_data, how="left", left_index=True, right_index=True)

    # Remove duplicate SMILES column
    if tdc_all_smiles_column != smiles_column:
        data.drop(columns=[tdc_all_smiles_column], inplace=True)

    # Save data
    if save_path is None:
        save_path = data_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)


if __name__ == '__main__':
    from tap import tapify

    tapify(lookup_in_tdc)
