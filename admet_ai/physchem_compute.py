"""Compute physicochemical properties using RDKit."""
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAtomStereoCenters,
    CalcNumHBA,
    CalcNumHBD,
    CalcTPSA,
)
from tap import tapify
from tqdm import tqdm

from admet_ai.utils import load_and_preprocess_data


def lipinski_rule_of_five(mol: Chem.Mol) -> float:
    """Determines how many of the Lipinski rules are satisfied by the molecule.

    :param mol: An RDKit molecule.
    :return: The number of Lipinski rules satisfied by the molecule.
    """
    return float(
        sum(
            [
                MolWt(mol) <= 500,
                MolLogP(mol) <= 5,
                CalcNumHBA(mol) <= 10,
                CalcNumHBD(mol) <= 5,
            ]
        )
    )


PHYSCHEM_PROPERTY_TO_FUNCTION = {
    "molecular_weight": MolWt,
    "logP": MolLogP,
    "hydrogen_bond_acceptors": CalcNumHBA,
    "hydrogen_bond_donors": CalcNumHBD,
    "Lipinski": lipinski_rule_of_five,
    "QED": qed,
    "stereo_centers": CalcNumAtomStereoCenters,
    "tpsa": CalcTPSA,
}


def compute_physicochemical_properties(
    all_smiles: list[str], mols: list[Chem.Mol] | None = None
) -> pd.DataFrame:
    """Compute physicochemical properties for a list of molecules.

    :param all_smiles: A list of SMILES.
    :param mols: A list of RDKit molecules. If None, RDKit molecules will be computed from the SMILES.
    :return: A DataFrame containing the computed physicochemical properties with SMILES strings as the index.
    """
    # Compute RDKit molecules if needed
    if mols is None:
        mols = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]
    else:
        assert len(all_smiles) == len(mols)

    # Compute phyiscochemical properties and put in DataFrame with SMILES as index
    physchem_properties = pd.DataFrame(
        data=[
            {
                property_name: property_function(mol)
                for property_name, property_function in PHYSCHEM_PROPERTY_TO_FUNCTION.items()
            }
            for mol in tqdm(mols, desc="Computing physchem properties")
        ],
        index=all_smiles,
    )

    return physchem_properties


def physchem_compute(
    data_path: Path, save_path: Path | None = None, smiles_column: str = "smiles",
) -> None:
    """Compute physicochemical properties using RDKit.

    :param data_path: Path to a CSV file containing a dataset of molecules.
    :param save_path: Path to a CSV file where the computed properties will be saved. If None, defaults to data_path.
    :param smiles_column: Name of the column containing SMILES strings.
    """
    # Load and preprocess data
    data = load_and_preprocess_data(data_path=data_path, smiles_column=smiles_column)

    # Compute physicochemical properties
    physchem_properties = compute_physicochemical_properties(
        all_smiles=list(data.index)
    )

    # Merge data and preds
    data_with_preds = pd.concat((data, physchem_properties), axis=1)

    # Save predictions
    if save_path is None:
        save_path = data_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    data_with_preds.to_csv(save_path, index_label=smiles_column)


def physchem_compute_command_line() -> None:
    """Run physchem_compute from the command line."""
    tapify(physchem_compute)
