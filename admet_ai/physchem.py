"""Compute physicochemical properties using RDKit."""
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
from tqdm import tqdm


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
