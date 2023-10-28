"""Defines functions for computing physicochemical properties of molecules."""
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


def compute_physicochemical_properties(
    smiles: list[str],
) -> tuple[list[str], list[list[float]]]:
    """Compute physicochemical properties for a list of molecules.

    :param smiles: A list of SMILES.
    :return: A tuple containing a list of property names and a list of properties (num_molecules, num_properties).
    """
    # Compute RDKit molecules
    # TODO: avoid recomputing the RDKit molecules if they have to be computed for Chemprop anyway
    # TODO: handle invalid molecules
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    # Set up physicochemical property functions
    property_functions = {
        "molecular_weight": MolWt,
        "logP": MolLogP,
        "hydrogen_bond_acceptors": CalcNumHBA,
        "hydrogen_bond_donors": CalcNumHBD,
        "Lipinski": lipinski_rule_of_five,
        "QED": qed,
        "stereo_centers": CalcNumAtomStereoCenters,
        "tpsa": CalcTPSA,
    }

    # Compute properties
    property_names = list(property_functions)
    properties = [
        [property_function(mol) for mol in mols]
        for property_name, property_function in property_functions.items()
    ]

    return property_names, properties
