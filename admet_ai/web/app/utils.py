"""Utility functions related to the ADMET-AI web server."""
from pathlib import Path
from tempfile import TemporaryDirectory

from chemprop.data import get_header, get_smiles, cache_mol
from chemprop.data.data import SMILES_TO_MOL
from flask import request
from rdkit import Chem
from werkzeug.utils import secure_filename


def get_smiles_from_request() -> list[str]:
    """Gets SMILES from a request."""
    if request.form["textSmiles"] != "":
        smiles = request.form["textSmiles"].split()
    elif request.form["drawSmiles"] != "":
        smiles = [request.form["drawSmiles"]]
    else:
        # Upload data file with SMILES
        data = request.files["data"]
        data_name = secure_filename(data.filename)

        with TemporaryDirectory() as temp_dir:
            data_path = str(Path(temp_dir) / data_name)
            data.save(data_path)

            # Check if header is smiles
            possible_smiles = get_header(data_path)[0]

            # TODO: standardize format (i.e., header or no header)
            smiles = (
                [possible_smiles]
                if Chem.MolFromSmiles(possible_smiles) is not None
                else []
            )

            # Get remaining smiles
            smiles.extend(get_smiles(data_path))

    return smiles


def smiles_to_mols(smiles: list[str]) -> list[Chem.Mol]:
    """Convert a list of SMILES to a list of RDKit molecules with caching if turned on.

    :param smiles: A list of SMILES.
    :return: A list of RDKit molecules.
    """
    mols = []
    for smile in smiles:
        if smile in SMILES_TO_MOL:
            mol = SMILES_TO_MOL[smile]
        else:
            mol = Chem.MolFromSmiles(smile)

        mols.append(mol)

        if cache_mol():
            SMILES_TO_MOL[smile] = mol

    return mols
