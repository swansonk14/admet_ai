"""Utility functions related to the ADMET-AI web server."""
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from chemprop.data.data import SMILES_TO_MOL
from flask import request
from rdkit import Chem
from werkzeug.utils import secure_filename

from admet_ai.web.app import app


# TODO: handle smiles column not being present
def get_smiles_from_request() -> list[str]:
    """Gets SMILES from a request.

    :return: A list of SMILES.
    """
    if request.form["textSmiles"] != "":
        smiles = request.form["textSmiles"].split()
    elif request.form["drawSmiles"] != "":
        smiles = [request.form["drawSmiles"]]
    else:
        # Upload data file with SMILES
        data = request.files["data"]
        data_name = secure_filename(data.filename)
        smiles_column = request.form["smilesColumn"]

        with TemporaryDirectory() as temp_dir:
            data_path = str(Path(temp_dir) / data_name)
            data.save(data_path)
            smiles = pd.read_csv(data_path)[smiles_column].tolist()

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

        if app.config["CACHE_MOLECULES"]:
            SMILES_TO_MOL[smile] = mol

    return mols
