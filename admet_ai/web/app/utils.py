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


def get_smiles_from_request() -> tuple[list[str] | None, str | None]:
    """Gets SMILES from a request.

    :return: A tuple with a list of SMILES or None and an error message or None.
    """
    if request.form["text-smiles"] != "":
        smiles = request.form["text-smiles"].split()
    elif request.form["draw-smiles"] != "":
        smiles = [request.form["draw-smiles"]]
    else:
        # Upload data file with SMILES
        data = request.files["data"]
        data_name = secure_filename(data.filename)
        smiles_column = request.form["smiles-column"]

        with TemporaryDirectory() as temp_dir:
            data_path = str(Path(temp_dir) / data_name)
            data.save(data_path)
            df = pd.read_csv(data_path)

            if smiles_column in df:
                smiles = df[smiles_column].astype(str).tolist()
            else:
                return None, f"SMILES column '{smiles_column}' not found in data file."

    return smiles, None


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


def string_to_html_sup(string: str) -> str:
    """Converts a string with an exponential to HTML superscript.

    :param string: A string.
    :return: The string with an exponential in HTML superscript.
    """
    return re.sub(r"\^(\d+)", r"<sup>\1</sup>", string)


def string_to_latex_sup(string: str) -> str:
    """Converts a string with an exponential to LaTeX superscript.

    :param string: A string.
    :return: The string with an exponential in LaTeX superscript.
    """
    return re.sub(r"\^(\d+)", r"$^{\1}$", string)


def get_drugbank_suffix(atc_code: str | None) -> str:
    """Gets the DrugBank percentile suffix for the given ATC code.

    :param atc_code: The ATC code.
    """
    if atc_code is None:
        return "drugbank_approved_percentile"

    return f"drugbank_approved_{atc_code}_percentile"
