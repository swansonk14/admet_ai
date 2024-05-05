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


SVG_WIDTH_PATTERN = re.compile(r"width=['\"]\d+(\.\d+)?[a-z]+['\"]")
SVG_HEIGHT_PATTERN = re.compile(r"height=['\"]\d+(\.\d+)?[a-z]+['\"]")


def get_smiles_from_request() -> tuple[list[str] | None, str | None]:
    """Gets SMILES from a request.

    :return: A tuple with a list of SMILES or None and an error message or None.
    """
    # Get SMILES from request
    if request.form["text-smiles"] != "":
        smiles = request.form["text-smiles"].split("\n")
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

    # Strip SMILES of whitespace
    smiles = [smile.strip() for smile in smiles]

    # Skip empty lines
    smiles = [smile for smile in smiles if smile != ""]

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
    return re.sub(r"\^(-?\d+)", r"<sup>\1</sup>", string)


def replace_svg_dimensions(svg_content: str) -> str:
    """Replace the SVG width and height with 100%.

    :param svg_content: The SVG content.
    :return: The SVG content with the width and height replaced with 100%.
    """
    # Replacing the width and height with 100%
    svg_content = SVG_WIDTH_PATTERN.sub('width="100%"', svg_content)
    svg_content = SVG_HEIGHT_PATTERN.sub('height="100%"', svg_content)

    return svg_content
