"""Utility functions related to the ADMET-AI web server."""
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from chemprop.data.data import SMILES_TO_MOL
from flask import request
from rdkit import Chem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from werkzeug.utils import secure_filename

from admet_ai.web.app import app


SVG_WIDTH_PATTERN = re.compile(r"width=['\"]\d+(\.\d+)?[a-z]+['\"]")
SVG_HEIGHT_PATTERN = re.compile(r"height=['\"]\d+(\.\d+)?[a-z]+['\"]")


# TODO: provide option for selecting SMILES column
def get_smiles_from_request(smiles_column: str = "smiles") -> list[str]:
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


def replace_svg_dimensions(svg_content: str) -> str:
    """Replace the SVG width and height with 100%.

    :param svg_content: The SVG content.
    :return: The SVG content with the width and height replaced with 100%.
    """
    # Replacing the width and height with 100%
    svg_content = SVG_WIDTH_PATTERN.sub('width="100%"', svg_content)
    svg_content = SVG_HEIGHT_PATTERN.sub('height="100%"', svg_content)

    return svg_content


def smiles_to_svg(mol: str | Chem.Mol) -> str:
    """Converts a SMILES string to an SVG image of the molecule.

    :param mol: A SMILES string or RDKit molecule.
    :return: An SVG image of the molecule.
    """
    # Convert SMILES to Mol if needed
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    # Convert Mol to SVG
    d = MolDraw2DSVG(200, 200)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    smiles_svg = d.GetDrawingText()

    # Set the SVG width and height to 100%
    # TODO: get this to work
    # smiles_svg = replace_svg_dimensions(smiles_svg)

    return smiles_svg
