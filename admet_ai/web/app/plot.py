"""Defines functions for plotting for the ADMET-AI website."""
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG

from admet_ai.web.app.admet_info import (
    get_admet_id_to_units,
    get_admet_name_to_id,
    get_toxicity_ids,
)
from admet_ai.web.app.drugbank import get_drugbank
from admet_ai.web.app.utils import string_to_latex_sup


SVG_WIDTH_PATTERN = re.compile(r"width=['\"]\d+(\.\d+)?[a-z]+['\"]")
SVG_HEIGHT_PATTERN = re.compile(r"height=['\"]\d+(\.\d+)?[a-z]+['\"]")


def replace_svg_dimensions(svg_content: str) -> str:
    """Replace the SVG width and height with 100%.

    :param svg_content: The SVG content.
    :return: The SVG content with the width and height replaced with 100%.
    """
    # Replacing the width and height with 100%
    svg_content = SVG_WIDTH_PATTERN.sub('width="100%"', svg_content)
    svg_content = SVG_HEIGHT_PATTERN.sub('height="100%"', svg_content)

    return svg_content


def plot_drugbank_reference(
    preds_df: pd.DataFrame,
    x_property_name: str | None = None,
    y_property_name: str | None = None,
    atc_code: str | None = None,
    max_molecule_num: int | None = None,
) -> str:
    """Creates a 2D scatter plot of the DrugBank reference set vs the new set of molecules on two properties.

    :param preds_df: A DataFrame containing the predictions on the new molecules.
    :param x_property_name: The name of the property to plot on the x-axis.
    :param y_property_name: The name of the property to plot on the y-axis.
    :param atc_code: The ATC code to filter the DrugBank reference set by.
    :param max_molecule_num: If provided, will display molecule numbers up to this number.
    :return: A string containing the SVG of the plot.
    """
    # Set default values
    if x_property_name is None:
        x_property_name = "Human Intestinal Absorption"

    if y_property_name is None:
        y_property_name = "Clinical Toxicity"

    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    # Map property names to IDs
    admet_name_to_id = get_admet_name_to_id()
    x_property_id = admet_name_to_id[x_property_name]
    y_property_id = admet_name_to_id[y_property_name]

    # Scatter plot of DrugBank molecules with histogram marginals
    sns.jointplot(
        x=drugbank[x_property_id],
        y=drugbank[y_property_id],
        kind="scatter",
        marginal_kws=dict(bins=50, fill=True),
        label="DrugBank Reference",
    )

    # Scatter plot of new molecules
    if len(preds_df) > 0:
        sns.scatterplot(
            x=preds_df[x_property_id],
            y=preds_df[y_property_id],
            color="red",
            marker="*",
            s=200,
            label="Input Molecule" + ("s" if len(preds_df) > 1 else ""),
        )

    # Add molecule numbers
    if max_molecule_num is not None:
        # Create buffer around scatter plot points
        x_min, x_max = plt.xlim()
        x_range = x_max - x_min
        x_buffer = x_range * 0.01

        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        y_buffer = y_range * 0.01

        # Get x and y values for first max_molecule_num molecules
        x_vals = preds_df[x_property_id].values[:max_molecule_num]
        y_vals = preds_df[y_property_id].values[:max_molecule_num]

        # Add molecule numbers for first max_molecule_num molecules
        for i, (x_val, y_val) in enumerate(zip(x_vals, y_vals)):
            plt.text(
                x_val - x_buffer,
                y_val + y_buffer,
                str(i + 1),
                horizontalalignment="right",
                verticalalignment="bottom",
                c="red",
            )

    # Get ADMET property units
    admet_id_to_units = get_admet_id_to_units()

    x_property_units = admet_id_to_units[x_property_id]
    y_property_units = admet_id_to_units[y_property_id]

    x_property_units = x_property_units if x_property_units != "-" else "probability"
    y_property_units = y_property_units if y_property_units != "-" else "probability"

    # Set axis labels
    plt.xlabel(f"{x_property_name} ({string_to_latex_sup(x_property_units)})")
    plt.ylabel(f"{y_property_name} ({string_to_latex_sup(y_property_units)})")

    # Save plot as svg to pass to frontend
    buf = BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    drugbank_svg = buf.getvalue().decode("utf-8")

    # Set the SVG width and height to 100%
    drugbank_svg = replace_svg_dimensions(drugbank_svg)

    return drugbank_svg


def plot_radial_summary(
    property_id_to_percentile: dict[str, float], percentile_suffix: str = "",
) -> str:
    """Creates a radial plot summary of important properties of a molecule in terms of DrugBank approved percentiles.

    :param property_id_to_percentile: A dictionary mapping property IDs to their DrugBank approved percentiles.
                                      Keys are the property name along with the percentile_suffix.
    :param percentile_suffix: The suffix to add to the property names to get the DrugBank approved percentiles.
    :return: A string containing the SVG of the plot.
    """
    # Set max percentile
    max_percentile = 100

    # Set up properties
    properties = {
        "Blood-Brain Barrier Safe": {
            "percentile": max_percentile
            - property_id_to_percentile[f"BBB_Martins_{percentile_suffix}"],
        },
        "Non-\nToxic": {
            "percentile": max_percentile
            - max(
                property_id_to_percentile[f"{toxicity_name}_{percentile_suffix}"]
                for toxicity_name in get_toxicity_ids()
            ),
            "vertical_alignment": "bottom",
        },
        "Soluble": {
            "percentile": property_id_to_percentile[
                f"Solubility_AqSolDB_{percentile_suffix}"
            ],
            "vertical_alignment": "top",
        },
        "Bioavailable": {
            "percentile": property_id_to_percentile[
                f"Bioavailability_Ma_{percentile_suffix}"
            ],
            "vertical_alignment": "top",
        },
        "hERG\nSafe": {
            "percentile": max_percentile
            - property_id_to_percentile[f"hERG_{percentile_suffix}"],
            "vertical_alignment": "bottom",
        },
    }
    property_names = [property_name for property_name in properties]
    percentiles = [
        properties[property_name]["percentile"] for property_name in properties
    ]

    # Calculate the angles of the plot (angles start at pi / 2 and go counter-clockwise)
    angles = (
        (np.linspace(0, 2 * np.pi, len(properties), endpoint=False) + np.pi / 2)
        % (2 * np.pi)
    ).tolist()

    # Complete the loop
    percentiles += percentiles[:1]
    angles += angles[:1]

    # Create a plot
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))

    # Plot the data
    ax.fill(angles, percentiles, color="red", alpha=0.25)
    ax.plot(angles, percentiles, color="red", linewidth=2)

    # Set y limits
    ax.set_ylim(0, 100)

    # Labels for radial lines
    yticks = [0, 25, 50, 75, 100]
    yticklabels = [str(ytick) for ytick in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_rlabel_position(335)

    # Labels for categories
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(property_names)

    # Adjust xticklabels so they don't overlap the plot
    for label, property_name in zip(ax.get_xticklabels(), property_names):
        if "vertical_alignment" in properties[property_name]:
            label.set_verticalalignment(properties[property_name]["vertical_alignment"])

    # Make the plot square (to match square molecule images)
    ax.set_aspect("equal", "box")

    # Ensure no text labels are cut off
    plt.tight_layout()

    # Save plot as svg to pass to frontend
    buf = BytesIO()
    plt.savefig(buf, format="svg")
    plt.close()
    buf.seek(0)
    radial_svg = buf.getvalue().decode("utf-8")

    # Set the SVG width and height to 100%
    radial_svg = replace_svg_dimensions(radial_svg)

    return radial_svg


def plot_molecule_svg(mol: str | Chem.Mol) -> str:
    """Plots a molecule as an SVG image.

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
    smiles_svg = replace_svg_dimensions(smiles_svg)

    return smiles_svg
