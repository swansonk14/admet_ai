"""Defines functions for plotting for the ADMET-AI website."""
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from admet_ai.web.app.admet_info import get_admet_name_to_id
from admet_ai.web.app.drugbank import get_drugbank
from admet_ai.web.app.utils import replace_svg_dimensions


def plot_drugbank_reference(
    preds_df: pd.DataFrame,
    x_property_name: str | None = None,
    y_property_name: str | None = None,
    atc_code: str | None = None,
) -> str:
    """Creates a 2D scatter plot of the DrugBank reference set vs the new set of molecules on two properties.

    :param preds_df: A DataFrame containing the predictions on the new molecules.
    :param x_property_name: The name of the property to plot on the x-axis.
    :param y_property_name: The name of the property to plot on the y-axis.
    :param atc_code: The ATC code to filter the DrugBank reference set by.
    :return: A string containing the SVG of the plot.
    """
    # Set default values
    if x_property_name is None:
        x_property_name = "Human Intestinal Absorption"

    if y_property_name is None:
        y_property_name = "Clinical Toxicity"

    if atc_code is None:
        atc_code = "all"

    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    # Map property names to IDs
    admet_name_to_id = get_admet_name_to_id()
    x_property_id = admet_name_to_id[x_property_name]
    y_property_id = admet_name_to_id[y_property_name]

    # Scatter plot of DrugBank molecules with density coloring
    sns.scatterplot(
        x=drugbank[x_property_id],
        y=drugbank[y_property_id],
        edgecolor=None,
        label="DrugBank Approved" + (" (ATC filter)" if atc_code != "all" else ""),
    )

    # Set input label
    input_label = "Input Molecule" + ("s" if len(preds_df) > 1 else "")

    # Scatter plot of new molecules
    if len(preds_df) > 0:
        sns.scatterplot(
            x=preds_df[x_property_id],
            y=preds_df[y_property_id],
            color="red",
            marker="*",
            s=200,
            label=input_label,
        )

    # Set title
    plt.title(
        f"{input_label} vs DrugBank Approved"
        + (f"\nATC = {atc_code}" if atc_code != "all" else "")
    )

    # Set axis labels
    plt.xlabel(x_property_name)
    plt.ylabel(y_property_name)

    # Save plot as svg to pass to frontend
    buf = BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    drugbank_svg = buf.getvalue().decode("utf-8")

    # Set the SVG width and height to 100%
    drugbank_svg = replace_svg_dimensions(drugbank_svg)

    return drugbank_svg


# TODO: one for preds, one for DrugBank percentile
# TODO: how to handle negative numbers
def plot_radial_summary(
    molecule_preds: dict[str, float], property_names: list[str],
) -> str:
    """Creates a radial plot summary of the most important properties of a molecule.

    :param molecule_preds: A dictionary mapping property names to predictions.
    :param property_names: A list of property names to plot.
    :return: A string containing the SVG of the plot.
    """
    # Get the predictions for the properties
    admet_name_to_id = get_admet_name_to_id()
    preds = [
        molecule_preds[
            f"{admet_name_to_id[property_name]}_drugbank_approved_percentile"
        ]
        for property_name in property_names
    ]

    # Calculate the angles of the plot
    angles = np.linspace(0, 2 * np.pi, len(property_names), endpoint=False).tolist()

    # Complete the loop
    preds += preds[:1]
    angles += angles[:1]

    # Step 3: Create a plot
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # Plot the data
    ax.fill(angles, preds, color="red", alpha=0.25)
    ax.plot(angles, preds, color="red", linewidth=2)

    # Labels for categories
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(property_names)

    # Save plot as svg to pass to frontend
    buf = BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    radial_svg = buf.getvalue().decode("utf-8")

    # Set the SVG width and height to 100%
    radial_svg = replace_svg_dimensions(radial_svg)

    return radial_svg
