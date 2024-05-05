"""Plots a scatter plot of two properties of molecules compared to the DrugBank reference set."""
from pathlib import Path

import pandas as pd

from admet_ai.drugbank import get_drugbank
from admet_ai.plot import plot_drugbank_reference as _plot_drugbank_reference


def plot_drugbank_reference(
    data_path: Path,
    save_path: Path,
    x_property: str,
    y_property: str,
    max_molecule_num: int | None = None,
    atc_code: str | None = None,
) -> None:
    """Plots a scatter plot of two properties of molecules compared to the DrugBank reference set.

    :param data_path: The path to a CSV file containing the data to plot.
    :param save_path: The path to save the plots to.
    :param x_property: The name of the property to plot on the x-axis (e.g., "Clinical Toxicity").
    :param y_property: The name of the property to plot on the y-axis (e.g., "Human Intestinal Absorption").
    :param max_molecule_num: If provided, will display molecule numbers up to this number.
    :param atc_code: The ATC code to filter the DrugBank reference set by.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Create save directory
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot drugbank reference
    plot_bytes = _plot_drugbank_reference(
        preds_df=data,
        drugbank_df=get_drugbank(atc_code=atc_code),
        x_property_name=x_property,
        y_property_name=y_property,
        max_molecule_num=max_molecule_num,
        image_type=save_path.suffix.lstrip("."),
    )

    # Save plot
    with open(save_path, "wb") as f:
        f.write(plot_bytes)


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_drugbank_reference)
