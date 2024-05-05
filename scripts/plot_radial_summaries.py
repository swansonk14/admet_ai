"""Plots radial summaries of each molecule's most important predicted properties."""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from admet_ai.plot import plot_radial_summary


def plot_radial_summaries(
    data_path: Path, save_dir: Path, image_type: str = "pdf",
) -> None:
    """Plots radial summaries of each molecule's most important predicted properties.

    :param data_path: The path to a CSV file containing the data to plot.
    :param save_dir: The directory to save the plots to.
    :param image_type: The image type for the plot (e.g., pdf).
    """
    # Load data
    data = pd.read_csv(data_path)

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get DrugBank percentile suffix
    percentile_suffix = "drugbank_approved_percentile"

    for column in data.columns:
        drugbank_index = column.find("drugbank")

        if drugbank_index != -1:
            percentile_suffix = column[drugbank_index:]
            break

    # Plot radial summaries for each molecule
    for i, row in tqdm(data.iterrows(), total=len(data)):
        plot_bytes = plot_radial_summary(
            property_id_to_percentile=row.to_dict(),
            percentile_suffix=percentile_suffix,
            image_type=image_type,
        )

        with open(save_dir / f"{i}.{image_type}", "wb") as f:
            f.write(plot_bytes)


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_radial_summaries)
