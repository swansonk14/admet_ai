"""Plot a comparison of the speed of ADMET-AI version for large-scale ADMET prediction."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGSIZE = (14, 10)
matplotlib.rcParams["font.size"] = 28
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def plot_admet_speed(
    results_path: Path, save_path: Path, sheet_name: str = "ADMET Speed Large Scale"
) -> None:
    """Plot a comparison of the speed of ADMET-AI version for large-scale ADMET prediction.

    :param results_path: Path to a CSV file containing the ADMET website speed results.
    :param save_path: Path to a PDF file where the plot will be saved.
    :param sheet_name: The name of the sheet containing the ADMET website speed results.
    """
    # Load speed results
    results = pd.read_excel(results_path, sheet_name=sheet_name)

    # Reformat ADMET-AI Version
    results["ADMET-AI Version"] = results["ADMET-AI Version"].apply(
        lambda version: version.replace(", ", "\n")
    )

    # Plot results
    plt.subplots(figsize=FIGSIZE)
    sns.lineplot(
        x="ADMET-AI Version",
        y="Time (h)",
        data=results,
        marker="o",
        markersize=20,
        linewidth=5,
    )

    # Turn off x-axis label
    plt.xlabel("")

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_admet_speed)
