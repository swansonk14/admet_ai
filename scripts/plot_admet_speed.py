"""Plot a comparison of the speed of ADMET websites."""
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
    results_path: Path,
    save_path: Path,
    sheet_name: str = "ADMET Speed",
    max_time: int = 800,
) -> None:
    """Plot a comparison of the speed of ADMET websites.

    :param results_path: Path to a CSV file containing the ADMET website speed results.
    :param save_path: Path to a PDF file where the plot will be saved.
    :param sheet_name: The name of the sheet containing the ADMET website speed results.
    :param max_time: The maximum time to plot on the x-axis.
    """
    # Load speed results
    results = pd.read_excel(results_path, sheet_name=sheet_name)

    # Melt results for seaborn plotting
    results = results.melt(
        id_vars="Website", var_name="Number of Molecules", value_name="Time (s)"
    )

    # Reformat number of molecules (e.g., "1,000 molecules" -> "1,000")
    results["Number of Molecules"] = results["Number of Molecules"].apply(
        lambda num: num.split(" ")[0]
    )

    # Plot results
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        x="Number of Molecules", y="Time (s)", hue="Website", data=results,
    )

    # Limit y-axis
    plt.ylim(0, max_time)

    # Label any bars that extend beyond maximum
    for bar in ax.patches:
        time = bar.get_height()
        if time > max_time:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max_time * 0.89,
                f"{round(time):,}",
                ha="center",
                color="black",
                fontsize=20,
                rotation=90,
            )

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_admet_speed)
