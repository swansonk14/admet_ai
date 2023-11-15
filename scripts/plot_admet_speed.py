"""Plot a comparison of the speed of ADMET websites."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGSIZE = (14, 10)
matplotlib.rcParams["font.size"] = 16


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

    # Plot results
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        x="Time (s)", y="Number of Molecules", hue="Website", data=results,
    )

    # Limit x-axis
    plt.xlim(0, max_time)

    # Label any bars that extend beyond maximum
    for bar in ax.patches:
        time = bar.get_width()
        if time > max_time:
            ax.text(
                max_time * 0.94,
                bar.get_y() + bar.get_height() / 2.0,
                f"{round(time):,}",
                va="center",
                color="white",
                fontsize=14,
            )

    # Hide y label
    plt.ylabel("")

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_admet_speed)
