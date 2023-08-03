"""Plot an ADMET prediction in context of a reference dataset (e.g., DrugBank)."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm

from constants import ADMET_GROUPS, ADMET_PLOTTING_DETAILS, DATASET_TO_TYPE


BLUE = sns.color_palette()[0]


def plot_predictions_in_context(
    preds_path: Path,
    reference_path: Path,
    save_dir: Path,
    num_plots_per_row: int = 3,
    bins: int = 50,
    alpha: float = 0.5,
    linewidth: int = 2,
) -> None:
    """Plot an ADMET prediction in context of a reference dataset (e.g., DrugBank).

    :param preds_path: Path to a CSV file containing predictions.
    :param reference_path: Path to a CSV file containing reference data.
    :param save_dir: Path to directory where the plot will be saved.
    :param num_plots_per_row: Number of plots to display per row.
    :param bins: Number of bins to use for the histogram.
    :param alpha: Transparency of the histogram.
    :param linewidth: Width of the vertical line indicating the prediction.
    """
    # Load data
    preds = pd.read_csv(preds_path)
    reference = pd.read_csv(reference_path)

    # Loop over the predicted molecules and plot in context of reference data
    for i, pred in tqdm(preds.iterrows(), total=len(preds)):
        # Create directory for saving plots for the molecule
        pred_save_dir = save_dir / f"molecule_{i}"
        pred_save_dir.mkdir(parents=True, exist_ok=True)

        # Plot predictions in context of reference data for each ADMET group
        for admet_group in ADMET_GROUPS:
            plt.clf()

            # Create subplots with at most num_plots_per_row
            n_plots = len(ADMET_PLOTTING_DETAILS[admet_group])
            n_rows = math.ceil(n_plots / num_plots_per_row)
            n_cols = min(n_plots, num_plots_per_row)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
            axes = axes.flatten()

            # Add supertitle with ADMET group name
            fig.suptitle(admet_group.capitalize(), fontsize=20)

            # Remove unused axes
            for i in range(n_plots, len(axes)):
                fig.delaxes(axes[i])

            # Plot each property
            for ax, prop, prop_dict in zip(
                axes,
                ADMET_PLOTTING_DETAILS[admet_group],
                ADMET_PLOTTING_DETAILS[admet_group].values(),
            ):
                # Get prediction value
                prop_pred = pred[prop]

                # Compute percentile of prediction in reference data
                percentile = stats.percentileofscore(reference[prop], prop_pred)

                # Set x-axis label value
                value = prop_dict["value"]
                if DATASET_TO_TYPE[prop] == "classification":
                    value = r"$\it{probability}$ of " + value

                # PLot histogram of reference data
                sns.histplot(reference[prop], bins=bins, color=BLUE, alpha=alpha, ax=ax)
                ax.set_xlabel(
                    rf"{prop_dict['lower']} $\leftarrow$ {value} $\rightarrow$ {prop_dict['upper']}"
                )

                # Plot vertical line for prediction value
                ax.axvline(
                    prop_pred,
                    color="r",
                    linestyle="--",
                    linewidth=linewidth,
                    label=f"Molecule (percentile: {percentile:.1f}%)",
                )
                ax.legend()

            plt.savefig(pred_save_dir / f"{admet_group}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    from tap import tapify

    tapify(plot_predictions_in_context)
