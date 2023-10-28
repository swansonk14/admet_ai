"""Defines functions for the DrugBank approved reference set."""
from collections import defaultdict
from functools import lru_cache
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import percentileofscore

from admet_ai.web.app import app
from admet_ai.web.app.utils import replace_svg_dimensions

matplotlib.use("Agg")


DRUGBANK_DF = pd.DataFrame()
ATC_CODE_TO_DRUGBANK_INDICES: dict[str, list[int]] = {}


def load_drugbank() -> None:
    """Loads the reference set of DrugBank approved molecules with their model predictions."""
    print("--- LOADING DRUGBANK ---")

    # Set up global variables
    global DRUGBANK_DF, ATC_CODE_TO_DRUGBANK_INDICES

    # Load DrugBank DataFrame
    DRUGBANK_DF = pd.read_csv(app.config["DRUGBANK_PATH"])

    # Map ATC codes to all indices of the DRUGBANK_DF with that ATC code
    atc_code_to_drugbank_indices = defaultdict(set)
    for atc_column in [
        column for column in DRUGBANK_DF.columns if column.startswith("atc_")
    ]:
        for index, atc_codes in DRUGBANK_DF[atc_column].dropna().items():
            for atc_code in atc_codes.split(";"):
                atc_code_to_drugbank_indices[atc_code.lower()].add(index)

    # Save ATC code to indices mapping to global variable and convert set to sorted list
    ATC_CODE_TO_DRUGBANK_INDICES = {
        atc_code: sorted(indices)
        for atc_code, indices in atc_code_to_drugbank_indices.items()
    }


def get_drugbank(atc_code: str | None = None) -> pd.DataFrame:
    """Get the DrugBank reference DataFrame, optionally filtered by ATC code.

    :param atc_code: The ATC code to filter by. If None or 'all', returns the entire DrugBank.
    :return: A DataFrame containing the DrugBank reference set, optionally filtered by ATC code.
    """
    if atc_code is None or atc_code == "all":
        return DRUGBANK_DF

    return DRUGBANK_DF.loc[ATC_CODE_TO_DRUGBANK_INDICES[atc_code]]


def compute_drugbank_percentile(
    task_name: str, predictions: np.ndarray, atc_code: str | None = None
) -> np.ndarray:
    """Computes the percentile of the predictions compared to the DrugBank approved molecules.

    :param task_name: The name of the task (property) that is predicted.
    :param predictions: A 1D numpy array of predictions.
    :param atc_code: The ATC code to filter by. If None or 'all', returns the entire DrugBank.
    :return: A 1D numpy array of percentiles of the predictions compared to the DrugBank approved molecules.
    """
    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    # Compute percentiles
    return percentileofscore(drugbank[task_name], predictions)


@lru_cache()
def get_drugbank_unique_atc_codes() -> list[str]:
    """Get the unique ATC codes in the DrugBank reference set.

    :return: A list of unique ATC codes in the DrugBank reference set.
    """
    return sorted(
        {
            atc_code.lower()
            for atc_column in [
                column for column in DRUGBANK_DF.columns if column.startswith("atc_")
            ]
            for atc_codes in DRUGBANK_DF[atc_column].dropna().str.split(";")
            for atc_code in atc_codes
        }
    )


@lru_cache()
def get_drugbank_tasks() -> list[str]:
    """Get the tasks (properties) predicted by the DrugBank reference set.

    :return: A list of tasks (properties) predicted in the DrugBank reference set.
    """
    non_task_columns = ["name", "smiles"] + [
        column for column in DRUGBANK_DF.columns if column.startswith("atc_")
    ]
    task_columns = set(DRUGBANK_DF.columns) - set(non_task_columns)

    return sorted(task_columns)


def plot_drugbank_reference(
    preds_df: pd.DataFrame,
    x_task: str | None = None,
    y_task: str | None = None,
    atc_code: str | None = None,
) -> str:
    """Creates a 2D scatter plot of the DrugBank reference set vs the new set of molecules on two tasks.

    :param preds_df: A DataFrame containing the predictions on the new molecules.
    :param x_task: The name of the task to plot on the x-axis.
    :param y_task: The name of the task to plot on the y-axis.
    :param atc_code: The ATC code to filter the DrugBank reference set by.
    :return: A string containing the SVG of the plot.
    """
    # Set default values
    if x_task is None:
        x_task = "HIA_Hou"

    if y_task is None:
        y_task = "ClinTox"

    if atc_code is None:
        atc_code = "all"

    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    # Scatter plot of DrugBank molecules with density coloring
    sns.scatterplot(
        x=drugbank[x_task],
        y=drugbank[y_task],
        edgecolor=None,
        label="DrugBank Approved" + (" (ATC filter)" if atc_code != "all" else ""),
    )

    # Set input label
    input_label = "Input Molecule" + ("s" if len(preds_df) > 1 else "")

    # Scatter plot of new molecules
    if len(preds_df) > 0:
        sns.scatterplot(
            x=preds_df[f"{x_task}_prediction"],
            y=preds_df[f"{y_task}_prediction"],
            color="red",
            marker="*",
            s=200,
            label=input_label,
        )

    # Title
    plt.title(
        f"{input_label} vs DrugBank Approved"
        + (f"\nATC = {atc_code}" if atc_code != "all" else "")
    )

    # Save plot as svg to pass to frontend
    buf = BytesIO()
    plt.savefig(buf, format="svg")
    plt.close()
    buf.seek(0)
    drugbank_svg = buf.getvalue().decode("utf-8")

    # Set the SVG width and height to 100%
    drugbank_svg = replace_svg_dimensions(drugbank_svg)

    return drugbank_svg
