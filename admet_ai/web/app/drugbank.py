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
from admet_ai.web.app.admet_info import get_admet_id_to_name, get_admet_name_to_id
from admet_ai.web.app.utils import replace_svg_dimensions

matplotlib.use("Agg")


DRUGBANK_DF = pd.DataFrame()
ATC_CODE_TO_DRUGBANK_INDICES: dict[str, list[int]] = {}


def load_drugbank() -> None:
    """Loads the reference set of DrugBank approved molecules with their model predictions."""
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
    property_name: str, predictions: np.ndarray, atc_code: str | None = None
) -> np.ndarray:
    """Computes the percentile of the predictions compared to the DrugBank approved molecules.

    :param property_name: The name of the property that is predicted.
    :param predictions: A 1D numpy array of predictions.
    :param atc_code: The ATC code to filter by. If None or 'all', returns the entire DrugBank.
    :return: A 1D numpy array of percentiles of the predictions compared to the DrugBank approved molecules.
    """
    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    # Compute percentiles
    return percentileofscore(drugbank[property_name], predictions)


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
def get_drugbank_tasks_ids() -> list[str]:
    """Get the tasks (properties) predicted by the DrugBank reference set.

    :return: A list of tasks (properties) predicted in the DrugBank reference set.
    """
    non_task_columns = ["name", "smiles"] + [
        column for column in DRUGBANK_DF.columns if column.startswith("atc_")
    ]
    task_columns = set(DRUGBANK_DF.columns) - set(non_task_columns)
    drugbank_task_ids = sorted(task_columns)

    return drugbank_task_ids


@lru_cache()
def get_drugbank_task_names() -> list[str]:
    """Get the names of the tasks (properties) predicted by the DrugBank reference set.

    :return: A list of task names (properties) predicted in the DrugBank reference set.
    """
    admet_id_to_name = get_admet_id_to_name()
    drugbank_task_names = [
        admet_id_to_name[task_id] for task_id in get_drugbank_tasks_ids()
    ]

    return drugbank_task_names


def plot_drugbank_reference(
    preds_df: pd.DataFrame,
    x_task_name: str | None = None,
    y_task_name: str | None = None,
    atc_code: str | None = None,
) -> str:
    """Creates a 2D scatter plot of the DrugBank reference set vs the new set of molecules on two tasks.

    :param preds_df: A DataFrame containing the predictions on the new molecules.
    :param x_task_name: The name of the task to plot on the x-axis.
    :param y_task_name: The name of the task to plot on the y-axis.
    :param atc_code: The ATC code to filter the DrugBank reference set by.
    :return: A string containing the SVG of the plot.
    """
    # Set default values
    if x_task_name is None:
        x_task_name = "Human Intestinal Absorption"

    if y_task_name is None:
        y_task_name = "Clinical Toxicity"

    if atc_code is None:
        atc_code = "all"

    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    # Map task names to IDs
    admet_name_to_id = get_admet_name_to_id()
    x_task_id = admet_name_to_id[x_task_name]
    y_task_id = admet_name_to_id[y_task_name]

    # Scatter plot of DrugBank molecules with density coloring
    sns.scatterplot(
        x=drugbank[x_task_id],
        y=drugbank[y_task_id],
        edgecolor=None,
        label="DrugBank Approved" + (" (ATC filter)" if atc_code != "all" else ""),
    )

    # Set input label
    input_label = "Input Molecule" + ("s" if len(preds_df) > 1 else "")

    # Scatter plot of new molecules
    if len(preds_df) > 0:
        sns.scatterplot(
            x=preds_df[x_task_id],
            y=preds_df[y_task_id],
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
    plt.xlabel(x_task_name)
    plt.ylabel(y_task_name)

    # Save plot as svg to pass to frontend
    buf = BytesIO()
    plt.savefig(buf, format="svg")
    plt.close()
    buf.seek(0)
    drugbank_svg = buf.getvalue().decode("utf-8")

    # Set the SVG width and height to 100%
    drugbank_svg = replace_svg_dimensions(drugbank_svg)

    return drugbank_svg
