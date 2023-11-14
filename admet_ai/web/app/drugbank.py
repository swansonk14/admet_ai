"""Defines functions for the DrugBank approved reference set."""
from collections import defaultdict
from functools import lru_cache

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from admet_ai.web.app import app
from admet_ai.web.app.admet_info import get_admet_id_to_name

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


@lru_cache()
def get_drugbank_size(atc_code: str | None = None) -> int:
    """Get the number of molecules in the DrugBank reference set, optionally filtered by ATC code.

    :param atc_code: The ATC code to filter by. If None or 'all', returns the entire DrugBank.
    :return: The number of molecules in the DrugBank reference set, optionally filtered by ATC code.
    """
    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank(atc_code=atc_code)

    return len(drugbank)


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
    drugbank_task_names = sorted(
        [admet_id_to_name[task_id] for task_id in get_drugbank_tasks_ids()],
        key=lambda name: name.lower(),
    )

    return drugbank_task_names
