"""Defines functions for the DrugBank approved reference set."""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from admet_ai.admet_info import get_admet_id_to_name
from admet_ai.constants import (
    DEFAULT_DRUGBANK_PATH,
    DRUGBANK_ATC_NAME_PREFIX,
    DRUGBANK_ATC_PREFIX,
    DRUGBANK_DELIMITER,
    DRUGBANK_ID_COLUMN,
    DRUGBANK_NAME_COLUMN,
    DRUGBANK_SMILES_COLUMN,
)


DRUGBANK_DF = pd.DataFrame()
ATC_CODE_TO_DRUGBANK_INDICES: dict[str, list[int]] = {}


def load_drugbank(drugbank_path: Path = DEFAULT_DRUGBANK_PATH) -> None:
    """Loads the reference set of DrugBank approved molecules with their model predictions.

    :param drugbank_path: The path to the DrugBank reference set.
    """
    # Set up global variables
    global DRUGBANK_DF, ATC_CODE_TO_DRUGBANK_INDICES

    # Load DrugBank DataFrame
    DRUGBANK_DF = read_drugbank_data(drugbank_path)

    # Save ATC code to indices mapping to global variable and convert set to sorted list
    ATC_CODE_TO_DRUGBANK_INDICES = create_atc_code_mapping(DRUGBANK_DF)


def read_drugbank_data(drugbank_path: Path) -> pd.DataFrame:
    """Load the drugbank data and returns a dataframe"""
    if not drugbank_path.exists():
        raise FileNotFoundError(
            f"The path to the drugbank archive is not correct: {drugbank_path}"
        )

    drugbank = pd.read_csv(drugbank_path)
    return drugbank


def create_atc_code_mapping(drugbank: pd.DataFrame) -> dict:
    """Map ATC codes to drugbank indices."""
    atc_code_to_drugbank_indices = defaultdict(set)

    # Map ATC codes to all indices of the drugbank with that ATC code
    for column in drugbank.columns:
        if column.startswith(DRUGBANK_ATC_NAME_PREFIX):
            for idx, atc_codes in drugbank[column].dropna().items():
                for atc_code in atc_codes.split(DRUGBANK_DELIMITER):
                    atc_code_to_drugbank_indices[atc_code.lower()].add(idx)

    return {
        atc_code: sorted(indices)
        for atc_code, indices in atc_code_to_drugbank_indices.items()
    }


def filter_drugbank_by_atc(atc_code: str, drugbank: pd.DataFrame) -> pd.DataFrame:
    """Filter DrugBank data by ATC code."""
    if not atc_code:
        return drugbank

    atc_code_to_drugbank_indices = create_atc_code_mapping(drugbank)

    if atc_code not in atc_code_to_drugbank_indices:
        raise ValueError(f"Invalid ATC code: {atc_code}")
    return drugbank.loc[atc_code_to_drugbank_indices[atc_code]]


def get_drugbank(atc_code: str | None = None) -> pd.DataFrame:
    """Get the DrugBank reference DataFrame, optionally filtered by ATC code.

    :param atc_code: The ATC code to filter by. If None or 'all', returns the entire DrugBank.
    :return: A DataFrame containing the DrugBank reference set, optionally filtered by ATC code.
    """
    if DRUGBANK_DF.empty:
        load_drugbank()

    if atc_code is None:
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


@lru_cache()
def get_drugbank_unique_atc_codes() -> list[str]:
    """Get the unique ATC codes in the DrugBank reference set.

    :return: A list of unique ATC codes in the DrugBank reference set.
    """
    drugbank = get_drugbank()

    return sorted(
        {
            atc_code.lower()
            for atc_column in [
                column
                for column in drugbank.columns
                if column.startswith(DRUGBANK_ATC_NAME_PREFIX)
            ]
            for atc_codes in drugbank[atc_column].dropna().str.split(DRUGBANK_DELIMITER)
            for atc_code in atc_codes
        }
    )


@lru_cache()
def get_drugbank_tasks_ids() -> list[str]:
    """Get the tasks (properties) predicted by the DrugBank reference set.

    :return: A list of tasks (properties) predicted in the DrugBank reference set.
    """
    drugbank = get_drugbank()

    non_task_columns = [
        DRUGBANK_ID_COLUMN,
        DRUGBANK_NAME_COLUMN,
        DRUGBANK_SMILES_COLUMN,
    ] + [
        column for column in drugbank.columns if column.startswith(DRUGBANK_ATC_PREFIX)
    ]
    task_columns = set(drugbank.columns) - set(non_task_columns)
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
