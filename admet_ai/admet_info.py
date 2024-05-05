"""Defines functions for ADMET info."""
from pathlib import Path

import pandas as pd

from admet_ai.constants import DEFAULT_ADMET_PATH


ADMET_DF = pd.DataFrame()
ADMET_ID_TO_NAME: dict[str, str] = {}
ADMET_NAME_TO_ID: dict[str, str] = {}
ADMET_ID_TO_UNITS: dict[str, str] = {}


def load_admet_info(admet_path: Path = DEFAULT_ADMET_PATH) -> None:
    """Loads the ADMET info."""
    # Set up global variables
    global ADMET_DF, ADMET_ID_TO_NAME, ADMET_ID_TO_UNITS, ADMET_NAME_TO_ID

    # Load ADMET info DataFrame
    ADMET_DF = pd.read_csv(admet_path)

    # Map ADMET IDs to names and vice versa
    ADMET_ID_TO_NAME = dict(zip(ADMET_DF["id"], ADMET_DF["name"]))
    ADMET_NAME_TO_ID = dict(zip(ADMET_DF["name"], ADMET_DF["id"]))

    # Map ADMET IDs to units
    ADMET_ID_TO_UNITS = dict(zip(ADMET_DF["id"], ADMET_DF["units"]))


def lazy_load_admet_info(func: callable) -> callable:
    """Decorator to lazily load the ADMET info."""

    def wrapper(*args, **kwargs):
        if ADMET_DF.empty:
            load_admet_info()

        return func(*args, **kwargs)

    return wrapper


@lazy_load_admet_info
def get_admet_info() -> pd.DataFrame:
    """Get the ADMET info.

    :return: A DataFrame containing the ADMET info.
    """
    return ADMET_DF


@lazy_load_admet_info
def get_admet_id_to_name() -> dict[str, str]:
    """Get the ADMET ID to name mapping.

    :return: A dictionary mapping ADMET IDs to names.
    """
    return ADMET_ID_TO_NAME


@lazy_load_admet_info
def get_admet_name_to_id() -> dict[str, str]:
    """Get the ADMET name to ID mapping.

    :return: A dictionary mapping ADMET names to IDs.
    """
    return ADMET_NAME_TO_ID


@lazy_load_admet_info
def get_admet_id_to_units() -> dict[str, str]:
    """Get the ADMET ID to units mapping.

    :return: A dictionary mapping ADMET IDs to units.
    """
    return ADMET_ID_TO_UNITS
