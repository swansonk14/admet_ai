"""Defines functions for ADMET info."""
import pandas as pd

from admet_ai.web.app import app


ADMET_DF = pd.DataFrame()
ADMET_ID_TO_NAME: dict[str, str] = {}
ADMET_NAME_TO_ID: dict[str, str] = {}
ADMET_ID_TO_UNITS: dict[str, str] = {}
TOXICITY_IDS: list[str] = []


def load_admet_info() -> None:
    """Loads the ADMET info."""
    # Set up global variables
    global ADMET_DF, ADMET_ID_TO_NAME, ADMET_ID_TO_UNITS, ADMET_NAME_TO_ID, TOXICITY_IDS

    # Load ADMET info DataFrame
    ADMET_DF = pd.read_csv(app.config["ADMET_PATH"])

    # Map ADMET IDs to names and vice versa
    ADMET_ID_TO_NAME = dict(zip(ADMET_DF["id"], ADMET_DF["name"]))
    ADMET_NAME_TO_ID = dict(zip(ADMET_DF["name"], ADMET_DF["id"]))

    # Map ADMET IDs to units
    ADMET_ID_TO_UNITS = dict(zip(ADMET_DF["id"], ADMET_DF["units"]))

    # Get toxicity IDs
    TOXICITY_IDS = ADMET_DF[ADMET_DF["category"] == "Toxicity"]["id"].tolist()


def get_admet_info() -> pd.DataFrame:
    """Get the ADMET info.

    :return: A DataFrame containing the ADMET info.
    """
    return ADMET_DF


def get_admet_id_to_name() -> dict[str, str]:
    """Get the ADMET ID to name mapping.

    :return: A dictionary mapping ADMET IDs to names.
    """
    return ADMET_ID_TO_NAME


def get_admet_name_to_id() -> dict[str, str]:
    """Get the ADMET name to ID mapping.

    :return: A dictionary mapping ADMET names to IDs.
    """
    return ADMET_NAME_TO_ID


def get_admet_id_to_units() -> dict[str, str]:
    """Get the ADMET ID to units mapping.

    :return: A dictionary mapping ADMET IDs to units.
    """
    return ADMET_ID_TO_UNITS


def get_toxicity_ids() -> list[str]:
    """Get the property IDs of all toxicity properties.

    :return: A list of property IDs of all toxicity properties.
    """
    return TOXICITY_IDS
