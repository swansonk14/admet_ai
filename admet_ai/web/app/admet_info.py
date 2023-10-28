"""Defines functions for ADMET info."""
import pandas as pd

from admet_ai.web.app import app


ADMET_DF = pd.DataFrame()
ADMET_ID_TO_NAME: dict[str, str] = {}
ADMET_NAME_TO_ID: dict[str, str] = {}


def load_admet_info() -> None:
    """Loads the ADMET info."""
    print("--- LOADING ADMET INFO ---")

    # Set up global variables
    global ADMET_DF, ADMET_ID_TO_NAME, ADMET_NAME_TO_ID

    # Load ADMET info DataFrame
    ADMET_DF = pd.read_csv(app.config["ADMET_PATH"])

    # Map ADMET IDs to names and vice versa
    ADMET_ID_TO_NAME = dict(zip(ADMET_DF["id"], ADMET_DF["name"]))
    ADMET_NAME_TO_ID = dict(zip(ADMET_DF["name"], ADMET_DF["id"]))


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
