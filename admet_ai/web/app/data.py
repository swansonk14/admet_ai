"""Defines functions for ADMET info."""
import pandas as pd

from admet_ai.web.app import app


ADMET_DF = pd.DataFrame()


def load_admet_info() -> None:
    """Loads the ADMET info."""
    print("--- LOADING ADMET INFO ---")

    # Set up global variables
    global ADMET_DF

    # Load ADMET info DataFrame
    ADMET_DF = pd.read_csv(app.config["ADMET_PATH"])


def get_admet_info() -> pd.DataFrame:
    """Get the ADMET info.

    :return: A DataFrame containing the ADMET info.
    """
    return ADMET_DF
