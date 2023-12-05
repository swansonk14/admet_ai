"""Defines functions for ADMET-AI models."""
from admet_ai import ADMETModel
from admet_ai.web.app import app


ADMET_MODEL: ADMETModel | None = None


def load_admet_model() -> None:
    """Loads the models into memory."""
    global ADMET_MODEL

    ADMET_MODEL = ADMETModel(
        models_dir=app.config["MODELS_DIR"],
        drugbank_path=app.config["DRUGBANK_PATH"],
        num_workers=app.config["NUM_WORKERS"],
        cache_molecules=app.config["CACHE_MOLECULES"],
    )


def get_admet_model() -> ADMETModel:
    """Get the ADMET-AI model.

    :return: The ADMET-AI model.
    """
    return ADMET_MODEL
