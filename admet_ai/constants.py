"""Contains constants used throughout ADMET-AI."""
from pathlib import Path

# Paths to DrugBank reference data and models
# TODO: update DrugBank path once it's added to the repo
FILES_DIR = Path(__file__).parent / "files"
DEFAULT_ADMET_PATH = FILES_DIR / "data" / "admet.csv"
DEFAULT_DRUGBANK_PATH = None  # FILES_DIR / "data" / "drugbank_approved.csv"
DEFAULT_MODELS_DIR = FILES_DIR / "models"
