"""Sets the config parameters for the ADMET-AI Flask app object."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
ADMET_DIR = DATA_DIR / "admet"
MODEL_DIR = DATA_DIR / "models"
DRUGBANK_DIR = DATA_DIR / "drugbank"
DRUGBANK_PATH = DRUGBANK_DIR / "drugbank_approved.csv"
