"""Sets the config parameters for the ADMET-AI Flask app object."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
ADMET_PATH = DATA_DIR / "admet.csv"
DRUGBANK_PATH = DATA_DIR / "drugbank_approved.csv"
LOW_PERFORMANCE_THRESHOLD = 0.6
NUM_WORKERS = 0
