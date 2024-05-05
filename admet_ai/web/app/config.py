"""Sets the config parameters for the ADMET-AI Flask app object."""
from admet_ai.constants import (
    DEFAULT_DRUGBANK_PATH,
    DEFAULT_MODELS_DIR,
)


MODELS_DIR = DEFAULT_MODELS_DIR
DRUGBANK_PATH = DEFAULT_DRUGBANK_PATH
LOW_PERFORMANCE_THRESHOLD = 0.6
NUM_WORKERS = 0
