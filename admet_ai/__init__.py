"""Import all submodules of admet_ai."""

__version__ = "1.4.0"

from admet_ai.admet_model import ADMETModel
from admet_ai.admet_predict import admet_predict

__all__ = ["ADMETModel", "admet_predict", "__version__"]
