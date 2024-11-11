"""Contains constants used for the Therapeutics Data Commons Data."""

from tdc.single_pred import ADME, Tox
from tdc.utils import retrieve_label_name_list


TDC_DATASET_TO_CLASS = {
    "Caco2_Wang": ADME,
    "PAMPA_NCATS": ADME,
    "HIA_Hou": ADME,
    "Pgp_Broccatelli": ADME,
    "Bioavailability_Ma": ADME,
    "Lipophilicity_AstraZeneca": ADME,
    "Solubility_AqSolDB": ADME,
    "HydrationFreeEnergy_FreeSolv": ADME,
    "BBB_Martins": ADME,
    "PPBR_AZ": ADME,
    "VDss_Lombardo": ADME,
    "CYP2C19_Veith": ADME,
    "CYP2D6_Veith": ADME,
    "CYP3A4_Veith": ADME,
    "CYP1A2_Veith": ADME,
    "CYP2C9_Veith": ADME,
    "CYP2C9_Substrate_CarbonMangels": ADME,
    "CYP2D6_Substrate_CarbonMangels": ADME,
    "CYP3A4_Substrate_CarbonMangels": ADME,
    "Half_Life_Obach": ADME,
    "Clearance_Hepatocyte_AZ": ADME,
    "Clearance_Microsome_AZ": ADME,
    "LD50_Zhu": Tox,
    "hERG": Tox,
    "herg_central": Tox,
    "hERG_Karim": Tox,
    "AMES": Tox,
    "DILI": Tox,
    "Skin_Reaction": Tox,
    "Carcinogens_Lagunin": Tox,
    "Tox21": Tox,
    "ToxCast": Tox,
    "ClinTox": Tox,
}
DATASET_TO_TYPE = {
    "Caco2_Wang": "regression",
    "PAMPA_NCATS": "classification",
    "HIA_Hou": "classification",
    "Pgp_Broccatelli": "classification",
    "Bioavailability_Ma": "classification",
    "Lipophilicity_AstraZeneca": "regression",
    "Solubility_AqSolDB": "regression",
    "HydrationFreeEnergy_FreeSolv": "regression",
    "BBB_Martins": "classification",
    "PPBR_AZ": "regression",
    "VDss_Lombardo": "regression",
    "CYP2C19_Veith": "classification",
    "CYP2D6_Veith": "classification",
    "CYP3A4_Veith": "classification",
    "CYP1A2_Veith": "classification",
    "CYP2C9_Veith": "classification",
    "CYP2C9_Substrate_CarbonMangels": "classification",
    "CYP2D6_Substrate_CarbonMangels": "classification",
    "CYP3A4_Substrate_CarbonMangels": "classification",
    "Half_Life_Obach": "regression",
    "Clearance_Hepatocyte_AZ": "regression",
    "Clearance_Microsome_AZ": "regression",
    "LD50_Zhu": "regression",
    "hERG": "classification",
    "herg_central": "classification",
    "hERG_Karim": "classification",
    "AMES": "classification",
    "DILI": "classification",
    "Skin_Reaction": "classification",
    "Carcinogens_Lagunin": "classification",
    "Tox21": "classification",
    "NR-AR": "classification",
    "NR-AR-LBD": "classification",
    "NR-AhR": "classification",
    "NR-Aromatase": "classification",
    "NR-ER": "classification",
    "NR-ER-LBD": "classification",
    "NR-PPAR-gamma": "classification",
    "SR-ARE": "classification",
    "SR-ATAD5": "classification",
    "SR-HSE": "classification",
    "SR-MMP": "classification",
    "SR-p53": "classification",
    "ToxCast": "classification",
    "ClinTox": "classification",
    "admet_regression": "regression",
    "admet_classification": "classification",
}
DATASET_TO_TYPE_LOWER = {
    dataset.lower(): dataset_type for dataset, dataset_type in DATASET_TO_TYPE.items()
}
DATASET_TYPE_TO_METRICS_COMMAND_LINE = {
    "classification": ["--metric", "binary-mcc"],
    "regression": ["--metric", "mae"],
}
DATASET_TO_LABEL_NAMES = {
    "herg_central": ["hERG_inhib"],
    "Tox21": retrieve_label_name_list("Tox21"),
    "ToxCast": retrieve_label_name_list("Toxcast"),
}
ADMET_GROUP_SEEDS = [1, 2, 3, 4, 5]
ADMET_ALL_SMILES_COLUMN = "smiles"
ADMET_GROUP_SMILES_COLUMN = "Drug"
ADMET_GROUP_TARGET_COLUMN = "Y"
