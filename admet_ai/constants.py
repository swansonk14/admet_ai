"""Contains constants used throughout the project."""
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
    "classification": ["--metric", "auc", "--extra_metrics", "prc-auc"],
    "regression": ["--metric", "mae", "--extra_metrics", "r2"],
}
DATASET_TO_LABEL_NAMES = {
    "herg_central": ["hERG_inhib"],
    "Tox21": retrieve_label_name_list("Tox21"),
    "ToxCast": retrieve_label_name_list("Toxcast"),
}
LABEL_NAME_TO_TYPE = {
    label: data_type
    for dataset, data_type in DATASET_TO_TYPE.items()
    for label in DATASET_TO_LABEL_NAMES.get(dataset, [dataset])
}
ADMET_GROUP_SEEDS = [1, 2, 3, 4, 5]
ADMET_ALL_SMILES_COLUMN = "smiles"
ADMET_GROUP_SMILES_COLUMN = "Drug"
ADMET_GROUP_TARGET_COLUMN = "Y"
ADMET_GROUPS = ["absorption", "distribution", "metabolism", "excretion", "toxicity"]
# TODO: add Tox21 below
# TODO: add units for regression datasets
# TODO: double check interpretation of values
ADMET_PLOTTING_DETAILS = {
    "absorption": {
        "Caco2_Wang": {
            "lower": "less permeable",
            "value": "permeability (Caco2)",
            "upper": "more permeable",
        },
        "PAMPA_NCATS": {
            "lower": "less likely",
            "value": "permeability (PAMPA)",
            "upper": "more likely",
        },
        "HIA_Hou": {
            "lower": "less likely",
            "value": "human intestinal absorption",
            "upper": "more likely",
        },
        "Pgp_Broccatelli": {
            "lower": "less likely",
            "value": "P-glycoprotein inhibition",
            "upper": "more likely",
        },
        "Bioavailability_Ma": {
            "lower": "less likely",
            "value": "bioavailability",
            "upper": "more likely",
        },
        "Lipophilicity_AstraZeneca": {
            "lower": "less lipophilic",
            "value": "lipophilicity",
            "upper": "more lipophilic",
        },
        "Solubility_AqSolDB": {
            "lower": "less soluble",
            "value": "solubility",
            "upper": "more soluble",
        },
        "HydrationFreeEnergy_FreeSolv": {
            "lower": "less energy",
            "value": "hydration free energy",
            "upper": "more energy",
        },
    },
    "distribution": {
        "BBB_Martins": {
            "lower": "less likely",
            "value": "blood-brain barrier penetration",
            "upper": "more likely",
        },
        "PPBR_AZ": {
            "lower": "less bound",
            "value": "plasma protein binding",
            "upper": "more bound",
        },
        "VDss_Lombardo": {
            "lower": "smaller volume",
            "value": "volume of distribution",
            "upper": "larger volume",
        },
    },
    "metabolism": {
        "CYP2C19_Veith": {
            "lower": "less likely",
            "value": "CYP P450 2C19 inhibition",
            "upper": "more likely",
        },
        "CYP2D6_Veith": {
            "lower": "less likely",
            "value": "CYP P450 2D6 inhibition",
            "upper": "more likely",
        },
        "CYP3A4_Veith": {
            "lower": "less likely",
            "value": "CYP P450 3A4 inhibition",
            "upper": "more likely",
        },
        "CYP1A2_Veith": {
            "lower": "less likely",
            "value": "CYP P450 1A2 inhibition",
            "upper": "more likely",
        },
        "CYP2C9_Veith": {
            "lower": "less likely",
            "value": "CYP P450 2C9 inhibition",
            "upper": "more likely",
        },
        "CYP2C9_Substrate_CarbonMangels": {
            "lower": "less likely",
            "value": "CYP P450 2C9 substrate",
            "upper": "more likely",
        },
        "CYP2D6_Substrate_CarbonMangels": {
            "lower": "less likely",
            "value": "CYP P450 2D6 substrate",
            "upper": "more likely",
        },
        "CYP3A4_Substrate_CarbonMangels": {
            "lower": "less likely",
            "value": "CYP P450 3A4 substrate",
            "upper": "more likely",
        },
    },
    "excretion": {
        "Half_Life_Obach": {
            "lower": "longer half-life",
            "value": "half life",
            "upper": "shorter half-life",
        },
        "Clearance_Hepatocyte_AZ": {
            "lower": "lower clearance",
            "value": "hepatocyte clearance",
            "upper": "higher clearance",
        },
        "Clearance_Microsome_AZ": {
            "lower": "lower clearance",
            "value": "microsome clearance",
            "upper": "higher clearance",
        },
    },
    "toxicity": {
        "LD50_Zhu": {
            "lower": "more toxic",
            "value": "LD50 toxicity",
            "upper": "less toxic",
        },
        "hERG": {
            "lower": "less likely",
            "value": "hERG inhibition (Wang)",
            "upper": "more likely",
        },
        "hERG_inhib": {
            "lower": "less likely",
            "value": "hERG inhibition (central)",
            "upper": "more likely",
        },
        "hERG_Karim": {
            "lower": "less likely",
            "value": "hERG inhibition (Karim)",
            "upper": "more likely",
        },
        "AMES": {
            "lower": "less likely",
            "value": "mutagenicity",
            "upper": "more likely",
        },
        "DILI": {
            "lower": "less likely",
            "value": "drug induced liver injury",
            "upper": "more likely",
        },
        "Skin_Reaction": {
            "lower": "less likely",
            "value": "skin reaction",
            "upper": "more likely",
        },
        "Carcinogens_Lagunin": {
            "lower": "less likely",
            "value": "carcinogenicity",
            "upper": "more likely",
        },
        "ClinTox": {
            "lower": "less likely",
            "value": "clinical toxicity",
            "upper": "more likely",
        },
    },
}
