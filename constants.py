"""Contains constants used throughout the project."""
from tdc.utils import retrieve_label_name_list


ADME_DATASET_TO_TYPE = {
    'Caco2_Wang': 'regression',
    'PAMPA_NCATS': 'classification',
    'HIA_Hou': 'classification',
    'Pgp_Broccatelli': 'classification',
    'Bioavailability_Ma': 'classification',
    'Lipophilicity_AstraZeneca': 'regression',
    'Solubility_AqSolDB': 'regression',
    'HydrationFreeEnergy_FreeSolv': 'regression',
    'BBB_Martins': 'classification',
    'PPBR_AZ': 'regression',
    'VDss_Lombardo': 'regression',
    'CYP2C19_Veith': 'classification',
    'CYP2D6_Veith': 'classification',
    'CYP3A4_Veith': 'classification',
    'CYP1A2_Veith': 'classification',
    'CYP2C9_Veith': 'classification',
    'CYP2C9_Substrate_CarbonMangels': 'classification',
    'CYP2D6_Substrate_CarbonMangels': 'classification',
    'CYP3A4_Substrate_CarbonMangels': 'classification',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',

}
TOX_DATASET_TO_TYPE = {
    'LD50_Zhu': 'regression',
    'hERG': 'classification',
    'herg_central': 'classification',
    'hERG_Karim': 'classification',
    'AMES': 'classification',
    'DILI': 'classification',
    'Skin_Reaction': 'classification',
    'Carcinogens_Lagunin': 'classification',
    'Tox21': 'classification',
    'ToxCast': 'classification',
    'ClinTox': 'classification'
}
DATASET_TO_TYPE = ADME_DATASET_TO_TYPE | TOX_DATASET_TO_TYPE
DATASET_TO_TYPE_LOWER = {
    dataset.lower(): dataset_type
    for dataset, dataset_type in DATASET_TO_TYPE.items()
}
DATASET_TYPE_TO_METRICS_COMMAND_LINE = {
    'classification': ['--metric', 'prc-auc', '--extra_metrics', 'auc'],
    'regression': ['--metric', 'mae', '--extra_metrics', 'r2']
}
DATASET_TO_LABEL_NAMES = {
    'herg_central': ['hERG_inhib'],
    'Tox21': retrieve_label_name_list('Tox21'),
    'ToxCast': retrieve_label_name_list('Toxcast')
}
ADMET_GROUP_SEEDS = [1, 2, 3, 4, 5]
ADMET_ALL_SMILES_COLUMN = 'smiles'
ADMET_GROUP_SMILES_COLUMN = 'Drug'
ADMET_GROUP_TARGET_COLUMN = 'Y'
