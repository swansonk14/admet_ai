"""Script to get the names and SMILES for approved drugs from the DrugBank database."""
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from admet_ai.constants import (
    DRUGBANK_ATC_CODE_COLUMN,
    DRUGBANK_ATC_NAME_PREFIX,
    DRUGBANK_DELIMITER,
    DRUGBANK_ID_COLUMN,
    DRUGBANK_NAME_COLUMN,
    DRUGBANK_SMILES_COLUMN,
)

DRUGBANK_NAMESPACES = {"db": "http://www.drugbank.ca"}


def get_approved_smiles_from_drugbank(data_path: Path, save_path: Path) -> None:
    """Gets the names and SMILES for approved drugs from the DrugBank database.

    :param data_path: Path to the DrugBank database XML file.
    :param save_path: Path to a CSV where the SMILES should be saved.
    """
    # Get root of XML tree
    print("Loading DrugBank XML")
    drugbank = ET.parse(data_path).getroot()
    drugs = list(drugbank)

    approved_names = []
    approved_ids = []
    approved_smiles = []
    approved_atc_codes = []
    approved_atc_names = []

    # Loop through drugs to find approved drugs and get their SMILES
    for drug in tqdm(drugs):
        # Get DrugBank ID
        drugbank_ids = drug.findall("db:drugbank-id", DRUGBANK_NAMESPACES)
        drugbank_ids = tuple(
            drugbank_id.text
            for drugbank_id in drugbank_ids
            if drugbank_id.text.startswith("DB")
        )

        # DrugBank ID length validation
        if len(drugbank_ids) == 0:
            raise ValueError("DrugBank ID missing")

        # Get groups to determine approval status
        groups_list = drug.findall("db:groups", DRUGBANK_NAMESPACES)

        # Groups length validation
        if len(groups_list) == 0:
            continue
        elif len(groups_list) > 1:
            raise ValueError("More than one groups list found")

        # Determine approval status
        approved = False

        for group in groups_list[0].findall("db:group", DRUGBANK_NAMESPACES):
            if group.text == "approved":
                approved = True
                break

        if not approved:
            continue

        # Get calculated properties to determine SMILES
        calculated_properties_list = drug.findall(
            "db:calculated-properties", DRUGBANK_NAMESPACES
        )

        # Calculated properties length validation
        if len(calculated_properties_list) == 0:
            continue
        elif len(calculated_properties_list) > 1:
            raise ValueError("More than one calculated-properties list found")

        smiles = None
        for prop in calculated_properties_list[0].findall(
            "db:property", DRUGBANK_NAMESPACES
        ):
            kind = prop.find("db:kind", DRUGBANK_NAMESPACES)
            value = prop.find("db:value", DRUGBANK_NAMESPACES)
            new_smiles = value.text

            if kind.text == "SMILES":
                if smiles is None:
                    smiles = new_smiles
                elif smiles != new_smiles:
                    raise ValueError("More than one SMILES found")

        if smiles is None:
            continue

        # Get name of drug
        names = drug.findall("db:name", DRUGBANK_NAMESPACES)

        # Names length validation
        if len(names) == 0:
            raise ValueError(f"No name found for {smiles}")
        elif len(names) > 1:
            raise ValueError("More than one name found")

        name = names[0].text

        # Get ATC codes list
        atcs_list = drug.findall("db:atc-codes", DRUGBANK_NAMESPACES)

        # ATC codes list length validation
        if len(atcs_list) == 0:
            atc_codes = []
        elif len(atcs_list) > 1:
            raise ValueError("More than one ATC code list found")
        else:
            # Get ATC codes
            atc_codes = atcs_list[0].findall("db:atc-code", DRUGBANK_NAMESPACES)

        # Get unique ATC info
        drug_unique_atc_codes = set()
        drug_level_to_unique_atc_names = {level: set() for level in range(1, 5)}
        for atc_code in atc_codes:
            atc_levels = atc_code.findall("db:level", DRUGBANK_NAMESPACES)[::-1]

            if len(atc_levels) != 4:
                raise ValueError("ATC code does not have 4 levels")

            drug_unique_atc_codes.add(atc_levels[-1].get("code"))

            for level in range(1, 5):
                drug_level_to_unique_atc_names[level].add(atc_levels[level - 1].text)

        # Add info for approved drug
        approved_names.append(name)
        approved_ids.append(drugbank_ids)
        approved_smiles.append(smiles)
        approved_atc_codes.append(drug_unique_atc_codes)
        approved_atc_names.append(drug_level_to_unique_atc_names)

    # Create dataset of approved drugs, drop duplicates, and sort
    data = pd.DataFrame(
        {
            DRUGBANK_NAME_COLUMN: approved_names,
            DRUGBANK_ID_COLUMN: [DRUGBANK_DELIMITER.join(ids) for ids in approved_ids],
            DRUGBANK_SMILES_COLUMN: approved_smiles,
            DRUGBANK_ATC_CODE_COLUMN: [
                DRUGBANK_DELIMITER.join(sorted(atc_codes))
                for atc_codes in approved_atc_codes
            ],
            **{
                f"{DRUGBANK_ATC_NAME_PREFIX}_{level}": [
                    DRUGBANK_DELIMITER.join(sorted(level_to_atc_names[level]))
                    for level_to_atc_names in approved_atc_names
                ]
                for level in range(1, 5)
            },
        }
    )
    data.drop_duplicates(DRUGBANK_SMILES_COLUMN, inplace=True)
    data.sort_values(DRUGBANK_NAME_COLUMN, inplace=True)

    # Convert to RDKit SMILES and remove invalid molecules
    mols = data.smiles.apply(lambda x: Chem.MolFromSmiles(x))
    data = data[mols.notnull()]
    data.smiles = mols[mols.notnull()].apply(lambda x: Chem.MolToSmiles(x))

    # Save drugs
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(get_approved_smiles_from_drugbank)
