"""Script to get the names and SMILES for approved drugs from the DrugBank database."""
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

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

    approved_smiles = []
    approved_names = []
    approved_atcs = []

    # Loop through drugs to find approved drugs and get their SMILES
    for drug in tqdm(drugs):
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
            continue
        elif len(atcs_list) > 1:
            raise ValueError("More than one ATC code list found")

        # Get ATC codes
        atc_codes = atcs_list[0].findall("db:atc-code", DRUGBANK_NAMESPACES)

        # Get unique ATC codes
        unique_atc_codes = set()
        for atc_code in atc_codes:
            atc_levels = atc_code.findall("db:level", DRUGBANK_NAMESPACES)

            if len(atc_levels) != 4:
                raise ValueError("ATC code does not have 4 levels")

            unique_atc_codes.add(tuple(atc_levels[-i].text for i in range(1, 5)))

        # Add info for approved drug
        approved_smiles.append(smiles)
        approved_names.append(name)
        approved_atcs.append(sorted(unique_atc_codes))

    # Create dataset of approved drugs, drop duplicates, and sort
    data = pd.DataFrame(
        {
            "name": approved_names,
            "smiles": approved_smiles,
            **{
                f"atc_{i + 1}": [
                    ";".join(atc_code[i] for atc_code in atc_codes)
                    for atc_codes in approved_atcs
                ]
                for i in range(4)
            },
        }
    )
    data.drop_duplicates("smiles", inplace=True)
    data.sort_values("name", inplace=True)

    # Convert to RDKit SMILES and remove invalid molecules
    mols = data.smiles.apply(lambda x: Chem.MolFromSmiles(x))
    data = data[mols.notnull()]
    data.smiles = mols[mols.notnull()].apply(lambda x: Chem.MolToSmiles(x))

    # Save drugs
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    from tap import tapify

    tapify(get_approved_smiles_from_drugbank)
