import pytest
from admet_ai import ADMETModel
from pytest_check import check


@pytest.fixture
def model():
    """Fixture to initialize the ADMET model."""
    return ADMETModel()


def test_admet_model_prediction_consistency(model):
    """Test that predictions for a specific SMILES string remain consistent."""

    # Known SMILES string
    smiles_string = "O(c1ccc(cc1)CCOC)CC(O)CNC(C)C"

    # The expected predictions from the original run
    expected_predictions = {
        "molecular_weight": 267.369,
        "logP": 1.6131999999999997,
        "hydrogen_bond_acceptors": 4.0,
        "hydrogen_bond_donors": 2.0,
        "Lipinski": 4.0,
        "QED": 0.7135696186852986,
        "stereo_centers": 1.0,
        "tpsa": 50.72,
        "AMES": 0.05029045604169369,
        "BBB_Martins": 0.6027289688587188,
        "Bioavailability_Ma": 0.9117547869682312,
        "CYP1A2_Veith": 0.10642183870077133,
        "CYP2C19_Veith": 0.05501440726220608,
        "CYP2C9_Substrate_CarbonMangels": 0.08333753868937492,
        "CYP2C9_Veith": 0.004176703677512705,
        "CYP2D6_Substrate_CarbonMangels": 0.8224924325942993,
        "CYP2D6_Veith": 0.36821651458740234,
        "CYP3A4_Substrate_CarbonMangels": 0.21682136356830597,
        "CYP3A4_Veith": 0.007005595415830612,
        "Carcinogens_Lagunin": 0.190507273375988,
        "ClinTox": 0.118291936814785,
        "DILI": 0.01261518318206072,
        "HIA_Hou": 0.9990445852279664,
        "NR-AR-LBD": 0.00180250586126931,
        "NR-AR": 0.01616315431892872,
        "NR-AhR": 0.02598277237266302,
        "NR-Aromatase": 0.002084914408624172,
        "NR-ER-LBD": 0.003831711853854358,
        "NR-ER": 0.08347824439406396,
        "NR-PPAR-gamma": 0.000878037977963686,
        "PAMPA_NCATS": 0.8766368746757507,
        "Pgp_Broccatelli": 0.05037392545491457,
        "SR-ARE": 0.012489270721562206,
        "SR-ATAD5": 0.0002849712705938146,
        "SR-HSE": 0.00326909099239856,
        "SR-MMP": 0.0019159407005645336,
        "SR-p53": 0.00037771926436107605,
        "Skin_Reaction": 0.49268582463264465,
        "hERG": 0.5052155733108521,
        "Caco2_Wang": -4.630231262004588,
        "Clearance_Hepatocyte_AZ": 7.048268700053804,
        "Clearance_Microsome_AZ": -7.312460987935873,
        "Half_Life_Obach": -20.45813934129591,
        "HydrationFreeEnergy_FreeSolv": -11.140134329373197,
        "LD50_Zhu": 2.142849650273514,
        "Lipophilicity_AstraZeneca": -0.4439330795659552,
        "PPBR_AZ": 25.930748219326183,
        "Solubility_AqSolDB": -0.9562666231751195,
        "VDss_Lombardo": -1.060748671568551,
    }

    # Get the actual predictions
    actual_predictions = model.predict(smiles=smiles_string)

    # Compare the actual predictions to the expected predictions
    for key, expected_value in expected_predictions.items():
        assert key in actual_predictions, f"Missing key in actual predictions: {key}"
        check.almost_equal(
            actual_predictions[key],
            expected_value,
            rel=1e-5,
            msg=f"Prediction for {key} has changed. Expected: {expected_value}, "
            f"but got: {actual_predictions[key]}",
        )
