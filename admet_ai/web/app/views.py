"""Defines the routes of the ADMET-AI Flask app."""
from uuid import uuid4
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import numpy as np
import pandas as pd
from chemprop.data import get_header, get_smiles
from flask import (
    after_this_request,
    jsonify,
    render_template,
    request,
    Response,
    send_file,
    session,
)
from rdkit import Chem
from werkzeug.utils import secure_filename

from admet_ai.web.app import app
from admet_ai.web.app.drugbank import (
    compute_drugbank_percentile,
    get_drugbank_tasks,
    get_drugbank_unique_atc_codes,
    plot_drugbank_reference,
)
from admet_ai.web.app.models import predict_all_models
from admet_ai.web.app.physchem import compute_physicochemical_properties


USER_TO_PREDS = {}


def render(**kwargs) -> str:
    """Renders the page with specified kwargs"""
    return render_template(
        "index.html",
        drugbank_atc_codes=["all"] + get_drugbank_unique_atc_codes(),
        drugbank_tasks=get_drugbank_tasks(),
        max_molecules=app.config["MAX_MOLECULES"],
        **kwargs,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the page and makes predictions if the method is POST."""
    # Assign user ID to session
    if "user_id" not in session:
        session["user_id"] = uuid4().hex

    # If GET request, simply return the page; otherwise if POST request, make predictions
    if request.method == "GET":
        return render()

    # Get the SMILES from the request
    if request.form["textSmiles"] != "":
        smiles = request.form["textSmiles"].split()
    elif request.form["drawSmiles"] != "":
        smiles = [request.form["drawSmiles"]]
    else:
        # Upload data file with SMILES
        data = request.files["data"]
        data_name = secure_filename(data.filename)

        with TemporaryDirectory() as temp_dir:
            data_path = str(Path(temp_dir) / data_name)
            data.save(data_path)

            # Check if header is smiles
            possible_smiles = get_header(data_path)[0]

            # TODO: if we're already computing RDKit molecules, try to make use of them later
            smiles = (
                [possible_smiles]
                if Chem.MolFromSmiles(possible_smiles) is not None
                else []
            )

            # Get remaining smiles
            smiles.extend(get_smiles(data_path))

    # Error if too many molecules
    if (
        app.config["MAX_MOLECULES"] is not None
        and len(smiles) > app.config["MAX_MOLECULES"]
    ):
        return render(
            errors=[
                f"Received too many molecules. "
                f"Maximum number of molecules is {app.config['MAX_MOLECULES']:,}."
            ]
        )

    # TODO: validate that SMILES are valid, remove invalid ones, and put a warning if there are invalid ones

    # Make predictions
    task_names, preds = predict_all_models(smiles=smiles)
    num_tasks = len(task_names)

    # TODO: Display physicochemical properties (and compare to DrugBank)
    physchem_names, physchem_preds = compute_physicochemical_properties(smiles=smiles)

    # Compute DrugBank percentiles
    preds_numpy = np.array(preds).transpose()  # (num_tasks, num_molecules)
    drugbank_percentiles = np.stack(
        [
            compute_drugbank_percentile(
                task_name=task_name,
                predictions=task_preds,
                atc_code=session.get("atc_code"),
            )
            for task_name, task_preds in zip(task_names, preds_numpy)
        ]
    ).transpose()  # (num_molecules, num_tasks)

    # Convert predictions to list of dicts
    preds_dicts = []
    for smiles_index, smile in enumerate(smiles):
        preds_dict = {"smiles": smile}

        for task_index, task_name in enumerate(task_names):
            preds_dict[task_name] = preds[smiles_index][task_index]
            preds_dict[
                f"{task_name}_drugbank_approved_percentile"
            ] = drugbank_percentiles[smiles_index][task_index]

        preds_dicts.append(preds_dict)

    # Convert predictions to DataFrame
    preds_df = pd.DataFrame(preds_dicts)

    # Store predictions in memory
    # TODO: figure out how to remove predictions from memory once no longer needed (i.e., once session ends)
    USER_TO_PREDS[session["user_id"]] = preds_df

    # Handle invalid SMILES
    if all(p is None for p in preds):
        return render(errors=["All SMILES are invalid"])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [
        pred if pred is not None else [invalid_smiles_warning] * num_tasks
        for pred in preds
    ]

    # Create DrugBank reference plot
    drugbank_plot_svg = plot_drugbank_reference(
        preds_df=USER_TO_PREDS[session["user_id"]],
        x_task=session.get("drugbank_x_task"),
        y_task=session.get("drugbank_y_task"),
        atc_code=session.get("atc_code"),
    )

    # TODO: refactor this and move this logic and data loader elsewhere and store in memory
    absorption_data = pd.read_csv(app.config["ADMET_DIR"] / "absorption_data.csv")
    distribution_data = pd.read_csv(app.config["ADMET_DIR"] / "distribution_data.csv")
    metabolism_data = pd.read_csv(app.config["ADMET_DIR"] / "metabolism_data.csv")
    excretion_data = pd.read_csv(app.config["ADMET_DIR"] / "excretion_data.csv")
    toxicity_data = pd.read_csv(app.config["ADMET_DIR"] / "toxicity_data.csv")

    return render(
        predicted=True,
        smiles=smiles,
        num_smiles=min(10, len(smiles)),
        show_more=max(0, len(smiles) - 10),
        task_names=task_names,
        num_tasks=num_tasks,
        cat=absorption_data,
        dist=distribution_data,
        meta=metabolism_data,
        excr=excretion_data,
        tox=toxicity_data,
        preds=preds,
        drugbank_percentiles=drugbank_percentiles,
        drugbank_plot=drugbank_plot_svg,
        warnings=["List contains invalid SMILES strings"] if None in preds else None,
        errors=["No SMILES strings given"] if len(preds) == 0 else None,
    )


@app.route("/drugbank_plot", methods=["GET"])
def drugbank_plot():
    # Get requested ATC code
    session["atc_code"] = request.args.get(
        "atc_code", default=session.get("atc_code"), type=str
    )

    # Get requested X and Y axes
    session["drugbank_x_task"] = request.args.get(
        "x_task", default=session.get("drugbank_x_task"), type=str
    )
    session["drugbank_y_task"] = request.args.get(
        "y_task", default=session.get("drugbank_y_task"), type=str
    )

    # Create DrugBank reference plot with ATC code
    drugbank_plot_svg = plot_drugbank_reference(
        preds_df=USER_TO_PREDS.get(session["user_id"], pd.DataFrame()),
        x_task=session["drugbank_x_task"],
        y_task=session["drugbank_y_task"],
        atc_code=session["atc_code"],
    )

    return jsonify({"svg": drugbank_plot_svg})


@app.route("/download_predictions")
def download_predictions() -> Response:
    """Downloads predictions as a CSV file."""
    # Create a temporary file to hold the predictions
    preds_file = NamedTemporaryFile()

    # Set up a function to close the file after the response is sent
    @after_this_request
    def remove_file(response: Response) -> Response:
        preds_file.close()
        return response

    # Save predictions to temporary file
    USER_TO_PREDS.get(session["user_id"], pd.DataFrame()).to_csv(
        preds_file.name, index=False
    )
    preds_file.seek(0)

    # Return the temporary file as a response
    return send_file(
        preds_file.name, as_attachment=True, download_name="predictions.csv"
    )
