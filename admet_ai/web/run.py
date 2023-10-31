"""Runs the development web interface for ADMET-AI (Flask)."""
from chemprop.data import set_cache_graph, set_cache_mol
from tap import Tap

from admet_ai.web.app import app
from admet_ai.web.app.admet_info import load_admet_info
from admet_ai.web.app.drugbank import load_drugbank
from admet_ai.web.app.models import load_admet_model


class WebArgs(Tap):
    host: str = "127.0.0.1"  # Host IP address
    port: int = 5000  # Host port
    secret_key: str = "f*3^iWiue*maS35MgYAJ"  # Secret key for Flask app (TODO: do not use this default secret key in production)
    max_molecules: int | None = None  # Maximum number of molecules to allow predictions for
    no_cache_molecules: bool = False  # Whether to turn off molecule caching (reduces memory but slows down predictions)


def setup_web(
    secret_key: str,
    max_molecules: int | None = None,
    no_cache_molecules: bool = False,
) -> None:
    app.secret_key = secret_key
    app.config["MAX_MOLECULES"] = max_molecules
    app.config["CACHE_MOLECULES"] = not no_cache_molecules

    # Load ADMET info, DrugBank, and models into memory
    with app.app_context():
        print("Loading ADMET INFO")
        load_admet_info()

        print("Loading DrugBank")
        load_drugbank()

        print("Loading models")
        load_admet_model()


def admet_web() -> None:
    """Runs the ADMET-AI website locally."""
    # Parse arguments
    args = WebArgs().parse_args()

    # Set up web app
    setup_web(
        secret_key=args.secret_key,
        max_molecules=args.max_molecules,
        no_cache_molecules=args.no_cache_molecules,
    )

    # Run web app
    app.run(host=args.host, port=args.port)
