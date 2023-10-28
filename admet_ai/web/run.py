"""Runs the development web interface for ADMET-AI (Flask)."""
from chemprop.data import set_cache_graph, set_cache_mol
from tap import Tap

from admet_ai.web.app import app
from admet_ai.web.app.data import load_admet_info
from admet_ai.web.app.drugbank import load_drugbank
from admet_ai.web.app.models import load_models


class WebArgs(Tap):
    host: str = "127.0.0.1"  # Host IP address
    port: int = 5000  # Host port
    secret_key: str = "f*3^iWiue*maS35MgYAJ"  # Secret key for Flask app (TODO: do not use this default secret key in production)
    max_molecules: int | None = None  # Maximum number of molecules to allow predictions for
    no_cache: bool = False  # Whether to turn off molecule caching (reduces memory but slows down predictions)


def setup_web(
    secret_key: str,
    max_molecules: int | None = None,
    no_cache: bool = False,
) -> None:
    app.secret_key = secret_key
    app.config["MAX_MOLECULES"] = max_molecules

    # Turn off caching to save memory (at the cost of speed when using ensembles)
    if no_cache:
        set_cache_graph(False)
        set_cache_mol(False)

    # Load ADMET info, DrugBank, and models into memory
    with app.app_context():
        load_admet_info()
        load_drugbank()
        load_models()


def admet_web() -> None:
    """Runs the ADMET-AI website locally."""
    # Parse arguments
    args = WebArgs().parse_args()

    # Set up web app
    setup_web(
        secret_key=args.secret_key,
        max_molecules=args.max_molecules,
        no_cache=args.no_cache,
    )

    # Run web app
    app.run(host=args.host, port=args.port)
