"""Runs the development web interface for ADMET-AI (Flask)."""
from datetime import timedelta
from threading import Thread

from tap import Tap

from admet_ai.web.app import app
from admet_ai.web.app.admet_info import load_admet_info
from admet_ai.web.app.drugbank import load_drugbank
from admet_ai.web.app.models import load_admet_model
from admet_ai.web.app.storage import cleanup_storage


class WebArgs(Tap):
    host: str = "127.0.0.1"
    """Host IP address."""
    port: int = 5000
    """Host port."""
    secret_key: str = "f*3^iWiue*maS35MgYAJ"
    """Secret key for Flask app. (TODO: do not use this default secret key in production.)"""
    session_lifetime: int = 5 * 60
    """Session lifetime in seconds."""
    heartbeat_frequency: int = 60
    """Frequency of client heartbeat in seconds."""
    max_molecules: int | None = None
    """Maximum number of molecules to allow predictions for."""
    no_cache_molecules: bool = False
    """Whether to turn off molecule caching (reduces memory but slows down predictions)."""


def setup_web(
    secret_key: str,
    session_lifetime: int = 5 * 60,
    heartbeat_frequency: int = 60,
    max_molecules: int | None = None,
    no_cache_molecules: bool = False,
) -> None:
    """Sets up the ADMET-AI website.

    :param secret_key: Secret key for Flask app.
    :param session_lifetime: Session lifetime in seconds.
    :param heartbeat_frequency: Frequency of client heartbeat in seconds.
    :param max_molecules: Maximum number of molecules to allow predictions for.
    :param no_cache_molecules: Whether to turn off molecule caching (reduces memory but slows down predictions).
    """
    # Set up Flask app variables
    app.secret_key = secret_key
    app.permanent_session_lifetime = timedelta(seconds=session_lifetime)
    app.config["SESSION_LIFETIME"] = session_lifetime
    app.config["HEARTBEAT_FREQUENCY"] = heartbeat_frequency
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

    # Set up garbage collection
    Thread(target=cleanup_storage).start()


def admet_web() -> None:
    """Runs the ADMET-AI website locally."""
    # Parse arguments
    args = WebArgs().parse_args()

    # Set up web app
    setup_web(
        secret_key=args.secret_key,
        session_lifetime=args.session_lifetime,
        heartbeat_frequency=args.heartbeat_frequency,
        max_molecules=args.max_molecules,
        no_cache_molecules=args.no_cache_molecules,
    )

    # Run web app
    app.run(host=args.host, port=args.port)
