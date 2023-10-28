"""Runs the production web interface for ADMET-AI (Gunicorn)."""
from flask import Flask

from admet_ai.web.app import app
from admet_ai.web.run import setup_web


def build_app(*args, **kwargs) -> Flask:
    setup_web(**kwargs)

    return app
