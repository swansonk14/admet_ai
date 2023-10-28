"""Runs the web interface version of ADMET-AI using Flask."""
from flask import Flask

app = Flask(__name__)
app.config.from_object("admet_ai.web.app.config")

from admet_ai.web.app import views
