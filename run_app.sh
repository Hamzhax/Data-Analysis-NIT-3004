#!/bin/bash
# Script to run the Flask app with the virtual environment activated

cd "$(dirname "$0")"
source .venv/bin/activate
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
