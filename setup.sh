#!/usr/bin/env bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt -r requirements-dev.txt

echo "Setup complete. Please run 'source venv/bin/activate' to activate the virtual environment before running the application."
