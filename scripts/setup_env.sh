#!/bin/bash

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r alfworld_runs/requirements.txt

# Install alfworld specifically (if not fully covered or to ensure data download script is available)
# It is added to requirements.txt, so pip install -r should match it, but we explicit here for clarity if needed.

echo "Setup complete. To activate venv: source .venv/bin/activate"
