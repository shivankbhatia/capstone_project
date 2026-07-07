#!/usr/bin/env bash
# One-time environment setup.
# Usage: bash setup_env.sh

set -e

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Done. Activate the environment in future sessions with:"
echo "  source .venv/bin/activate"
