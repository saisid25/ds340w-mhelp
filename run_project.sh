#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Python is not installed or is not available in PATH."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  "$PYTHON_CMD" -m venv venv
fi

echo "Activating virtual environment..."
. venv/bin/activate

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Running enhanced experiment..."
python enhanced_mhelp_experiment.py
