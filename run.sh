#!/usr/bin/env bash
# Advisor Sandbox launcher for Mac / Linux / Raspberry Pi.
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "Python 3.11+ is required (https://www.python.org/downloads/)."
  exit 1
fi
if [ ! -d .venv ]; then
  echo "First run: creating a virtual environment..."
  "$PY" -m venv .venv
fi
echo "Checking packages (the first run takes a minute or two)..."
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt

echo "Starting Advisor Sandbox at http://localhost:8501 (Ctrl+C to stop)"
( sleep 5; (command -v open >/dev/null && open http://localhost:8501) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8501) || true ) >/dev/null 2>&1 &
exec .venv/bin/python -m streamlit run app.py "$@"
