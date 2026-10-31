#!/bin/bash
# macOS launcher (double-click). Keep this Terminal window open while captions run.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Setup is missing. In Terminal run: bash setup-mac.sh"
    read -r -p "Press Enter to close."
    exit 1
fi
exec .venv/bin/python run_app.py --api
