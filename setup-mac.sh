#!/bin/bash
# macOS Realtime-only setup. System sound is captured through BlackHole 2ch.
# Run once in Terminal: bash setup-mac.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for candidate in python3.12 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
                 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$(command -v "$candidate")"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python 3.12 is required: https://www.python.org/downloads/macos/ (then run this again)"
    exit 1
fi
if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-light.txt
.venv/bin/python -m pip check
chmod +x "Start API Captions.command" setup-mac.sh
echo "Ready (Realtime only). Install BlackHole 2ch and select a Multi-Output Device (speakers + BlackHole 2ch) as sound output,"
echo "then double-click \"Start API Captions.command\" and enter your OpenAI API key in the app."
