#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Python 3.10+ is required.' >&2; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo '[PAForge] Creating local virtual environment (first launch)...'
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import paforge,gradio' >/dev/null 2>&1; then
  echo '[PAForge] Installing package and UI dependencies (first launch may require network)...'
  .venv/bin/python -m pip install -e '.[ui]'
fi
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
printf '\n[PAForge] Opening local dashboard at http://127.0.0.1:7860\n'
exec .venv/bin/python -m paforge.cli ui --port 7860
