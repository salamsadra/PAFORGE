"""One-click local Windows/macOS/Linux bootstrap for the PA-FORGE dashboard."""
from pathlib import Path
import importlib.util, subprocess, sys
ROOT=Path(__file__).resolve().parent

def run(cmd):
    print('\n>',' '.join(map(str,cmd)))
    return subprocess.run(cmd,cwd=ROOT).returncode

if importlib.util.find_spec('gradio') is None:
    print('Gradio is missing. Installing the dashboard dependency...')
    code=run([sys.executable,'-m','pip','install','gradio>=5'])
    if code:raise SystemExit(code)
# Install the local package in editable mode so the CLI and dashboard point here.
code=run([sys.executable,'-m','pip','install','-e',str(ROOT)])
if code:raise SystemExit(code)
print('\nStarting PA-FORGE. Open http://127.0.0.1:7860 if it is not opened automatically.')
raise SystemExit(run([sys.executable,'-m','paforge.cli','ui']))
