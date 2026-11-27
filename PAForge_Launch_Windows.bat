@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -c "import paforge,gradio" >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -e ".[ui]" || goto :fail
)
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_OFFLINE=1
echo Open http://127.0.0.1:7860 in your browser
".venv\Scripts\python.exe" -m paforge.cli ui --port 7860
goto :eof
:fail
echo PAForge launcher failed. Check Python version and internet availability.
pause
exit /b 1
