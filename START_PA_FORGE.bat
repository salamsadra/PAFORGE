@echo off
cd /d "%~dp0"
py setup_and_launch.py
if errorlevel 1 (
  echo.
  echo PA-FORGE could not start. Read the error above.
  pause
)
