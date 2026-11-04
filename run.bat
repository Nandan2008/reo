@echo off
cd /d "%~dp0"
title reo
if not exist ".venv\Scripts\python.exe" (
  echo reo is not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m reo
pause
