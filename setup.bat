@echo off
setlocal
cd /d "%~dp0"
title reo setup
echo ==============================
echo         reo setup
echo ==============================
echo.

rem --- Find Python: newest first, since the pins were resolved against 3.14 ---
set "PY="
for %%v in (3.14 3.13 3.12 3.11) do (
  if not defined PY py -%%v --version >nul 2>&1 && set "PY=py -%%v"
)
if not defined PY (
  python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python is not installed.
  echo Opening the download page. Install Python and TICK "Add python.exe to PATH".
  echo Then run setup.bat again.
  start https://www.python.org/downloads/release/python-3128/
  goto :end
)
echo Using: %PY%
%PY% --version
echo.

rem --- Virtual environment ---
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv || goto :fail
) else (
  echo [1/3] Virtual environment already exists.
)

rem --- Packages ---
echo [2/3] Installing packages - this can take a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
echo.

rem --- API key ---
rem Handed to set-key.ps1 on purpose. "set /p" echoes the key on screen,
rem leaves it in this window's history and puts it in an environment
rem variable; set-key.ps1 reads it as a secure string and types it straight
rem into .env.
if not exist ".env" (
  echo [3/3] Setting up your free Groq API key.
  echo Opening console.groq.com/keys - sign in, click "Create API Key", copy it.
  start https://console.groq.com/keys
  powershell -NoProfile -ExecutionPolicy Bypass -File set-key.ps1
) else (
  echo [3/3] .env already exists, keeping your key.
)

echo.
echo Setup complete! Starting reo...
echo.
call run.bat
goto :eof

:fail
echo.
echo Setup failed. The first red line above is the one that matters.
:end
pause
