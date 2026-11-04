@echo off
setlocal
cd /d "%~dp0"
title reo setup
echo ==============================
echo         reo setup
echo ==============================
echo.

rem --- Find Python 3.12 / 3.11 ---
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
  py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
)
if not defined PY (
  python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python is not installed.
  echo Opening the download page. Install Python 3.12 and TICK "Add python.exe to PATH".
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
if not exist ".env" (
  echo [3/3] Setting up your free Groq API key.
  echo Opening console.groq.com/keys - sign in, click "Create API Key", copy it.
  start https://console.groq.com/keys
  copy .env.example .env >nul
  set /p "KEY=Paste your Groq API key here and press Enter: "
  powershell -NoProfile -Command "(Get-Content .env) -replace 'your-groq-key-here', $env:KEY | Set-Content .env"
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
echo Setup failed. Copy the red error text above and send it to Claude.
:end
pause
