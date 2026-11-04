@echo off
setlocal
cd /d "%~dp0"
title reo - start on boot
set "REO_HOME=%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo reo is not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)

powershell -NoProfile -Command ^
  "$link = Join-Path ([Environment]::GetFolderPath('Startup')) 'reo.lnk';" ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($link);" ^
  "$s.TargetPath = Join-Path $env:REO_HOME 'run.bat';" ^
  "$s.WorkingDirectory = $env:REO_HOME;" ^
  "$s.WindowStyle = 7;" ^
  "$s.Description = 'reo voice assistant';" ^
  "$s.Save();" ^
  "Write-Host ('Installed: ' + $link)"

echo.
echo reo will now start minimized when you sign in.
echo Run uninstall-startup.bat to stop that.
pause
