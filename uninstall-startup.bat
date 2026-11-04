@echo off
setlocal
title reo - remove from startup

powershell -NoProfile -Command ^
  "$link = Join-Path ([Environment]::GetFolderPath('Startup')) 'reo.lnk';" ^
  "if (Test-Path $link) { Remove-Item $link -Force; Write-Host ('Removed: ' + $link) }" ^
  "else { Write-Host 'reo was not set to start on boot.' }"

echo.
pause
