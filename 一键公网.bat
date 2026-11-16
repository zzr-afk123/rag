@echo off
chcp 936 >nul
setlocal
cd /d "%~dp0"
title RAG Assistant - public demo (close this window to stop)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\public_tunnel.ps1" %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo.
  echo [exit code %CODE%] press any key to close...
  pause >nul
)
endlocal
