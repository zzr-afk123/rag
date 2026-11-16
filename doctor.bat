@echo off
cd /d "%~dp0"
title RAG Assistant - doctor
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"

echo Running self check with %PY% ...
"%PY%" scripts\doctor.py %*
echo.
pause
