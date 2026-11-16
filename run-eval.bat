@echo off
setlocal enableextensions
cd /d "%~dp0"
title RAG Assistant - real model evaluation
color 0F

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

set "PYTHONPATH=%CD%\src"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_XET=1"

echo ==========================================================
echo   Real model evaluation  (index rebuild + threshold sweep)
echo ==========================================================
echo.
echo   step 1: rebuild index with real bge-m3 (+ reranker if present)
echo   step 2: run data\eval\golden_set.jsonl  ->  reports\eval-*.md
echo   step 3: print the recommended score threshold
echo.
echo   On CPU this can take 5-20 minutes. Keep this window open.
echo   Extra args are passed through, e.g.:  run-eval.bat --label v2
echo.
timeout /t 3 >nul
echo.

"%PY%" -u scripts\real_eval.py %*
echo.
echo Done. See the reports folder for the markdown report.
pause
