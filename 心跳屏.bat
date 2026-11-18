@echo off
chcp 65001 >nul
setlocal
set "PY=C:\Users\ZZR\Desktop\rag-assistant-v3\rag-assistant\.venv\Scripts\python.exe"
set "ROOT=%~dp0.."
pushd "%ROOT%"
echo ==== RAG heartbeat monitor ====
echo It reads http://127.0.0.1:8000/health and pushes one line to the board
echo over Bluetooth (device name RAG-HB). Close this window to stop.
echo.
"%PY%" scripts\heartbeat_server.py --url http://127.0.0.1:8000 %*
echo.
echo [stopped] exit code %ERRORLEVEL%
pause
