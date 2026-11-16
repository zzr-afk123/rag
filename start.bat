@echo off
setlocal enableextensions
cd /d "%~dp0"
title RAG Assistant
color 0F

rem Usage:
rem   start.bat                     local only (127.0.0.1)
rem   start.bat --lan               bind 0.0.0.0, phones on same Wi-Fi can connect
rem   start.bat --port 8080         change port
rem   start.bat --no-browser        do not open a browser
set "HOST=127.0.0.1"
set "PORT=8000"
set "LANMODE=0"
set "OPENBROWSER=1"
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--lan" (
    set "HOST=0.0.0.0"
    set "LANMODE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--server" (
    set "HOST=0.0.0.0"
    set "LANMODE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--no-browser" (
    set "OPENBROWSER=0"
    shift
    goto parse_args
)
shift
goto parse_args
:args_done

echo ==========================================================
echo   RAG Assistant  -  one click start
echo ==========================================================
echo.

rem ---------- python ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python not found in PATH.
    echo     Install Python 3.10+ from python.org and tick "Add to PATH".
    pause
    exit /b 1
)

rem ---------- venv + deps ----------
if not exist ".venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment ...
    python -m venv .venv
    if errorlevel 1 (
        echo [X] Failed to create venv.
        pause
        exit /b 1
    )
    echo [2/6] Installing dependencies - first run takes a few minutes ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [X] pip install failed. See the error above.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install -e . >nul
) else (
    echo [1/6] Virtual environment found.
)

set "PY=.venv\Scripts\python.exe"
set "HFCACHE=%USERPROFILE%\.cache\huggingface\hub"
set "PYTHONPATH=%CD%\src"

rem ---------- mode ----------
set "MODE=MOCK"
"%PY%" -c "import sentence_transformers" >nul 2>nul
if not errorlevel 1 (
    if exist "%HFCACHE%\models--BAAI--bge-m3" set "MODE=REAL"
)
if "%MODE%"=="MOCK" (
    echo [2/6] Mode: MOCK  ^(no sentence-transformers or no bge-m3^)
    set "RAG_FORCE_MOCK=1"
) else (
    echo [2/6] Mode: REAL embedding ^(BAAI/bge-m3^)
    set "RAG_FORCE_MOCK="
    if not exist "%HFCACHE%\models--BAAI--bge-reranker-v2-m3" (
        echo       reranker model absent -^> reranker disabled
        set "RAG_RERANKER__ENABLED=false"
    )
)
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_XET=1"
set "PYTHONIOENCODING=utf-8"

rem ---------- pick a free port ----------
set "PICKED="
for /f "usebackq delims=" %%p in (`"%PY%" scripts\port_pick.py --host %HOST% --port %PORT%`) do set "PICKED=%%p"
if not "%PICKED%"=="" if not "%PICKED%"=="%PORT%" (
    echo [3/6] Port %PORT% is busy -^> using %PICKED%
    set "PORT=%PICKED%"
)
if "%PICKED%"=="%PORT%" echo [3/6] Port %PORT% is free.
set "RAG_SERVER__HOST=%HOST%"
set "RAG_SERVER__PORT=%PORT%"

rem ---------- login accounts ----------
if not exist "config\users.yaml" (
    echo Creating demo accounts ^(random passwords, see logs\credentials.txt^) ...
    "%PY%" scripts\manage_users.py init
)

rem ---------- index ----------
if not exist "data\index\manifest.json" (
    echo [4/6] Building index from data\raw ...
    "%PY%" scripts\sync.py --rebuild
    if errorlevel 1 echo       index build reported failures - see log above
) else (
    echo [4/6] Index found -^> incremental sync ...
    "%PY%" scripts\sync.py
)

rem ---------- urls ----------
echo.
echo [5/6] Reachable URLs:
"%PY%" scripts\show_urls.py --host %HOST% --port %PORT%
if "%LANMODE%"=="1" (
    echo       LAN mode: phones on the same Wi-Fi can open the URLs above.
    echo       If a phone cannot connect, allow the port in Windows Firewall:
    echo         netsh advfirewall firewall add rule name="RAG %PORT%" dir=in action=allow protocol=TCP localport=%PORT%
    echo       Security: no login. Anyone on your LAN can query it.
) else (
    echo       Local mode: this machine only. Use start-server.bat for LAN/mobile.
)

rem ---------- serve ----------
echo.
echo [6/6] Starting server on port %PORT% ...
echo       API docs: http://127.0.0.1:%PORT%/docs
echo       Browser will open automatically once /health is ready.
echo       Press Ctrl+C to stop.
echo.
if "%OPENBROWSER%"=="1" start "" /b "%PY%" scripts\open_when_ready.py --port %PORT% --timeout 180
"%PY%" -m uvicorn rag_assistant.api.main:app --host %HOST% --port %PORT% --workers 1

echo.
echo Server exited. If you saw an error above, run doctor.bat for a checklist.
pause
