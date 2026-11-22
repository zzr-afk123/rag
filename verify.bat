@echo off
rem ---------------------------------------------------------------
rem  Reviewer quick check: build venv, init demo accounts,
rem  run the offline test suite, then offer to start the service.
rem  Everything printed here is ASCII on purpose (the Windows
rem  console on some machines double-draws CJK characters when a
rem  child process writes at the same time).
rem ---------------------------------------------------------------
setlocal
chcp 65001 > nul
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "FAIL=0"

echo ============================================================
echo  RAG assistant - reviewer check
echo ============================================================
echo.

rem --- 1/4 python interpreter -----------------------------------
where py >nul 2>nul
if %errorlevel%==0 (
    set "BOOT=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "BOOT=python"
    ) else (
        echo [X] Python 3.10+ not found. Install it from python.org first.
        goto :done
    )
)

rem --- 2/4 virtualenv + dependencies ----------------------------
if not exist "%PY%" (
    echo [1/4] Creating .venv ...
    %BOOT% -m venv "%VENV%"
    if not exist "%PY%" (
        echo [X] Failed to create .venv
        goto :done
    )
    echo [1/4] Installing dependencies ^(first run, may take a few minutes^) ...
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r requirements-dev.txt --quiet
) else (
    echo [1/4] .venv found, reusing it.
)

rem --- 3/4 demo accounts ---------------------------------------
"%PY%" -c "import pytest" >nul 2>nul
if errorlevel 1 (
    echo [2/4] Installing test dependencies ^(requirements-dev^) ...
    "%PY%" -m pip install -r requirements-dev.txt --quiet
)
if not exist "config\users.yaml" (
    echo [2/4] Creating demo accounts ^(weak passwords, local demo only^) ...
    "%PY%" scripts\manage_users.py init --weak-demo-passwords
) else (
    echo [2/4] config\users.yaml found, keeping existing accounts.
)

rem --- 4/4 offline test suite ----------------------------------
echo [3/4] Running the offline test suite ^(RAG_FORCE_MOCK=1, no network needed^) ...
echo       Expect roughly 3 minutes and a green summary line.
set "RAG_FORCE_MOCK=1"
rem 注意：pyproject 里已有 addopts="-q"，这里不要再加 -q（-qq 会把汇总行吞掉）
"%PY%" -m pytest
if errorlevel 1 (
    echo [!] Some tests failed. Scroll up for the first failure.
    set "FAIL=1"
) else (
    echo [3/4] All tests passed.
)

echo.
echo [4/4] Optional: start the service so you can click through the UI.
echo       Note: without an API key or a local model the page runs in
echo       demo mode ^(answers are rule-based, banner shown on top^).
echo.
choice /c YN /n /m "Start the service now? [Y/N] "
if errorlevel 2 goto :summary
call "%~dp0start.bat"
goto :done

:summary
echo.
if "%FAIL%"=="0" (
    echo [OK] Environment verified. Run start.bat whenever you want the UI.
) else (
    echo [!] Environment verified, but tests failed - see the log above.
)

:done
echo.
echo Press any key to close this window...
pause > nul
endlocal
