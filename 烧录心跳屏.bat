@echo off
rem ---------------------------------------------------------------
rem  Flash the heartbeat screen firmware (ESP32-C3, MicroPython).
rem  Plug the board into USB first; the script finds the port by
rem  the Espressif USB id (VID 303A / PID 1001).
rem ---------------------------------------------------------------
setlocal
chcp 65001 > nul
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo  Heartbeat screen - flash firmware
echo ============================================================
echo.

"%PY%" scripts\upload_board.py %*
set "CODE=%ERRORLEVEL%"

echo.
if not "%CODE%"=="0" (
    echo [!] Upload failed. Is the USB cable a data cable? Is the port in use?
) else (
    echo [OK] Firmware flashed. The screen shows "v3" at the bottom right.
)
echo.
echo Press any key to close this window...
pause > nul
endlocal
