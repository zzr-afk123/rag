@echo off
rem Open the NEW UI with a cache-busting query string.
rem Usage: open-page.bat          (port 8000)
rem        open-page.bat 8080
setlocal
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
echo Opening http://127.0.0.1:%PORT%/?v=3 ...
start "" "http://127.0.0.1:%PORT%/?v=3"
endlocal
