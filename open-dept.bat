@echo off
rem Open the department dashboard (/dept).
rem Usage: open-dept.bat           (port 8000)
rem        open-dept.bat 8080
setlocal
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"
echo Opening http://127.0.0.1:%PORT%/dept ...
start "" "http://127.0.0.1:%PORT%/dept"
endlocal
