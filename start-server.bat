@echo off
rem LAN / mobile mode: bind 0.0.0.0 so other devices on the same network can connect.
call "%~dp0start.bat" --lan %*
