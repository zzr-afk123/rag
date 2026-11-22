@echo off
rem 中文名入口：转发到 verify.bat（同一套逻辑，避免两处维护）
call "%~dp0verify.bat" %*
