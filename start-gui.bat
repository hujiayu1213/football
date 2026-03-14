@echo off
setlocal
set "ROOT=%~dp0"
powershell.exe -ExecutionPolicy Bypass -File "%ROOT%scripts\start-gui.ps1" %*
endlocal

