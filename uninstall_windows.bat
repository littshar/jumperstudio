@echo off
setlocal
rem One-click uninstall for Jumper Studio.
rem Select parts with:
rem   uninstall_windows.bat -Components jumper,shortcut -Yes
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_windows.ps1" %*
set RC=%ERRORLEVEL%
echo.
pause
exit /b %RC%
