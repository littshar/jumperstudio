@echo off
setlocal
rem One-click setup: Jumper Studio app + PPO training environment.
rem Options are passed through, e.g.:
rem   setup_windows.bat -JumperDir D:\jumper_ws -TorchBuild cu126
rem   setup_windows.bat -Action diagnose
rem   setup_windows.bat -Action install -Components jumper,patch -FromApp
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
if errorlevel 1 (
    echo.
    echo Setup failed. See setup_windows.log for the full output.
    pause
    exit /b 1
)
echo.
echo Setup complete. Launch the app from the desktop shortcut.
pause
