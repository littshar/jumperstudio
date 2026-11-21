@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: setting up Jumper Studio...
    call "%~dp0setup_windows.bat"
    if errorlevel 1 (
        echo.
        echo Setup failed. See setup_windows.log.
        pause
        exit /b 1
    )
)

start "" ".venv\Scripts\pythonw.exe" "run.py"
exit /b 0
