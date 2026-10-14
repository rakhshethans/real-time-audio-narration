@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call Install-and-Run.bat
    exit /b
)
".venv\Scripts\python.exe" -m audio_narration.main
if errorlevel 1 pause
