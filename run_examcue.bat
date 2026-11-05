@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\pythonw.exe"

if not exist "%PYTHON_EXE%" (
  echo ExamCue 尚未安裝。請先在 PowerShell 執行：
  echo powershell -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\setup.ps1"
  pause
  exit /b 1
)

start "" "%PYTHON_EXE%" -m examcue
endlocal

