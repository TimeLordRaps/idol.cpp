@echo off
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0tools\idol_server.py" %*
) else (
    python "%~dp0tools\idol_server.py" %*
)
