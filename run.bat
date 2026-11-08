@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Running setup.bat first...
    call "%~dp0setup.bat"
    if errorlevel 1 (
        echo [ERROR] Setup failed.
        pause
        exit /b 1
    )
)

echo Starting idol.cpp local server...
echo Web UI will be available at http://127.0.0.1:8127
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\idol_server.py" --repo TimeLordRaps/Idol-GGUF --file Idol-Q8_0.gguf %*

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with code %errorlevel%.
    pause
)
