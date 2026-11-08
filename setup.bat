@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo idol.cpp setup
echo ========================================================

cd /d "%~dp0"

:: 1. Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: 2. Check Python version (>= 3.10)
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10 or newer is required.
    python --version
    pause
    exit /b 1
)

:: 3. Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Existing virtual environment found (.venv).
)

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"

:: 4. Upgrade pip
echo [2/4] Upgrading pip...
"%VENV_PYTHON%" -m pip install --quiet --upgrade pip

:: 5. Install PyTorch with CUDA or CPU
echo [3/4] Checking hardware and installing PyTorch...
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected via nvidia-smi. Installing PyTorch with CUDA support...
    "%VENV_PIP%" install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple || "%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
) else (
    echo No NVIDIA GPU detected. Installing CPU PyTorch...
    "%VENV_PIP%" install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
)

:: 6. Install idol-sdk and runtime dependencies
echo [4/4] Installing idol-sdk and runtime dependencies...
if exist "idol-sdk\setup.py" (
    echo Found local idol-sdk directory, installing from local copy...
    "%VENV_PIP%" install ./idol-sdk -r requirements-runtime.txt -r requirements-server.txt
) else (
    echo Installing pinned idol-sdk from GitHub...
    "%VENV_PIP%" install "git+https://github.com/speridlabs/idol-sdk.git@4702a7f00850e34b6e83a2170cc123c3e9570744" -r requirements-runtime.txt -r requirements-server.txt
)

:: 7. Check prebuilt native binaries
echo.
set "BIN_FOUND=0"
if exist "idol_mesh.exe" set "BIN_FOUND=1"
if exist "bin\idol_mesh.exe" set "BIN_FOUND=1"
if exist "build\idol_mesh.exe" set "BIN_FOUND=1"
if exist "build\Release\idol_mesh.exe" set "BIN_FOUND=1"

if "%BIN_FOUND%"=="1" (
    echo Prebuilt native binaries detected (idol_mesh.exe).
) else (
    echo [NOTE] Native idol_mesh.exe was not found in root, bin/, or build/.
    echo If you downloaded source without prebuilt binaries, build them with:
    echo   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    echo   cmake --build build --config Release --parallel
)

echo.
echo ========================================================
echo Setup complete!
echo To start the server, run: run.bat
echo ========================================================
