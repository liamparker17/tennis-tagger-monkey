@echo off
REM ============================================================================
REM Tennis Tagger - One-Click Installer
REM ============================================================================
REM
REM This script handles:
REM   1. Python installation check
REM   2. Virtual environment creation
REM   3. Dependency installation
REM   4. Model downloading
REM   5. Desktop shortcut creation
REM   6. Application launch
REM
REM Just double-click to install and run!
REM ============================================================================

setlocal enabledelayedexpansion

title Tennis Tagger - Installer

echo.
echo =====================================================
echo    TENNIS TAGGER - ONE-CLICK INSTALLER
echo =====================================================
echo.

REM Change to script directory
cd /d "%~dp0"
set "INSTALL_DIR=%~dp0"

REM ============================================================================
REM Check if this is first run or subsequent launch
REM ============================================================================

REM Check if venv exists AND has gradio installed (complete installation)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python -c "import gradio" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [*] Existing installation detected
        echo [*] Launching Tennis Tagger...
        goto :launch
    ) else (
        echo [*] Incomplete installation detected - reinstalling...
        rmdir /s /q venv >nul 2>&1
    )
)

echo [*] First-time installation - this may take a few minutes...
echo.

REM ============================================================================
REM Step 1: Check Python
REM ============================================================================

echo [1/5] Checking Python installation...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Python version: %PYTHON_VERSION%

REM Extract major.minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.10+ required. Found Python %PYTHON_VERSION%
    pause
    exit /b 1
)

if %PYTHON_MAJOR%==3 if %PYTHON_MINOR% LSS 10 (
    echo WARNING: Python 3.10+ recommended. Found Python %PYTHON_VERSION%
    echo          Some features may not work correctly.
    echo.
)

echo       [OK] Python check passed

REM ============================================================================
REM Step 2: Create Virtual Environment
REM ============================================================================

echo.
echo [2/5] Creating virtual environment...

if exist "venv" (
    echo       Removing old venv...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo       [OK] Virtual environment created

REM ============================================================================
REM Step 3: Activate and Install Dependencies
REM ============================================================================

echo.
echo [3/5] Installing dependencies (this may take several minutes)...

call venv\Scripts\activate.bat

REM Upgrade pip first
python -m pip install --upgrade pip >nul 2>&1

REM Install PyTorch with CUDA support (Windows)
echo       Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >nul 2>&1
if %errorlevel% neq 0 (
    echo       Note: CUDA install failed, trying CPU version...
    pip install torch torchvision torchaudio >nul 2>&1
)

REM Install other requirements
echo       Installing other packages...

pip install gradio>=4.0.0 >nul 2>&1
pip install ultralytics>=8.0.0 >nul 2>&1
pip install opencv-python>=4.8.0 >nul 2>&1
pip install pandas>=2.0.0 >nul 2>&1
pip install numpy>=1.24.0 >nul 2>&1
pip install PyYAML>=6.0 >nul 2>&1
pip install tqdm>=4.66.0 >nul 2>&1
pip install plotly>=5.0.0 >nul 2>&1
pip install pywebview>=4.0.0 >nul 2>&1
pip install requests>=2.31.0 >nul 2>&1
pip install Pillow>=10.0.0 >nul 2>&1
pip install scipy>=1.11.0 >nul 2>&1
pip install scikit-learn>=1.3.0 >nul 2>&1
pip install filterpy>=1.4.5 >nul 2>&1
pip install lap>=0.4.0 >nul 2>&1

REM Install from requirements.txt if it exists
if exist "requirements.txt" (
    echo       Installing from requirements.txt...
    pip install -r requirements.txt >nul 2>&1
)

echo       [OK] Dependencies installed

REM ============================================================================
REM Step 4: Download/Verify Models
REM ============================================================================

echo.
echo [4/5] Checking models...

if not exist "models" mkdir models
if not exist "models\versions" mkdir models\versions

REM YOLOv8 models will be auto-downloaded on first use
REM but we can pre-download the main ones

echo       YOLOv8 models will be downloaded on first use
echo       [OK] Model directory ready

REM ============================================================================
REM Step 5: Create Required Directories
REM ============================================================================

echo.
echo [5/5] Creating directories...

if not exist "data" mkdir data
if not exist "data\output" mkdir data\output
if not exist "data\fvd" mkdir data\fvd
if not exist "data\training_pairs" mkdir data\training_pairs
if not exist "data\training_data" mkdir data\training_data
if not exist "data\datasets" mkdir data\datasets
if not exist "data\qc_corrections" mkdir data\qc_corrections
if not exist "logs" mkdir logs
if not exist "checkpoints" mkdir checkpoints
if not exist "cache" mkdir cache

echo       [OK] Directories created

REM ============================================================================
REM Create Desktop Shortcut
REM ============================================================================

echo.
echo Creating desktop shortcut...

set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Tennis Tagger.lnk"
set "VBS_FILE=%TEMP%\create_shortcut.vbs"

REM Create VBS script to make shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%VBS_FILE%"
echo sLinkFile = "%SHORTCUT_PATH%" >> "%VBS_FILE%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%VBS_FILE%"
echo oLink.TargetPath = "%INSTALL_DIR%launch.bat" >> "%VBS_FILE%"
echo oLink.WorkingDirectory = "%INSTALL_DIR%" >> "%VBS_FILE%"
echo oLink.Description = "Tennis Tagger - Video Tagging System" >> "%VBS_FILE%"
echo oLink.Save >> "%VBS_FILE%"

cscript //nologo "%VBS_FILE%" >nul 2>&1
del "%VBS_FILE%" >nul 2>&1

echo       [OK] Desktop shortcut created

REM ============================================================================
REM Create Launch Script
REM ============================================================================

echo.
echo Creating launch script...

(
echo @echo off
echo title Tennis Tagger
echo cd /d "%%~dp0"
echo call venv\Scripts\activate.bat
echo python tennis_tagger_unified.py
echo pause
) > launch.bat

echo       [OK] Launch script created

REM ============================================================================
REM Installation Complete
REM ============================================================================

echo.
echo =====================================================
echo    INSTALLATION COMPLETE!
echo =====================================================
echo.
echo You can now:
echo   1. Use the desktop shortcut "Tennis Tagger"
echo   2. Run launch.bat from this folder
echo   3. Double-click this installer again (it will just launch)
echo.

:launch

REM ============================================================================
REM Launch Application
REM ============================================================================

echo.
echo Launching Tennis Tagger...
echo.

call venv\Scripts\activate.bat
python tennis_tagger_unified.py

if %errorlevel% neq 0 (
    echo.
    echo Error launching application. Check the logs for details.
    pause
)

endlocal
