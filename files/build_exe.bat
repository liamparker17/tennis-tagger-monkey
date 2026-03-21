@echo off
REM ============================================
REM Build Tennis Tagger as Standalone EXE
REM ============================================

color 0A
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   🎾 TENNIS TAGGER - BUILD EXECUTABLE                        ║
echo ║                                                              ║
echo ║   Creating standalone EXE files                             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
    echo.
) else (
    echo [!] No venv found - using system Python
    echo.
)

REM Install PyInstaller if not installed
echo [*] Checking PyInstaller...
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [!] PyInstaller not installed - installing now...
    echo.
    pip install pyinstaller
    echo.
    echo [OK] PyInstaller installed
    echo.
) else (
    echo [OK] PyInstaller already installed
    echo.
)

REM Install PyWebView if not installed
echo [*] Checking PyWebView...
python -c "import webview" >nul 2>&1
if errorlevel 1 (
    echo [!] PyWebView not installed - installing now...
    pip install pywebview
    echo [OK] PyWebView installed
    echo.
) else (
    echo [OK] PyWebView already installed
    echo.
)

echo ============================================
echo   BUILDING EXECUTABLES
echo ============================================
echo.
echo This will create 2 standalone EXE files:
echo  1. TennisTagger_Training.exe (Training System v3.1)
echo  2. TennisTagger_Tagging.exe (Video Tagging System)
echo.
echo This may take 5-10 minutes...
echo.

REM Create dist folder
if not exist "dist" mkdir dist

echo.
echo ============================================
echo   [1/2] Building Training EXE
echo ============================================
echo.

pyinstaller --noconfirm ^
    --onefile ^
    --windowed ^
    --name "TennisTagger_Training" ^
    --icon=NONE ^
    --add-data "config;config" ^
    --add-data "models;models" ^
    --hidden-import "gradio" ^
    --hidden-import "webview" ^
    --hidden-import "plotly" ^
    --hidden-import "torch" ^
    --hidden-import "torchvision" ^
    training_desktop.py

if errorlevel 1 (
    echo [ERROR] Failed to build Training EXE
    pause
    exit /b 1
)

echo [OK] Training EXE built successfully!
echo.

echo ============================================
echo   [2/2] Building Tagging EXE
echo ============================================
echo.

pyinstaller --noconfirm ^
    --onefile ^
    --windowed ^
    --name "TennisTagger_Tagging" ^
    --icon=NONE ^
    --add-data "config;config" ^
    --add-data "models;models" ^
    --hidden-import "gradio" ^
    --hidden-import "webview" ^
    --hidden-import "ultralytics" ^
    --hidden-import "mediapipe" ^
    --hidden-import "cv2" ^
    tagging_desktop.py

if errorlevel 1 (
    echo [ERROR] Failed to build Tagging EXE
    pause
    exit /b 1
)

echo [OK] Tagging EXE built successfully!
echo.

echo ============================================
echo   BUILD COMPLETE!
echo ============================================
echo.
echo Your EXE files are in the "dist" folder:
echo.
dir dist\*.exe
echo.
echo You can copy these EXE files to any Windows PC!
echo (No Python installation required on target PC)
echo.
echo IMPORTANT: Copy the following folders alongside the EXE:
echo  - config/
echo  - models/
echo  - data/
echo.

pause
