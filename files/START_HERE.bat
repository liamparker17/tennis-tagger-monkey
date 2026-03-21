@echo off
setlocal enabledelayedexpansion

:: Tennis Tagger - Unified Launcher
:: Single entry point for all Tennis Tagger applications

color 0B
title Tennis Tagger - Launcher

:MENU
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   🎾 TENNIS TAGGER - MAIN LAUNCHER                           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo   What would you like to do?
echo.
echo   [1] 🎾 Launch Video Tagging System (Desktop App)
echo   [2] 🏋️  Launch Training System (Desktop App)
echo   [3] 🔧 Diagnose PyWebView Issues
echo   [4] 🧪 Run Step-by-Step Tests
echo   [5] ❌ Exit
echo.
echo ══════════════════════════════════════════════════════════════
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto TAGGING
if "%choice%"=="2" goto TRAINING
if "%choice%"=="3" goto DIAGNOSE
if "%choice%"=="4" goto TESTS
if "%choice%"=="5" goto EXIT
goto MENU

:TAGGING
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   🎾 VIDEO TAGGING SYSTEM - STARTING...                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo   Features:
echo   ✓ Process tennis match videos
echo   ✓ Auto-detect serves, strokes, placements
echo   ✓ Generate Dartfish-compatible CSVs
echo   ✓ QC correction tools
echo.
echo ══════════════════════════════════════════════════════════════
echo.

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    goto MENU
)

:: Launch tagging desktop app
echo [*] Launching Video Tagging System...
echo.
python tagging_desktop.py

echo.
echo.
echo ══════════════════════════════════════════════════════════════
echo   Application closed.
echo ══════════════════════════════════════════════════════════════
pause
goto MENU

:TRAINING
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   🏋️  TRAINING SYSTEM - STARTING...                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo   Features:
echo   ✓ Train all 3 tasks simultaneously
echo   ✓ Model versioning (v1→v2→v3)
echo   ✓ Incremental learning
echo   ✓ Batch QC with accuracy tracking
echo   ✓ Dataset merging
echo.
echo ══════════════════════════════════════════════════════════════
echo.

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    goto MENU
)

:: Launch training desktop app
echo [*] Launching Training System...
echo.
python training_desktop.py

echo.
echo.
echo ══════════════════════════════════════════════════════════════
echo   Application closed.
echo ══════════════════════════════════════════════════════════════
pause
goto MENU

:DIAGNOSE
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   🔧 PYWEBVIEW DIAGNOSTICS                                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo   This will help diagnose and fix PyWebView display issues.
echo.
echo ══════════════════════════════════════════════════════════════
echo.

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo ══════════════════════════════════════════════════════════════
echo   STEP 1: Check WebView2 Runtime
echo ══════════════════════════════════════════════════════════════
echo.
python check_webview2.py

echo.
echo.
echo ══════════════════════════════════════════════════════════════
echo   STEP 2: Test PyWebView Backend
echo ══════════════════════════════════════════════════════════════
echo.
echo This will open a test window to verify PyWebView works...
echo.
pause
python test_pywebview_backend.py

echo.
echo.
echo ══════════════════════════════════════════════════════════════
echo   Diagnostics Complete
echo ══════════════════════════════════════════════════════════════
echo.
echo   If tests passed, you should be able to run the apps!
echo.
pause
goto MENU

:TESTS
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   🧪 STEP-BY-STEP TESTING                                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo ══════════════════════════════════════════════════════════════
echo   TEST 1: Python Installation
echo ══════════════════════════════════════════════════════════════
python --version
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    goto MENU
)
echo [OK] Python is installed
echo.
pause

echo ══════════════════════════════════════════════════════════════
echo   TEST 2: Virtual Environment
echo ══════════════════════════════════════════════════════════════
if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment found
) else (
    echo [ERROR] Virtual environment not found!
    pause
    goto MENU
)
echo.
pause

echo ══════════════════════════════════════════════════════════════
echo   TEST 3: Import Gradio
echo ══════════════════════════════════════════════════════════════
python -c "import gradio; print(f'Gradio {gradio.__version__}')"
if errorlevel 1 (
    echo [ERROR] Gradio not installed!
    pause
    goto MENU
)
echo [OK] Gradio imported successfully
echo.
pause

echo ══════════════════════════════════════════════════════════════
echo   TEST 4: Import PyWebView
echo ══════════════════════════════════════════════════════════════
python -c "import webview; print('PyWebView imported successfully')"
if errorlevel 1 (
    echo [ERROR] PyWebView not installed!
    pause
    goto MENU
)
echo [OK] PyWebView imported successfully
echo.
pause

echo ══════════════════════════════════════════════════════════════
echo   TEST 5: PyWebView Window Test
echo ══════════════════════════════════════════════════════════════
echo This will open a test window...
echo.
python test_pywebview_backend.py

echo.
echo ══════════════════════════════════════════════════════════════
echo   All Tests Complete!
echo ══════════════════════════════════════════════════════════════
echo.
pause
goto MENU

:EXIT
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo   Thanks for using Tennis Tagger!
echo ══════════════════════════════════════════════════════════════
echo.
timeout /t 2 /nobreak >nul
exit /b 0
