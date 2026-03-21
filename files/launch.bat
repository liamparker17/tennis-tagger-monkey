@echo off
REM ============================================================================
REM Tennis Tagger - Quick Launch
REM ============================================================================
REM
REM Quick launch script for after installation.
REM Just double-click to start Tennis Tagger!
REM
REM For first-time install, use install.bat instead.
REM ============================================================================

title Tennis Tagger

REM Change to script directory
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Tennis Tagger is not installed!
    echo.
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Launch the unified app
echo.
echo Starting Tennis Tagger...
echo.

python tennis_tagger_unified.py

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. See above for details.
    pause
)
