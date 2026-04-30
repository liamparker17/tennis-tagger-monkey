@echo off
REM Tennis Tagger launcher — uses the bundled embedded Python and ffmpeg.
REM Do not move this file out of the install directory.

setlocal
set "APP_DIR=%~dp0"
set "PATH=%APP_DIR%ffmpeg;%PATH%"
set "PYTHONHOME="
set "PYTHONPATH=%APP_DIR%ml;%APP_DIR%"
set "TENNIS_TAGGER_HOME=%APP_DIR%"
set "TENNIS_TAGGER_PYTHON=%APP_DIR%python\python.exe"

cd /d "%APP_DIR%"

REM Use pythonw.exe so the GUI launches without a console window.
REM _launch.py wraps tagger_ui.py and redirects stderr to launcher.log so
REM silent crashes (missing module, broken wheel, etc.) leave a trace.
start "" "%APP_DIR%python\pythonw.exe" "%APP_DIR%_launch.py" %*
endlocal
