@echo off
setlocal
cd /d "%~dp0"
if not exist "outputs" mkdir "outputs"
set "SOURCE_ROOT=%~1"
if "%SOURCE_ROOT%"=="" (
    if exist "..\AI_Flotation" (set "SOURCE_ROOT=..\AI_Flotation") else (set "SOURCE_ROOT=data")
)
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python run.py --source-root "%SOURCE_ROOT%"
if errorlevel 1 goto failed
if exist "outputs\latest\CLIENT_DASHBOARD.html" start "" "%CD%\outputs\latest\CLIENT_DASHBOARD.html"
echo.
echo Results: outputs\latest
pause
exit /b 0
:failed
echo.
echo Run failed. Review the error above.
pause
exit /b 1
