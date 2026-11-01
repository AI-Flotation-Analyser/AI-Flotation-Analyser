@echo off
setlocal
cd /d "%~dp0"
if "%~2"=="" (
  echo Usage: EXTRACT_AND_RUN_WINDOWS.bat "DATASET_ROOT" "VIDEO_MANIFEST.csv"
  pause
  exit /b 2
)
python -m pip install -r requirements-video.txt
if errorlevel 1 goto failed
python run.py --source-root "%~1" --video-manifest "%~2"
if errorlevel 1 goto failed
if exist "outputs\latest\CLIENT_DASHBOARD.html" start "" "%CD%\outputs\latest\CLIENT_DASHBOARD.html"
echo Results: outputs\latest
pause
exit /b 0
:failed
echo Error: see the preceding message.
pause
exit /b 1
