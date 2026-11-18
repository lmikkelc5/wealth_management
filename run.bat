@echo off
REM Advisor Sandbox launcher for Windows. Double-click to run.
cd /d "%~dp0"

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (
  where python >nul 2>nul && set PY=python
)
if not defined PY (
  echo Python wasn't found. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup. Then run this again.
  pause
  exit /b 1
)

if not exist .venv (
  echo First run: creating a virtual environment...
  %PY% -m venv .venv || (pause & exit /b 1)
)
echo Checking packages (the first run takes a minute or two)...
.venv\Scripts\python -m pip install -q --disable-pip-version-check -r requirements.txt || (pause & exit /b 1)

echo Starting Advisor Sandbox at http://localhost:8501  (close this window to stop it)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8501"
.venv\Scripts\python -m streamlit run app.py
pause
