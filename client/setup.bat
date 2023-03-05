echo off
echo Creating python virtual env
echo.
CALL rd /s /q venv 2> nul
CALL rd /s /q __pycache__ 2> nul
CALL py -m venv venv 2> nul
IF %ERRORLEVEL% NEQ 0 CALL python -m venv venv 2> nul
IF %ERRORLEVEL% NEQ 0 CALL python3 -m venv venv 2> nul
IF %ERRORLEVEL% NEQ 0 (
  echo Failed to create python virtual env
  echo Do you have python installed?
  echo.
  PAUSE
  EXIT 1
)
echo Activating venv
echo.
CALL .\venv\Scripts\activate.bat
pip install windows-curses
pip install -r requirements.txt
echo.
echo Done! Run start.bat to run the program
echo.
PAUSE