@echo off
echo Running Python scripts in the virtual environment...

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Run the first Python script
python Betting/odds.py

REM Run the second Python script
python Betting/covers.py

pause

REM Deactivate the virtual environment
call venv\Scripts\deactivate.bat

echo Scripts have finished running.
pause
