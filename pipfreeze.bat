cd /d %~dp0
call venv\Scripts\activate.bat 
call pip freeze > requirements.txt

cmd.exe