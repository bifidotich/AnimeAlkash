cd /d %~dp0
call venv\Scripts\activate.bat
call python.exe -m pip install --upgrade pip
pip install notebook
jupyter notebook
cmd.exe