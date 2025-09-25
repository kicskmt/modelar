

3Dモデリングを使用してARで遊ぶ


python -m venv env
Set-ExecutionPolicy RemoteSigned -Scope Process
env\Scripts\Activate.ps1
pip install -r requirements.txt

pyinstaller --onefile --exclude-module numpy --exclude-module pandas ???.py
pyinstaller --onefile test17-good3.py
pyinstaller --onefile test17-good2.py
pyinstaller --onefile test17-good.py
env\Scripts\pyinstaller --exclude-module pandas --onefile test17-good3.py 

pyinstaller --onefile 12_haacascade_2.py
pyinstaller --onefile 12_haacascade.py
