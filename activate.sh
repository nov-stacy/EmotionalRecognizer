#!/bin/bash
source venv/bin/activate
echo [+] Start checking for libraries
pip install -r venv/requirements.txt > pip.txt
echo [+] Stop checking for libraries
rm pip.txt
python3 main.py