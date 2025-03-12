# This is project of MAS291 - Topic 1:

The dataset offers comprehensive information on health factors influencing osteoporosis development, including demographic details, lifestyle choices, medical history, and bone health indicators. It aims to facilitate research in osteoporosis prediction, enabling machine learning models to identify individuals at risk. Analyzing factors like age, gender, hormonal changes, and lifestyle habits can help improve osteoporosis management and prevention strategies.

# Supported OS:

- Windows 11 x64/ x86
- Windows 11 ARM 64-bit
- macOS 15 Sequoia 15.1 ARM 64-bit or newer

The author has tested this program on several laptops running a wide variety of CPUs, including Intel Core Ultra, Snapdragon X Elite, Apple M-Series

# Required third party software

- Python: to run the script
- A default browser: Edge, Safari, or any default browser of your choice. Edge and Safari are recommended since they are default browser for most Windows and Mac machine

# How to run
## Installation

1. Install Python https://www.python.org/downloads/
2. Download this code by click on the Code (green button) > Download ZIP, or you can use `git clone` this repo to your machine.
3. Unzip the ZIP file you have just download
4. Open Command Prompt (cmd) and navigate to your extracted folder (e.g: 
`cd /Users/its.qvnvm/Downloads/MAS291`)

## Run the program

**Always navigate to project folder first**
**Before run, please delete all picture first in folder that you choose**


### macOS:

Open Terminal and run the following commands
```
python3 -m venv MAS291_venv
source MAS291_venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
**Select a folder want to run, e.g: phantich**

```
cd phantich
```

**Then run phantich.py**
```
python3 -m phantich
```

**You only need to do it once, on the first time**. For the following runs, you only need to run these code


```
source MAS291_venv/bin/activate
python3 -m phantich
```

### Windows:

Note: You must run the following commands in Command Prompt (CMD), do not run on PowerShell
```
py -m venv MAS291_venv
MAS291_venv\Scripts\activate.bat
py -m pip install --upgrade pip
pip install -r requirements.txt
cd phantich
py -m phantich
```

**You only need to do it once, on the first time**. For the following runs, you only need to run these code

```
cd phantich
py -m phantich
```