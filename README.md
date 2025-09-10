# PCB Object Detection App


# Installation

## Pre-requisites
python 3.10.11

## Virtual Environment
python -m venv venv

### Activate Virtual Environment
Windows
venv\Scripts\Activate.ps1

Mac/Linux
source venv/bin/activate

### Requirements
pip install -r requirements.txt

### Pytorch
CPU 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Launch
python app.py
open http://localhost:7860 in your browser
