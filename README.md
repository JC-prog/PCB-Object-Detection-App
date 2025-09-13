# PCB Object Detection App
A Toolbox to do object detection on PCB.

# Weights
Found in OneDrive  

# Installation
## Pre-requisites
python 3.10.11

# Demo
## Change directory

## 1. Virtual Environment
```python -m venv venv```

### 2. Activate Virtual Environment
Windows
```venv\Scripts\Activate.ps1```

Mac/Linux
```source venv/bin/activate```

### 3. Install Requirements
```pip install -r requirements.txt```

#### 3.1 Pytorch 
Install one of the following, check which CUDA version the GPU supports   
CPU 
```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```

GPU
```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

## 4. Launch Demo
```python app.py```  
open http://localhost:7860 in your browser
