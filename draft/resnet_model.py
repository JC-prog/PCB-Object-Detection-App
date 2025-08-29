import torch
import os
from app.models.resnet_model import ResNetSegmentation

# Use absolute path so it works regardless of current working dir
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "resnet_custom.pth")  # change to your weights filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model architecture (must match what you trained)
model = ResNetSegmentation(out_channels=1, pretrained=False)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()
model.to(device)

# Preprocessing function (resize, to tensor, etc.)
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # match your training input size
    transforms.ToTensor(),
])

def predict_resnet_seg(img: Image.Image):
    """Predict segmentation mask from PIL image"""
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(img_tensor)[0,0].cpu().numpy()

    # Threshold to binary mask
    mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Resize mask back to original image size
    mask_resized = cv2.resize(mask, img.size)

    return mask_resized
