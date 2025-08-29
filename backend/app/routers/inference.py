from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import numpy as np
import cv2
import base64
from torchvision import transforms as T
from ultralytics import YOLO

# Import your models
from app.models.unet_model import UNet

# --- Load YOLOv9 model once ---
try:
    yolo_model = YOLO("./weights/yolov9_model.pt")
    print("YOLO loaded. Classes:", yolo_model.names)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

router = APIRouter(prefix="/api", tags=["inference"])

# Available model classes
MODEL_CLASSES = {
    "unet": UNet
}

# Transform (must match training)
IMG_SIZE = (256, 256)
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility: PIL -> base64
def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def numpy_to_base64(np_img: np.ndarray) -> str:
    pil_img = Image.fromarray(np_img)
    return pil_to_base64(pil_img)

# Optional: overlay mask on original
def overlay_mask(orig: np.ndarray, mask: np.ndarray, color=(0,255,0), alpha=0.5):
    overlay = orig.copy()
    overlay[mask>0] = (overlay[mask>0]* (1-alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay

@router.post("/segment")
async def segment_image(file: UploadFile = File(...), model_name: str = "unet"):
    # Validate model
    if model_name.lower() not in MODEL_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    ModelClass = MODEL_CLASSES[model_name.lower()]
    model = ModelClass().to(device)
    
    # Load weights (adjust path)
    weight_path = f"./weights/{model_name}_custom.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Read uploaded image
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Keep original as numpy
    orig_np = np.array(pil_img)

    # Transform
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)  # 1x1xHxW
        mask = pred[0,0].cpu().numpy()
    
    # Binary mask 0/255
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    # Resize mask to match original image size
    mask_resized = cv2.resize(mask_bin, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_pil = Image.fromarray(mask_resized)

    # Overlay mask on original
    overlay_np = overlay_mask(orig_np, mask_resized)
    overlay_pil = Image.fromarray(overlay_np)


    return JSONResponse({
        "original": pil_to_base64(pil_img),
        "mask": pil_to_base64(mask_pil),
        "overlay": pil_to_base64(overlay_pil)
    })

@router.post("/segment_and_detect")
async def segment_and_detect(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    Run UNet segmentation + YOLO detection on uploaded image.
    Returns:
        - original image
        - UNet mask
        - UNet overlay
        - YOLO overlay
        - number of YOLO detections
    """
    # --- Read image ---
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # --- UNet segmentation ---
    model = UNet().to(device)
    model.load_state_dict(torch.load("./weights/unet_custom.pth", map_location=device))
    model.eval()

    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
        mask = pred[0,0].cpu().numpy()
    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_bin, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_np = overlay_mask(orig_np, mask_resized)

    # --- YOLO detection ---
    results = yolo_model.predict(orig_np, conf=conf_threshold, verbose=False)
    yolo_overlay = results[0].plot()  # BGR
    yolo_overlay = cv2.cvtColor(yolo_overlay, cv2.COLOR_BGR2RGB)
    num_detections = len(results[0].boxes)

    # --- Return all images as base64 ---
    return JSONResponse({
        "original": pil_to_base64(pil_img),
        "mask": numpy_to_base64(mask_resized),
        "overlay": numpy_to_base64(overlay_np),
        "yolo_overlay": numpy_to_base64(yolo_overlay),
        "num_detections": num_detections
    })
