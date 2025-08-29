# app/utils/image_utils.py
from PIL import Image
import numpy as np
import io
import base64
import cv2

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)

def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

def encode_png_base64(np_img: np.ndarray) -> str:
    # expects uint8 BGR or grayscale
    # Use cv2.imencode to produce PNG bytes
    success, encoded = cv2.imencode(".png", np_img)
    if not success:
        raise RuntimeError("Could not encode image to PNG")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def overlay_mask_on_image(image_np: np.ndarray, mask_np: np.ndarray, color=(0, 255, 0), alpha=0.5):
    """
    image_np: HxWx3 uint8 (RGB)
    mask_np: HxW uint8 (0 or 255)
    returns: HxWx3 uint8 (RGB) with overlay
    """
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    # ensure mask is 0/255
    mask_bin = (mask_np > 127).astype(np.uint8) * 255
    # create color layer
    color_layer = np.zeros_like(image_np)
    color_layer[..., 0] = color[0]
    color_layer[..., 1] = color[1]
    color_layer[..., 2] = color[2]
    # blend
    mask_bool = mask_bin.astype(bool)
    blended = image_np.copy()
    blended[mask_bool] = ((1 - alpha) * image_np[mask_bool] + alpha * color_layer[mask_bool]).astype(np.uint8)
    return blended
