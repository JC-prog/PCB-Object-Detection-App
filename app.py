import os
import hashlib
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from ultralytics import YOLO

# --- Directories (no need for OUTPUT_DIR anymore)
CHECKPOINT_DIR = "Checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------- Helpers ----------
def list_checkpoints():
    return [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]

def color_for_label(label: str):
    h = int(hashlib.sha256(label.encode("utf-8")).hexdigest(), 16)
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)

def ensure_numpy(x):
    try:
        return x.cpu().numpy()
    except Exception:
        return np.array(x)

def draw_boxes_on_pil(image, detections, labels_to_keep=None, draw_centers=False):
    """
    Draw bounding boxes (and optional center points) on a PIL image.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        label, conf, (x1, y1, x2, y2) = det["label"], det["conf"], det["bbox"]

        if labels_to_keep and label not in labels_to_keep:
            continue

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=tuple(det.get("color", (255,0,0))), width=3)

        # Draw label
        text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((x1, y1), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=tuple(det.get("color", (255,0,0))))
        draw.text((x1 + 3, y1 - th - 3), text, fill=(255,255,255))

        # Draw center point
        if draw_centers:
            cx, cy = det["center"]
            r = 5
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill="blue")

    return img

# ---------- Model handling ----------
MODEL = None
MODEL_PATH = None

def load_model(checkpoint_filename: str):
    global MODEL, MODEL_PATH
    if not checkpoint_filename:
        return "No checkpoint selected."
    full = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    if not os.path.exists(full):
        return f"Checkpoint not found: {full}"
    try:
        MODEL = YOLO(full)
        MODEL_PATH = full
        return f"Loaded model: {checkpoint_filename}"
    except Exception as e:
        return f"Failed to load model: {e}"

# ---------- Detection ----------
def run_detection(pil_image: Image.Image, conf_thresh: float, draw_centers: bool):
    global MODEL, MODEL_PATH
    if pil_image is None:
        return None, gr.update(choices=[], value=[]), {"error": "no image"}, None
    if MODEL is None:
        return None, gr.update(choices=[], value=[]), {"error": "model not loaded"}, None

    np_img = np.array(pil_image)
    results = MODEL.predict(source=np_img, conf=conf_thresh, imgsz=640, verbose=False)
    if len(results) == 0:
        return pil_image, gr.update(choices=[], value=[]), {"detections": []}, {"image": pil_image, "detections": []}
    r = results[0]

    xyxy = ensure_numpy(r.boxes.xyxy)
    confs = ensure_numpy(r.boxes.conf)
    cls_inds = ensure_numpy(r.boxes.cls).astype(int)
    names = getattr(MODEL, "names", {})

    detections = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(x) for x in box]
        ci = int(cls_inds[i])
        label = names.get(ci, str(ci))
        conf = float(confs[i])
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        color = color_for_label(label)
        detections.append({
            "label": label,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "color": color
        })

    overlay = draw_boxes_on_pil(pil_image, detections, labels_to_keep=[d["label"] for d in detections], draw_centers=draw_centers)
    labels = sorted({d["label"] for d in detections})
    labels_update = gr.update(choices=labels, value=labels)

    json_dict = {
        "image_size": [pil_image.width, pil_image.height],
        "model": MODEL_PATH,
        "detections": detections
    }

    state_obj = {"image": pil_image, "detections": detections, "draw_centers": draw_centers}
    return overlay, labels_update, json_dict, state_obj

# ---------- Update overlay ----------
def update_overlay(selected_labels, state_obj, draw_centers):
    if not state_obj:
        return None, {"detections": []}, None

    image = state_obj["image"]
    detections = state_obj["detections"]

    if not selected_labels:
        overlay = image
        filtered = []
    else:
        overlay = draw_boxes_on_pil(image, detections, labels_to_keep=selected_labels, draw_centers=draw_centers)
        filtered = [d for d in detections if d["label"] in selected_labels]

    json_dict = {"image_size": [image.width, image.height], "detections": filtered}
    return overlay, json_dict, state_obj

# ---------- UI with tabs ----------
CUSTOM_CSS = """
body { background: #0f1724; color: #e6eef8; }
.gradio-container { border-radius: 12px; padding: 18px; }
h1 { color: #ffffff; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="Object Detection") as demo:
    gr.Markdown("# Object Detection — Custom Tabs")

    with gr.Tabs():
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Input")
                    img_in = gr.Image(type="pil", label="Upload image")
                    submit = gr.Button("Run detection")
                with gr.Column():
                    gr.Markdown("## Output")
                    out_img = gr.Image(type="pil", label="Annotated image")  # user can download directly
                    labels_check = gr.CheckboxGroup(label="Toggle labels", choices=[], value=[])
                    draw_centers_toggle = gr.Checkbox(label="Draw center point", value=False)
                    json_output = gr.JSON(label="Detections JSON")

            state = gr.State()

        with gr.Tab("Model"):
            cp_dropdown = gr.Dropdown(choices=list_checkpoints(), label="Choose checkpoint (.pt)")
            load_btn = gr.Button("Load model")
            model_status = gr.Textbox(label="Model status", interactive=False)
            conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence threshold")

    load_btn.click(load_model, [cp_dropdown], [model_status])
    submit.click(run_detection, [img_in, conf_slider, draw_centers_toggle], [out_img, labels_check, json_output, state])
    labels_check.change(update_overlay, [labels_check, state, draw_centers_toggle], [out_img, json_output, state])
    draw_centers_toggle.change(update_overlay, [labels_check, state, draw_centers_toggle], [out_img, json_output, state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
