import os
import numpy as np
import gradio as gr
from PIL import Image

from utils.drawing import color_for_label, draw_boxes_on_pil
from inference.yolo_detector import YOLODetector
from inference.rtdetr_detector import RTDetrDetector
from inference.rfdetr_detector import RFDetrDetector

# --- Registry of detector backends
DETECTORS = {
    "YOLO": YOLODetector,
    "RT-DETR": RTDetrDetector,
    "RF-DETR": RFDetrDetector
}

CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def _normalize_name(s: str) -> str:
    """Remove non-alphanumeric and uppercase (for robust prefix matching)."""
    return ''.join(ch for ch in (s or "") if ch.isalnum()).upper()

def list_checkpoints(backend=None):
    """
    Optionally filter by backend: "YOLO", "RT-DETR", "RF-DETR".
    """
    files = os.listdir(CHECKPOINT_DIR)
    if backend:
        if backend.upper().startswith("YOLO"):
            exts = (".pt",)
            return [f for f in files if f.upper().startswith("YOLO") and f.endswith(exts)]
        elif backend.upper().startswith("RT-DETR"):
            return [f for f in files if f.upper().startswith("RTDETR") and f.endswith(".pt")]
        elif backend.upper().startswith("RF-DETR"):
            return [f for f in files if f.upper().startswith("RFDETR") and f.endswith(".pth")]
    # default: show all
    return [f for f in files if f.endswith((".pt", ".pth"))]

def refresh_checkpoints(selected_backend):
    """
    Refresh list according to currently selected backend.
    """
    return gr.Dropdown.update(choices=list_checkpoints(selected_backend), value=None)

detector = None  # global detector instance

def update_checkpoint_dropdown(backend_name):
    """
    Callback to update the checkpoint dropdown choices when backend is selected.
    Returns a gr.update for the cp_dropdown.
    """
    matches = list_checkpoints(backend_name)
    if matches:
        return gr.update(choices=matches, value=matches[0])
    else:
        # no matches: clear choices and value
        return gr.update(choices=[], value=None)

def load_model(backend_name, checkpoint_filename):
    global detector
    if not checkpoint_filename:
        return "No checkpoint selected."
    path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    if not os.path.exists(path):
        return f"Checkpoint not found: {path}"
    cls = DETECTORS.get(backend_name)
    if cls is None:
        return "Unknown backend."
    try:
        detector = cls(path)
        return f"Loaded {backend_name} model: {checkpoint_filename}"
    except Exception as e:
        return f"Failed to load: {e}"

def run_detection(pil_image: Image.Image, conf_thresh: float, draw_centers: bool):
    if pil_image is None:
        return None, gr.update(choices=[], value=[]), {"error": "no image"}, None
    if detector is None:
        return None, gr.update(choices=[], value=[]), {"error": "model not loaded"}, None

    np_img = np.array(pil_image)
    detections = detector.predict(np_img, conf_thresh)

    # add color + center for drawing
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        d["center"] = [(x1 + x2) / 2, (y1 + y2) / 2]
        d["color"] = color_for_label(d["label"])

    overlay = draw_boxes_on_pil(
        pil_image, detections,
        labels_to_keep=[d["label"] for d in detections],
        draw_centers=draw_centers
    )
    labels = sorted({d["label"] for d in detections})
    labels_update = gr.update(choices=labels, value=labels)

    json_dict = {
        "image_size": [pil_image.width, pil_image.height],
        "detections": detections
    }
    state_obj = {"image": pil_image, "detections": detections, "draw_centers": draw_centers}
    return overlay, labels_update, json_dict, state_obj

def update_overlay(selected_labels, state_obj, draw_centers):
    if not state_obj:
        return None, {"detections": []}, None
    image = state_obj["image"]
    detections = state_obj["detections"]

    if not selected_labels:
        overlay = image
        filtered = []
    else:
        overlay = draw_boxes_on_pil(image, detections,
                                    labels_to_keep=selected_labels,
                                    draw_centers=draw_centers)
        filtered = [d for d in detections if d["label"] in selected_labels]
    json_dict = {"image_size": [image.width, image.height], "detections": filtered}
    return overlay, json_dict, state_obj

CUSTOM_CSS = """
body { background: #0f1724; color: #e6eef8; }
.gradio-container { border-radius: 12px; padding: 18px; }
h1 { color: #ffffff; }
"""

# UI
with gr.Blocks(css=CUSTOM_CSS, title="PCB Object Detection") as demo:
    gr.Markdown("# Modular Object Detection Demo")

    with gr.Tabs():
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="Upload image")
                    submit = gr.Button("Run detection")
                with gr.Column():
                    out_img = gr.Image(type="pil", label="Annotated image")
                    labels_check = gr.CheckboxGroup(label="Toggle labels", choices=[], value=[])
                    draw_centers_toggle = gr.Checkbox(label="Draw center point", value=False)
                    json_output = gr.JSON(label="Detections JSON")
            state = gr.State()

        with gr.Tab("Model"):
            backend_dropdown = gr.Dropdown(
                choices=list(DETECTORS.keys()),
                label="Backend"
            )
            cp_dropdown = gr.Dropdown(
                choices=list_checkpoints(),   # initial (all) choices
                label="Checkpoint (.pt / .pth)"
            )
            with gr.Row():
                load_btn = gr.Button("Load model")
                refresh_btn = gr.Button("Refresh checkpoints")
            model_status = gr.Textbox(label="Model status", interactive=False)
            conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.01,
                                    label="Confidence threshold")

    # ---------------- Events ----------------
    # Update cp_dropdown whenever backend changes
    backend_dropdown.change(
        fn=update_checkpoint_dropdown,
        inputs=[backend_dropdown],
        outputs=[cp_dropdown]
    )

    # Refresh button re-scans based on current backend
    refresh_btn.click(
        fn=lambda backend: gr.Dropdown.update(
            choices=list_checkpoints(backend),
            value=None
        ),
        inputs=[backend_dropdown],
        outputs=[cp_dropdown]
    )

    # Load and inference
    load_btn.click(load_model, [backend_dropdown, cp_dropdown], [model_status])
    submit.click(run_detection, [img_in, conf_slider, draw_centers_toggle],
                 [out_img, labels_check, json_output, state])
    labels_check.change(update_overlay, [labels_check, state, draw_centers_toggle],
                        [out_img, json_output, state])
    draw_centers_toggle.change(update_overlay, [labels_check, state, draw_centers_toggle],
                               [out_img, json_output, state])


if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False)
