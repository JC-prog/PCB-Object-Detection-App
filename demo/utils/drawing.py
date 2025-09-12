import hashlib
from PIL import Image, ImageDraw

def color_for_label(label: str):
    h = int(hashlib.sha256(label.encode("utf-8")).hexdigest(), 16)
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)

def draw_boxes_on_pil(image, detections, labels_to_keep=None, draw_centers=False):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        label, conf, (x1, y1, x2, y2) = det["label"], det["conf"], det["bbox"]
        if labels_to_keep and label not in labels_to_keep:
            continue

        color = tuple(det.get("color", (255, 0, 0)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((x1, y1), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 3), text, fill=(255, 255, 255))

        if draw_centers:
            cx, cy = det["center"]
            r = 5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="blue")
    return img

