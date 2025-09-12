import numpy as np
from ultralytics import RTDETR
from .base import BaseDetector

class RTDetrDetector(BaseDetector):
    def __init__(self, weights_path: str):
        self.model = RTDETR(weights_path)
        self.names = self.model.names

    def predict(self, image: np.ndarray, conf: float):
        results = self.model.predict(source=image, conf=conf, imgsz=640, verbose=False)[0]
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_inds = results.boxes.cls.cpu().numpy().astype(int)

        out = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            label = self.names.get(cls_inds[i], str(cls_inds[i]))
            out.append({
                "label": label,
                "conf": float(confs[i]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        return out