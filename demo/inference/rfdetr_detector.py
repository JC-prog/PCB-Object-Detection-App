import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

class RFDetrDetector:
    """
    Adapter for RF-DETR to match the interface used in your Gradio app.
    It loads a .pth checkpoint and exposes a .predict(np_image, conf_thresh)
    method returning a list of dicts with keys:
        bbox  : [x1, y1, x2, y2]
        label : class name
        score : confidence score
    """

    # Change these if you have different categories
    CUSTOM_CLASSES = ["Background", "MountingHole", "ComponentBody",
                      "SolderJoint", "Lead"]

    def __init__(self, checkpoint_path: str):
        # RF-DETR expects a PIL image and its own weights path
        self.model = RFDETRBase(pretrain_weights=checkpoint_path)
        # optional for speed
        self.model.optimize_for_inference()

    def predict(self, np_image: np.ndarray, conf_thresh: float):
        """
        Parameters
        ----------
        np_image : np.ndarray
            RGB image (H,W,3) uint8.
        conf_thresh : float
            Minimum confidence threshold.

        Returns
        -------
        list of dict
            Each dict has keys: bbox, label, score
        """
        # convert back to PIL for RFDETR
        pil_img = Image.fromarray(np_image).convert("RGB")
        det = self.model.predict(pil_img, threshold=conf_thresh)

        results = []
        for bbox, cid, score in zip(det.xyxy, det.class_id, det.confidence):
            # det.xyxy is an array of [x1, y1, x2, y2]
            label = self.CUSTOM_CLASSES[int(cid)] if int(cid) < len(self.CUSTOM_CLASSES) else str(cid)
            results.append({
                "bbox": bbox.tolist(),
                "label": label,
                "score": float(score)
            })
        return results
