from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def __init__(self, weights_path: str):
        """Load the model weights"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float):
        """
        Run inference on a numpy image.
        Return list of dicts:
        [
          {"label": str, "conf": float,
           "bbox": [x1, y1, x2, y2]}
        ]
        """
        pass
