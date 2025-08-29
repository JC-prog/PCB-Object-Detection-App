from ultralytics import YOLO

MODEL_PATH = "backend/weights/yolov9_custom.pt"
yolo = YOLO(MODEL_PATH)

def predict_yolo(img):
    results = yolo.predict(img)
    return results[0].plot()  # returns numpy array with boxes drawn
