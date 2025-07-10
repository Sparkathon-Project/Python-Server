import cv2
import numpy as np


def detect_objects_func(model, image, CLASSES):
    # Read image bytes into numpy array
    npimg = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Run YOLO detection
    results = model(img)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        if class_name.lower() in CLASSES:
            detections.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3])
                }
            })

    return detections


def get_best_detections(detections, category):

    filtered = [d for d in detections if d["class"].lower() == category.lower()]
    if not filtered:
        return {}
    top = max(filtered, key=lambda x: x["confidence"])
    return top