from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, cfg):
        self.model = YOLO(cfg["model_path"])
        self.conf = cfg["confidence"]
        self.iou = cfg["iou"]
        self.classes = cfg["classes"]

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.classes.get(cls_id, "unknown")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
