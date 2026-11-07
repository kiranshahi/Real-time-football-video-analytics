from dataclasses import dataclass
import cv2
from ultralytics import YOLO

@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple [int, int, int, int]  # (x, y, width, height)


class FootballDetector:
    def __init__(self, model_path: str, confidence=0.25, device=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

        self.target_classes = [0, 32]  # Person and sports ball class IDs

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.confidence, classes=self.target_classes, device=self.device, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                bbox = map(int, box.xyxy[0].tolist())
                label=self.model.names[class_id]

                detections.append(Detection(class_id, label, confidence, bbox))
        return detections

    def draw(self, frame, detections):
        output_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            color = ((0, 255, 0) if detection.class_id == 0 else (0, 165, 255))

            cv2.rectangle(output_frame, (x, y), (w, h), color, 2)
            label = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(output_frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output_frame