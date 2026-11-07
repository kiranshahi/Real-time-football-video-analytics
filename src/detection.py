from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

@dataclass
class Detection:
    track_id: int
    class_id: int
    confidence: float
    bbox: tuple [int, int, int, int]  # (x, y, width, height)
    center: tuple[int, int]


class FootballDetector:
    def __init__(self, model_path: str, confidence=0.25, history_size=30, device=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.history = defaultdict(lambda: deque(maxlen=history_size))

        self.target_classes = [0, 32]  # Person and sports ball class IDs

    def detect(self, frame):
        results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=self.confidence, classes=self.target_classes, device=self.device, verbose=False)
        tracked_objects = []
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return tracked_objects

        for box in boxes:
            track_id = int(box.id.item())
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if class_id == 0: 
                center = ((x1 + x2) // 2, y2)
            else: 
                center = ( (x1 + x2) // 2, (y1 + y2) // 2)
            
            tracked_objects.append(Detection(track_id, class_id, confidence, bbox=(x1, y1, x2, y2), center=center))
        
        return tracked_objects

    def draw(self, frame, detections):
        output_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            is_player = detection.class_id == 0
            color = ((0, 255, 0) if is_player else (0, 165, 255))

            cv2.rectangle(output_frame, (x, y), (w, h), color, 2)

            label = f"ID {detection.track_id}" if is_player else f"Ball {detection.track_id}"
            
            cv2.putText(output_frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if is_player:
                points = list(self.history[detection.track_id])

                for i in range(1, len(points)):
                    cv2.line(output_frame, points[i - 1], points[i], color, 2)

        return output_frame
    