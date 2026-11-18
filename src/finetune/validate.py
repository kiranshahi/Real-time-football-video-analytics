
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]

model = YOLO(str(root / "/workspace/run/soccer_factory/weights/best.pt"))

metrics = model.val("dataset/dataset.yaml", split="test", imgsz=960, plots=True)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
