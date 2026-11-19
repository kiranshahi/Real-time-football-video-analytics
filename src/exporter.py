from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "log" / "weights" / "best.pt"

model = YOLO("../log/soccer_factory/weights/best.pt")
model.export(format="onnx", quantize=16, imgsz=960)