from ultralytics import YOLO

model = YOLO("log/soccer_factory/weights/best.onnx", task="detect")

results = model.predict(source="video/input/test.mp4", imgsz=960, conf=0.25, save=True)