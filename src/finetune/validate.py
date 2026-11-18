
import argparse
from pathlib import Path
from ultralytics import YOLO

def validate(args):
    dataset_yaml = Path(args.data).expanduser().resolve()

    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "/workspace/run/soccer_factory/weights/best.pt"))
    
    metrics = model.val(dataset_yaml, split="test", imgsz=960, plots=True)
    
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Soccer Factory")
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    
    validate(args)