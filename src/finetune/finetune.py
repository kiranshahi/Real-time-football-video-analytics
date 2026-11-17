import torch
from pathlib import Path
from ultralytics import YOLO

import yaml

dataset_path = Path("/kaggle/input/datasets/kiranshahi/soccer-factory/yolo_dataset")

config = {
    "path": str(dataset_path),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {
        0: "player",
        1: "referee",
        2: "goalkeeper"
    }
}

yaml_path = "/kaggle/working/dataset.yaml"

with open(yaml_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)



def train():
    dataset_yaml = "/kaggle/working/dataset.yaml"

    device = 0 if torch.cuda.is_available() else "cpu"

    # Load COCO-pretrained object detection weights
    model = YOLO("yolo26n.pt")

    results = model.train(
                            data=dataset_yaml, 
                            epochs=50, 
                            imgsz=960, 
                            batch=8, 
                            device=device, 
                            workers=4, 
                            patience=15, 
                            optimizer="auto", 
                            pretrained=True,
                            project="/kaggle/working/",
                            name="soccer_factory",
                            exist_ok=True,
                            plots=True,
                            seed=42)

    return results

if __name__ == "__main__":
    train()