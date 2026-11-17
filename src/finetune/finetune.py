import argparse
import torch
from pathlib import Path
from ultralytics import YOLO




def train(args):
    dataset_yaml = Path(args.data).expanduser().resolve()
    
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

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
                            project="/workspace/run/",
                            name="soccer_factory",
                            exist_ok=True,
                            plots=True,
                            seed=42)

    return results

if __name__ == "__main__":
    train()