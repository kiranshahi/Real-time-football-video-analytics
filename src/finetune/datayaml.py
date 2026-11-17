import argparse
from pathlib import Path
import yaml

def create_yaml(dataset_path: str):
    dataset_root = Path(dataset_path).expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    config = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "player",
            1: "referee",
            2: "goalkeeper"
        }
    }

    yaml_path = dataset_root / "dataset.yaml"

    with open(yaml_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f"YAML created: {yaml_path}")

    return yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate YOLO dataset YAML"
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root directory of the YOLO dataset"
    )

    args = parser.parse_args()

    create_yaml(args.path)
