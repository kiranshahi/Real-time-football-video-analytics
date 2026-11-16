import cv2
import json
from pathlib import Path

import numpy as np
import random
from collections import Counter


BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = BASE_DIR / "dataset" / "soccer_factory_release"
OUTPUT_DIR = BASE_DIR / "dataset" / "yolo_dataset"

FRAMES_PER_VIDEO = 10
SEED = 42

classes = {'player': 0, 'referee': 1, 'goalkeeper': 2}

videos = sorted(SOURCE_DIR.glob("soccer_factory_videos_part*/videos/*.mp4"))
json_files = list(SOURCE_DIR.glob("soccer_factory_annotations/annotations/*.json"))


annotation_map = {p.stem: p for p in json_files}

## Match video with json
videos = [
    video for video in videos
    if video.stem in annotation_map
]

# split video int train, val and test

rng = random.Random(SEED)
rng.shuffle(videos)

n = len(videos)


train_end = int(n * 0.8)
val_end = int(n * 0.9)

splits = {
    "train": videos[:train_end],
    "val": videos[train_end:val_end],
    "test": videos[val_end:]
}

counts = Counter()


def process_video(video_path, split):
    json_path = annotation_map[video_path.stem]
    
    with open(json_path, "r") as f:
        annotations = json.load(f)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num=min(FRAMES_PER_VIDEO, total_frames), dtype=int)

    image_dir = OUTPUT_DIR / "images" / split
    label_dir = OUTPUT_DIR / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()
        
        if not success:
            continue

        frame_id = str(int(index) + 1)

        frame_data = annotations.get(frame_id)

        if frame_data is None:
            continue

        height, width = frame.shape[:2]
        yolo_labels = []
        
        for person in frame_data.get("people", []):
            role = person.get("role")
            
            if role not in classes:
                continue

            x, y, w, h = person["bbox_ltwh"]

            # bounding box
            x1 = max(0.0, min(float(x), width))
            y1 = max(0.0, min(float(y), height))
            x2 = max(0.0, min(float(x + w), width))
            y2 = max(0.0, min(float(y + h), height))

            if x2 <= x1 or y2 <= y1:
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            wn = (x2 - x1) / width
            hn = (y2 - y1) / height

            class_id = classes[role]
            
            yolo_labels.append(f"{class_id} " f"{xc:.6f} {yc:.6f} " f"{wn:.6f} {hn:.6f}")

            counts[role] += 1

        filename = f"{video_path.stem}_{index:05d}"

        image_path = image_dir / f"{filename}.jpg"
        label_path = label_dir / f"{filename}.txt"

        saved = cv2.imwrite(str(image_path), frame)

        if saved:
            label_path.write_text("\n".join(yolo_labels), encoding="utf-8")

    cap.release()

# Process all videos
for split, video_list in splits.items():
    print(f"\nProcessing {split}: {len(video_list)} videos")

    for i, video in enumerate(video_list, start=1):
        process_video(video, split)

        if i % 100 == 0:
            print(f"{split}: {i}/{len(video_list)}")

# Generate YOLO dataset configuration
yaml_content = f"""path: '{OUTPUT_DIR.as_posix()}'
train: images/train
val: images/val
test: images/test

names:
  0: player
  1: referee
  2: goalkeeper
"""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

print("\nDataset preparation completed!")
print("Output:", OUTPUT_DIR)
print("Annotation counts:", dict(counts))

