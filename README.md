# Real-Time Football Video Analytics

A real-time computer vision pipeline for automated football video analysis, combining deep learning-based object detection, team classification and multi-object tracking to extract structured information from football broadcast footage.

## Visual Demonstration

| Input | Team Classification | Player Detection | Player Tracking |
| --- | --- | --- | --- |
| ![Input demo](/docs/img/input.gif) | ![Classification demo](/docs/img/team.gif) | ![Detection demo](/docs/img/detection.gif) | ![Tracking demo](/docs/img/track.gif) |
| Original football broadcast footage. | Identifies teams using jersey colour features and a K-Nearest Neighbours (KNN) classifier. | Detects players, referees and goalkeepers using a fine-tuned YOLO26n model. | Assigns persistent tracking IDs to players across consecutive video frames. |
| — | [Full Video](/video/output/team.mp4) | [Full Video](/video/output/detection.mp4) | [Full Video](/video/output/track.mp4) |

## Key Features

- Real-time football video processing.
- Custom-trained YOLO26n object detection.
- Player, referee and goalkeeper detection.
- Jersey colour-based team classification using KNN.
- Multi-object tracking across consecutive frames.
- Annotated video generation.
- Inference performance monitoring.

## Dataset

A curated version of the Soccer-Factory dataset, converted into YOLO annotation format for training, validation and testing.

- [Soccer-Factory Dataset on Kaggle](https://www.kaggle.com/datasets/kiranshahi/soccer-factory)

### Object Classes

- Player
- Referee
- Goalkeeper

## Model Evaluation

The YOLO26n model was fine-tuned on the curated
Soccer-Factory dataset.

| Metric | Result |
| --- | ---: |
| mAP@50 | 81.7% |
| mAP@50–95 | 67.7% |
| Player AP@50 | 97.0% |
| Test images | 6,979 |

## Getting Started

### Run Inference

```powershell
python .\src\main.py --source .\video\input\test.mp4 --output .\video\output\test.mp4 --model "yolo26n.pt"
```

## Inference Performance

The following results were recorded during a sample
video processing run.

| Metric | Result |
| --- | ---: |
| Frames processed | 404 |
| Average processing speed | 38.77 FPS |

## Technologies

- Python
- PyTorch
- Ultralytics YOLO26
- OpenCV
- Scikit-learn
- NumPy