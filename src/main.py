import argparse
from pathlib import Path
import time
from tracemalloc import start
import cv2

from detection import FootballDetector
from team_classifier import TeamClassifier

def process_video(source, output_path, model_path):
    detector = FootballDetector(model_path)
    team_classifier = TeamClassifier()

    video_source = (int(source) if source.isdigit() else source)

    captuer = cv2.VideoCapture(video_source)

    if not captuer.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    width = int(captuer.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captuer.get(cv2.CAP_PROP_FRAME_HEIGHT))

    source_fps = captuer.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        source_fps = 30.0

    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))

    if not writer.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        captuer.release()
        return

    frame_count = 0
    total_processing_time = 0.0

    try:
        while True:
            ret, frame = captuer.read()
            if not ret:
                break

            start = time.perf_counter()

            detections = detector.detect(frame)

            teams = team_classifier.classify_team(frame, detections)
            annotated_frame = detector.draw(frame, detections, teams)
          

            processing_time = (time.perf_counter() - start)
            total_processing_time += processing_time
            frame_count += 1

            fps = (1.0 / processing_time if processing_time > 0 else 0.0)

            cv2.putText(annotated_frame, f"Processing FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(annotated_frame)

            cv2.imshow("Football Analytics", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        captuer.release()
        writer.release()
        cv2.destroyAllWindows()

    avg_fps = (frame_count / total_processing_time if total_processing_time > 0 else 0.0)
    
    # print debugging data
    print(f"Frames processed: {frame_count}")
    print(f"Average processing FPS: {avg_fps:.2f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Detection in Video")
    parser.add_argument("--source", type=str, required=True, help="Video path or camera index (e.g., 0 for webcam)")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--model", type=str, required=True, default="yolo26n.pt", help="Path to the YOLO model file")
    args = parser.parse_args()

    process_video(args.source, args.output, args.model)