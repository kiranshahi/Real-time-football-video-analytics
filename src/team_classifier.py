from collections import defaultdict, deque

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


class TeamClassifier:
    def __init__(self, min_samples=80):
        self.min_samples = min_samples
        self.samples = []

        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

        self.is_trained =False
        self.team_history = defaultdict(lambda: deque(maxlen=15))

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]

        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))

        if x2 - x1 < 12 or y2 - y1 < 24:
            return None

        player = frame[y1:y2, x1:x2]
        h, w = player.shape[:2]

        jersey = player[int(h * 0.20):int(h * 0.50), int(w * 0.25):int(w * 0.75)]

        if jersey.size == 0:
            return None

        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)

        features = np.median(lab.reshape(-1, 3), axis=0)
        return features.astype(np.float32)


    def classify_team(self, frame, detections):
        predictions = {}
        observations = []

        for detection in detections:
            # skip other class, and classify only the players (class 0)
            if detection.class_id != 0:
                continue


            features = self.extract_features(frame, detection.bboxes)

            if features is None:
                continue

            observations.append((detection.track_id, features))

            if not self.is_trained:
                self.samples.append(features)

        if not self.is_trained and len(self.samples) >= self.min_samples:
            x = np.array(self.samples, dtype=np.float32)

            labels = self.kmeans.fit_predict(x)

            self.knn.fit(x, labels)
            self.is_trained = True

        if not self.is_trained:
            return predictions

        for track_id, features in observations:
            team = int(self.knn.predict(features.reshape(1, -1))[0])
            history = self.team_history[track_id]
            history.append(team)
            predictions[track_id] = max(set(history), key=history.count)
        return predictions

