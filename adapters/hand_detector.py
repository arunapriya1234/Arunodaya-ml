import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 Hand keypoint skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class HandDetector:
    """
    Clean Enterprise Adapter for MediaPipe Tasks HandLandmarker.
    Supports detecting BOTH hands simultaneously (max_num_hands=2).
    """
    def __init__(self, model_path: str = "hand_landmarker.task", max_num_hands: int = 2, static_image_mode: bool = False, **kwargs):
        self.model_path = model_path
        self.max_num_hands = max_num_hands
        self.static_image_mode = static_image_mode

        # 1. Download official MediaPipe HandLandmarker model asset if not present
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe HandLandmarker model asset ({self.model_path})...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download completed!")

        # 2. Create HandLandmarker detector for both hands
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def find_landmarks(self, frame_bgr: np.ndarray):
        """
        Extracts 21 3D hand keypoints (x, y, z) for ALL detected hands.
        Returns tuple: (list_of_hand_coords, annotated_frame)
        """
        annotated_frame = frame_bgr.copy()
        h, w, _ = frame_bgr.shape
        
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)
        all_coords = []

        if detection_result.hand_landmarks:
            for hand_lms in detection_result.hand_landmarks:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                all_coords.append(coords)

                # Draw green skeleton connections for each detected hand
                pixel_points = []
                for lm in hand_lms:
                    px, py = int(lm.x * w), int(lm.y * h)
                    pixel_points.append((px, py))
                    cv2.circle(annotated_frame, (px, py), 5, (0, 255, 0), -1)

                for p1, p2 in HAND_CONNECTIONS:
                    cv2.line(annotated_frame, pixel_points[p1], pixel_points[p2], (0, 255, 0), 2)

        return all_coords if len(all_coords) > 0 else None, annotated_frame

    def close(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
