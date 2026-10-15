"""Hand tracking code shared by collect.py and run.py."""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandTracker:
    """Wraps Google's pretrained hand model. Give it a frame, get 21 points back."""

    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def find_hand(self, frame):
        # OpenCV uses BGR colors, MediaPipe wants RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp_ms += 33  # about 30 frames per second
        result = self.landmarker.detect_for_video(image, self.timestamp_ms)
        return result.hand_landmarks[0] if result.hand_landmarks else None


# Turn 21 points into 63 numbers that only describe hand SHAPE,
# not where the hand is on screen or how close it is to the camera
def normalize(hand):
    wrist = hand[0]
    coords = [(p.x - wrist.x, p.y - wrist.y, p.z - wrist.z) for p in hand]
    biggest = max(abs(v) for point in coords for v in point) or 1
    return [v / biggest for point in coords for v in point]


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    # Points come as 0 to 1 fractions, so scale them to pixels
    points = [(int(p.x * w), int(p.y * h)) for p in hand]
    for c in CONNECTIONS:
        cv2.line(frame, points[c.start], points[c.end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
