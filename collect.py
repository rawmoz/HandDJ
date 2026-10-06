import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# Load Google's pretrained hand model
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

cap = cv2.VideoCapture(0)  # 0 = your Mac's built-in camera
timestamp_ms = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror it so it feels natural

    # OpenCV uses BGR colors, MediaPipe wants RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 33  # about 30 frames per second
    result = landmarker.detect_for_video(image, timestamp_ms)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        # Points come as 0 to 1 fractions, so scale them to pixels
        points = [(int(p.x * w), int(p.y * h)) for p in hand]
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (255, 255, 255), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

    cv2.imshow("HandDJ", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()