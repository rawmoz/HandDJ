import csv
import os

import cv2

from hands import HandTracker, draw_hand, normalize

DATA_FILE = "gestures.csv"
LABELS = {"1": "rock", "2": "palm", "3": "ok", "0": "nothing"}

tracker = HandTracker()

# Count samples already saved, so you can collect over multiple runs
counts = {label: 0 for label in LABELS.values()}
if os.path.exists(DATA_FILE):
    with open(DATA_FILE) as f:
        for row in csv.reader(f):
            counts[row[0]] += 1

data_file = open(DATA_FILE, "a", newline="")  # "a" = append, never erase
writer = csv.writer(data_file)
recording = None  # which label we're saving right now

cap = cv2.VideoCapture(0)  # 0 = your Mac's built-in camera

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror it so it feels natural

    hand = tracker.find_hand(frame)
    if hand:
        draw_hand(frame, hand)
        # Save one row: label + 63 numbers
        if recording:
            writer.writerow([recording] + normalize(hand))
            counts[recording] += 1

    # On-screen instructions and sample counts
    if recording:
        cv2.putText(frame, f"RECORDING: {recording}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "1=rock 2=palm 3=ok 0=nothing SPACE=stop q=quit", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    count_text = "  ".join(f"{label}:{n}" for label, n in counts.items())
    cv2.putText(frame, count_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("HandDJ", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # press q to quit
        break
    elif key == ord(" "):
        recording = None
    elif chr(key) in LABELS:
        recording = LABELS[chr(key)]

data_file.close()
cap.release()
cv2.destroyAllWindows()
