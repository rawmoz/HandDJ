import subprocess
import time

import cv2
import joblib

from hands import HandTracker, draw_hand, normalize

MUSIC_APP = "Spotify"  # change to "Music" if you use Apple Music
ACTIONS = {"rock": "previous track", "palm": "playpause", "ok": "next track"}
MIN_CONFIDENCE = 0.8  # ignore guesses the model isn't sure about
HOLD_SECONDS = 0.5  # how long to hold a gesture before it fires


def control_music(command):
    # AppleScript: macOS's built-in way to tell apps what to do
    subprocess.Popen(["osascript", "-e", f'tell application "{MUSIC_APP}" to {command}'])


model = joblib.load("model.joblib")
tracker = HandTracker()

held_label = "nothing"  # the gesture currently being held
held_since = time.time()  # when it started
fired = False  # already triggered its action during this hold?
last_action = ""

cap = cv2.VideoCapture(0)  # 0 = your Mac's built-in camera

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror it so it feels natural

    label = "nothing"
    probabilities = [0.0] * len(model.classes_)
    hand = tracker.find_hand(frame)
    if hand:
        draw_hand(frame, hand)
        # Ask the model: how likely is each gesture? Then take the most likely one
        probabilities = model.predict_proba([normalize(hand)])[0]
        best = probabilities.argmax()
        if probabilities[best] >= MIN_CONFIDENCE:
            label = model.classes_[best]

    # A new gesture restarts the hold timer
    if label != held_label:
        held_label, held_since, fired = label, time.time(), False

    # Fire once per hold, so holding palm doesn't pause 30 times a second
    if label in ACTIONS and not fired and time.time() - held_since >= HOLD_SECONDS:
        control_music(ACTIONS[label])
        fired = True
        last_action = f"{label} -> {ACTIONS[label]}"

    cv2.putText(frame, f"Gesture: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Last action: {last_action}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # One bar per gesture showing how likely the model thinks it is
    for i, (name, p) in enumerate(zip(model.classes_, probabilities)):
        y = 120 + i * 30
        cv2.putText(frame, f"{name} {p:.0%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (150, y - 15), (150 + int(p * 200), y), (0, 255, 0), -1)

    cv2.imshow("HandDJ", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
