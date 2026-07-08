"""
main.py
Hand-Sign Flower Garden
Captures webcam video, tracks your hand, and grows a flower based on the
sign you're making:

    Fist        -> grows the stem
    Open palm   -> blooms the petals
    Peace sign  -> grows the leaves

Controls:
    R -> reset the flower
    Q -> quit
"""

import cv2

from hand_tracker import HandTracker
from flower import Flower

GESTURE_LABELS = {
    "fist": "Fist -> growing stem",
    "open_palm": "Open Palm -> blooming petals",
    "peace": "Peace Sign -> growing leaves",
    "thumbs_up": "Thumbs Up",
    "unknown": "Show a hand sign...",
}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Try changing VideoCapture(0) to a different index.")
        return

    tracker = HandTracker(max_hands=1)
    flower = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        h, w, _ = frame.shape

        if flower is None:
            flower = Flower(base_point=(w // 2, h - 40))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.process(rgb)

        gesture = "unknown"
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label
            fingers = tracker.fingers_up(hand_landmarks, handedness_label)
            gesture = tracker.classify_gesture(fingers)
            tracker.draw_landmarks(frame, hand_landmarks)

            if gesture == "fist":
                flower.grow_stem()
            elif gesture == "open_palm":
                flower.grow_petals()
            elif gesture == "peace":
                flower.grow_leaves()

        frame = flower.draw(frame)

        cv2.putText(frame, GESTURE_LABELS.get(gesture, gesture), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    "Fist: stem | Open palm: petals | Peace: leaves | R: reset | Q: quit",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        if flower.is_fully_bloomed():
            cv2.putText(frame, "Fully Bloomed!", (w // 2 - 130, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Sign Flower Garden", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            flower.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
