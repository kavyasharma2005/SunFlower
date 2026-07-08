"""
hand_tracker.py
Wraps MediaPipe Hands to detect a hand, extract landmarks, figure out which
fingers are extended, and classify that into a simple gesture name.
"""

import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame_rgb):
        """Run MediaPipe hand detection on an RGB frame."""
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw the 21 hand landmarks + connections on top of the frame (BGR, in place)."""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

    @staticmethod
    def fingers_up(hand_landmarks, handedness_label):
        """
        Returns [thumb, index, middle, ring, pinky] as booleans.
        True = finger extended, False = curled.
        """
        lm = hand_landmarks.landmark
        fingers = []

        # Thumb: compare tip(4).x vs ip(3).x. Direction flips depending on
        # which hand MediaPipe thinks it is (mirrored frame -> label can feel reversed,
        # but this holds up in practice after we flip the frame in main.py).
        if handedness_label == "Right":
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)

        # Other 4 fingers: tip above pip (smaller y) means extended.
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for tip, pip in zip(tip_ids, pip_ids):
            fingers.append(lm[tip].y < lm[pip].y)

        return fingers

    @staticmethod
    def classify_gesture(fingers):
        """
        fingers: [thumb, index, middle, ring, pinky] booleans.
        Returns 'fist', 'open_palm', 'peace', 'thumbs_up', or 'unknown'.
        """
        thumb, index, middle, ring, pinky = fingers

        if not any(fingers):
            return "fist"
        if all(fingers):
            return "open_palm"
        if index and middle and not ring and not pinky:
            return "peace"
        if thumb and not index and not middle and not ring and not pinky:
            return "thumbs_up"
        return "unknown"