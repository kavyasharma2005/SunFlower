# Hand-Sign Flower Garden

Grow a flower on your webcam feed using hand signs, powered by MediaPipe Hands + OpenCV.

## Gestures

| Sign | Effect |
|---|---|
| ✊ Fist | Grows the stem |
| 🖐 Open palm | Blooms the petals |
| ✌ Peace sign | Grows the leaves |

Keys: **R** = reset flower, **Q** = quit

The stem has to be at least ~20% grown before leaves can start, and ~35% grown before petals can start — so grow the stem first for the most natural-looking flower.

## Setup

1. Make sure you have Python 3.9–3.11 installed (MediaPipe doesn't yet support every newer Python version — check `python3 --version`).
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run it:
   ```bash
   python main.py
   ```

A window titled "Hand Sign Flower Garden" will open showing your webcam feed with hand landmarks and the growing flower overlaid.

## Project structure

- `hand_tracker.py` — wraps MediaPipe Hands, figures out which fingers are up, classifies the gesture (fist / open_palm / peace / thumbs_up / unknown).
- `flower.py` — holds the flower's growth state (stem height, petal progress, leaf progress) and draws it with anti-aliased shapes and soft alpha blending.
- `main.py` — captures the webcam, ties tracking + flower together, and renders the on-screen HUD.

## Troubleshooting

- **Webcam doesn't open**: try changing `cv2.VideoCapture(0)` in `main.py` to `1` or `2` — some laptops enumerate the built-in camera at a different index, especially with an external webcam plugged in.
- **MediaPipe install fails**: it needs a 64-bit Python between 3.9 and 3.11 on most platforms. If you're on 3.12+, install 3.11 alongside it and create the venv with that version instead.
- **Gesture not recognized**: keep your hand roughly upright and 30–60cm from the camera, with decent lighting. The thumb detection direction depends on which hand MediaPipe thinks it's seeing, so it works best with one consistent hand (left or right).

## Ideas to extend it

- Add a "thumbs up" gesture (already detected, just unused) to trigger a sparkle/bloom-complete effect.
- Track two hands and grow two flowers side by side.
- Save a screenshot of the fully-bloomed flower with a keypress.
- Swap petal colors based on which finger count gesture is held (e.g. 3 fingers = red, 4 = blue).
