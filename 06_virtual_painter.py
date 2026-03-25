"""
06_virtual_painter.py — Draw on Screen with Your Hand
=======================================================
A fun project combining everything:
  - MediaPipe hand tracking
  - Index fingertip as a "brush"
  - Pinch gesture to lift/lower the pen
  - Color selection by hovering over palette
  - Eraser mode
  - Clear canvas

Run:
    python 06_virtual_painter.py

Controls:
    Q  — quit
    C  — clear canvas
    Pinch (thumb+index close) — pen UP (not drawing)
    Open index finger         — pen DOWN (drawing)
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CANVAS_ALPHA    = 0.6     # how opaque the drawing is on the video
BRUSH_THICKNESS = 8
ERASER_SIZE     = 40
PINCH_THRESHOLD = 55      # pixels — thumb-index distance to lift pen

PALETTE = [
    {"label": "Red",    "color": (0,   0,   255)},
    {"label": "Green",  "color": (0,   220, 60)},
    {"label": "Blue",   "color": (255, 80,  0)},
    {"label": "Yellow", "color": (0,   220, 255)},
    {"label": "White",  "color": (255, 255, 255)},
    {"label": "Erase",  "color": (0,   0,   0)},
]
PALETTE_Y   = 10      # top of palette bar
PALETTE_H   = 50
SWATCH_W    = 70


def lm_to_px(lm, w, h, idx):
    return int(lm[idx].x * w), int(lm[idx].y * h)


def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def draw_palette(frame, active_idx):
    h, w = frame.shape[:2]
    for i, swatch in enumerate(PALETTE):
        x1 = i * SWATCH_W + 10
        x2 = x1 + SWATCH_W - 4
        y1, y2 = PALETTE_Y, PALETTE_Y + PALETTE_H
        color = swatch["color"]

        # Draw swatch
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Active border
        border_color = (255, 255, 255) if i == active_idx else (80, 80, 80)
        border_thick = 3 if i == active_idx else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

        # Label
        label_color = (0, 0, 0) if color != (0, 0, 0) else (200, 200, 200)
        cv2.putText(frame, swatch["label"], (x1 + 4, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1)


def pick_color_from_palette(finger_tip, frame_w):
    """Return palette index if fingertip is hovering over palette bar."""
    fx, fy = finger_tip
    if PALETTE_Y <= fy <= PALETTE_Y + PALETTE_H:
        for i in range(len(PALETTE)):
            x1 = i * SWATCH_W + 10
            x2 = x1 + SWATCH_W - 4
            if x1 <= fx <= x2:
                return i
    return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("Virtual Painter")
    print("Point your index finger to draw")
    print("Pinch (thumb + index) to lift pen")
    print("Hover finger over palette to change color")
    print("Q = quit   C = clear canvas")
    print("=" * 50)

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(max_num_hands=1,
                               min_detection_confidence=0.75,
                               min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas       = np.zeros((h, w, 3), dtype=np.uint8)
    prev_pt      = None
    active_color_idx = 0
    pen_down     = False
    prev_time    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        status_text  = "No hand"
        status_color = (100, 100, 100)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            index_tip  = lm_to_px(lm, w, h, 8)   # index fingertip
            thumb_tip  = lm_to_px(lm, w, h, 4)   # thumb tip
            pinch_dist = distance(index_tip, thumb_tip)
            pen_down   = pinch_dist > PINCH_THRESHOLD

            # Palette hover
            hovered = pick_color_from_palette(index_tip, w)
            if hovered is not None and pen_down:
                active_color_idx = hovered

            current_color   = PALETTE[active_color_idx]["color"]
            is_eraser       = (active_color_idx == len(PALETTE) - 1)
            draw_thickness  = ERASER_SIZE if is_eraser else BRUSH_THICKNESS
            effective_color = (0, 0, 0) if is_eraser else current_color

            if pen_down and hovered is None:
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, index_tip, effective_color, draw_thickness)
                prev_pt = index_tip
                status_text  = "ERASER" if is_eraser else "Drawing"
                status_color = (0, 180, 255) if is_eraser else (0, 255, 100)
            else:
                prev_pt = None
                status_text  = "Pen UP (pinch)"
                status_color = (200, 200, 0)

            # Draw fingertip cursor
            cursor_color = (0, 0, 0) if is_eraser else current_color
            cv2.circle(frame, index_tip, draw_thickness // 2, cursor_color, -1)
            cv2.circle(frame, index_tip, draw_thickness // 2 + 2, (255, 255, 255), 1)

            # Pinch distance indicator
            cv2.line(frame, thumb_tip, index_tip, (180, 180, 0), 1)
            mid = ((thumb_tip[0]+index_tip[0])//2, (thumb_tip[1]+index_tip[1])//2)
            cv2.putText(frame, f"{int(pinch_dist)}",
                        (mid[0]+5, mid[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 100), 1)

        else:
            prev_pt = None

        # ── Blend canvas onto frame ────────────
        mask    = canvas.astype(bool)
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(
            canvas, CANVAS_ALPHA,
            frame,  1 - CANVAS_ALPHA, 0
        )[mask]
        frame = blended

        # ── UI overlay ─────────────────────────
        # Darken top strip for palette
        strip = frame.copy()
        cv2.rectangle(strip, (0, 0), (w, PALETTE_Y + PALETTE_H + 10), (20, 20, 20), -1)
        cv2.addWeighted(strip, 0.5, frame, 0.5, 0, frame)

        draw_palette(frame, active_color_idx)

        # Status
        cv2.putText(frame, status_text, (w - 180, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time = curr_time
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 75, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

        # Hint bar
        cv2.putText(frame, "C=clear  Q=quit  |  Hover finger on palette to change color",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

        cv2.imshow("06 - Virtual Painter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
            print("Canvas cleared")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
