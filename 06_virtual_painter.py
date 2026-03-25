"""
06_virtual_painter.py — Draw on Screen with Your Hand (Tasks API)
=================================================================
Updated to use the modern MediaPipe Tasks API.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Auto-Download Model ──────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print(f"Downloading MediaPipe model to {MODEL_PATH}...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

CANVAS_ALPHA    = 0.6
BRUSH_THICKNESS = 8
ERASER_SIZE     = 40
PINCH_THRESHOLD = 55

PALETTE = [
    {"label": "Red",    "color": (0,   0,   255)},
    {"label": "Green",  "color": (0,   220, 60)},
    {"label": "Blue",   "color": (255, 80,  0)},
    {"label": "Yellow", "color": (0,   220, 255)},
    {"label": "White",  "color": (255, 255, 255)},
    {"label": "Erase",  "color": (0,   0,   0)},
]
PALETTE_Y, PALETTE_H, SWATCH_W = 10, 50, 70

def lm_to_px(lm, w, h, idx):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw_palette(frame, active_idx):
    for i, swatch in enumerate(PALETTE):
        x1, x2 = i * SWATCH_W + 10, i * SWATCH_W + 10 + SWATCH_W - 4
        y1, y2 = PALETTE_Y, PALETTE_Y + PALETTE_H
        color = swatch["color"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        border_color = (255, 255, 255) if i == active_idx else (80, 80, 80)
        border_thick = 3 if i == active_idx else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

        label_color = (0, 0, 0) if color != (0, 0, 0) else (200, 200, 200)
        cv2.putText(frame, swatch["label"], (x1 + 4, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1)

def pick_color_from_palette(finger_tip):
    fx, fy = finger_tip
    if PALETTE_Y <= fy <= PALETTE_Y + PALETTE_H:
        for i in range(len(PALETTE)):
            x1, x2 = i * SWATCH_W + 10, i * SWATCH_W + 10 + SWATCH_W - 4
            if x1 <= fx <= x2: return i
    return None

def main():
    print("=" * 50)
    print("Virtual Painter (MediaPipe Tasks API)")
    print("Point index finger to draw | Pinch to lift pen | Hover palette to change color")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    prev_pt, active_color_idx, pen_down = None, 0, False
    prev_time, last_timestamp_ms = time.time(), 0
    hover_start_time, last_hovered_idx = 0, None

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,                                 # <-- FIXED HERE
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.6
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms: timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)
            status_text, status_color = "No hand", (100, 100, 100)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                index_tip  = lm_to_px(lm, w, h, 8)
                thumb_tip  = lm_to_px(lm, w, h, 4)
                pinch_dist = distance(index_tip, thumb_tip)
                pen_down   = pinch_dist > PINCH_THRESHOLD

                hovered = pick_color_from_palette(index_tip)
                if hovered is not None and pen_down:
                    if hovered == last_hovered_idx:
                        if time.time() - hover_start_time > 0.5: active_color_idx = hovered
                    else:
                        last_hovered_idx, hover_start_time = hovered, time.time()
                else:
                    last_hovered_idx = None

                current_color   = PALETTE[active_color_idx]["color"]
                is_eraser       = (active_color_idx == len(PALETTE) - 1)
                draw_thickness  = ERASER_SIZE if is_eraser else BRUSH_THICKNESS
                effective_color = (0, 0, 0) if is_eraser else current_color

                if pen_down and hovered is None:
                    if prev_pt is not None: cv2.line(canvas, prev_pt, index_tip, effective_color, draw_thickness)
                    prev_pt = index_tip
                    status_text, status_color = ("ERASER", (0, 180, 255)) if is_eraser else ("Drawing", (0, 255, 100))
                else:
                    prev_pt = None
                    status_text, status_color = "Pen UP (pinch)", (200, 200, 0)

                cursor_color = (0, 0, 0) if is_eraser else current_color
                cv2.circle(frame, index_tip, draw_thickness // 2, cursor_color, -1)
                cv2.circle(frame, index_tip, draw_thickness // 2 + 2, (255, 255, 255), 1)
                cv2.line(frame, thumb_tip, index_tip, (180, 180, 0), 1)
            else:
                prev_pt = None

            mask    = canvas.astype(bool)
            blended = frame.copy()
            blended[mask] = cv2.addWeighted(canvas, CANVAS_ALPHA, frame, 1 - CANVAS_ALPHA, 0)[mask]
            frame = blended

            strip = frame.copy()
            cv2.rectangle(strip, (0, 0), (w, PALETTE_Y + PALETTE_H + 10), (20, 20, 20), -1)
            cv2.addWeighted(strip, 0.5, frame, 0.5, 0, frame)

            draw_palette(frame, active_color_idx)
            cv2.putText(frame, status_text, (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time
            cv2.putText(frame, f"FPS {fps:.0f}", (w - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

            cv2.imshow("06 - Virtual Painter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()