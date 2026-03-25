"""
05_hand_gestures.py — Hand Gesture Detection (Tasks API)
========================================================
Updated to use the modern MediaPipe Tasks API.
Includes HD camera resolution and Full Screen display.
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
    print("Download complete!")

WRIST        = 0
THUMB_CMC    = 1;  THUMB_MCP    = 2;  THUMB_IP    = 3;  THUMB_TIP    = 4
INDEX_MCP    = 5;  INDEX_PIP    = 6;  INDEX_DIP   = 7;  INDEX_TIP    = 8
MIDDLE_MCP   = 9;  MIDDLE_PIP   = 10; MIDDLE_DIP  = 11; MIDDLE_TIP   = 12
RING_MCP     = 13; RING_PIP     = 14; RING_DIP    = 15; RING_TIP     = 16
PINKY_MCP    = 17; PINKY_PIP    = 18; PINKY_DIP   = 19; PINKY_TIP    = 20

def lm_to_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_finger_up(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y

def is_thumb_up(lm, hand_label):
    is_palm_forward = (lm[PINKY_MCP].x > lm[INDEX_MCP].x) if hand_label == "Right" else (lm[PINKY_MCP].x < lm[INDEX_MCP].x)
    if hand_label == "Right":
        return lm[THUMB_TIP].x < lm[THUMB_IP].x if is_palm_forward else lm[THUMB_TIP].x > lm[THUMB_IP].x
    else:
        return lm[THUMB_TIP].x > lm[THUMB_IP].x if is_palm_forward else lm[THUMB_TIP].x < lm[THUMB_IP].x

def count_fingers(lm, hand_label):
    states = {
        "thumb":  is_thumb_up(lm, hand_label),
        "index":  is_finger_up(lm, INDEX_TIP,  INDEX_PIP),
        "middle": is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        "ring":   is_finger_up(lm, RING_TIP,   RING_PIP),
        "pinky":  is_finger_up(lm, PINKY_TIP,  PINKY_PIP),
    }
    return sum(states.values()), states

def recognize_gesture(lm, hand_label, w, h):
    count, states = count_fingers(lm, hand_label)
    if states["thumb"] and count == 1: return "Thumbs Up"
    if not states["thumb"] and count == 0:
        return "Thumbs Down" if lm[WRIST].y < lm[THUMB_TIP].y else "Fist"
    if states["index"] and states["middle"] and count == 2: return "Peace"
    if states["index"] and count == 1: return "Pointing"
    if states["index"] and states["pinky"] and count == 2: return "Rock On!"
    
    thumb_px = lm_to_px(lm[THUMB_TIP], w, h)
    index_px = lm_to_px(lm[INDEX_TIP], w, h)
    if distance(thumb_px, index_px) < 40 and states["middle"]: return "OK"
    if count == 5: return "Open Hand"
    if distance(thumb_px, index_px) < 50: return "Pinch"
    return f"{count} fingers"

def draw_skeleton(frame, lm, w, h):
    FINGER_COLORS = {"thumb": (255, 150, 0), "index": (0, 255, 0), "middle": (0, 200, 255), "ring": (255, 0, 200), "pinky": (200, 100, 255)}
    PALM_COLOR = (200, 200, 200)

    def pt(id): return lm_to_px(lm[id], w, h)
    def line(a, b, color, thick=2): cv2.line(frame, pt(a), pt(b), color, thick)
    def dot(id, color, r=5): cv2.circle(frame, pt(id), r, color, -1)

    palm_ids = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP, WRIST]
    for i in range(len(palm_ids) - 1): line(palm_ids[i], palm_ids[i + 1], PALM_COLOR, 1)
    line(THUMB_CMC, INDEX_MCP, PALM_COLOR, 1)

    for a, b in [(WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP)]:
        line(a, b, FINGER_COLORS["thumb"], 3)

    finger_chains = {"index": [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP], "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP], "ring": [RING_MCP, RING_PIP, RING_DIP, RING_TIP], "pinky": [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]}
    for name, chain in finger_chains.items():
        color = FINGER_COLORS[name]
        for i in range(len(chain) - 1): line(chain[i], chain[i + 1], color, 3)

    for i in range(21):
        dot(i, (255, 255, 255), r=4); dot(i, (80, 80, 80), r=2)

def draw_landmark_labels(frame, lm, w, h):
    for i, point in enumerate(lm):
        x, y = lm_to_px(point, w, h)
        cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 100), 1)

def draw_finger_status(frame, states, count, gesture, x_offset=10, y_offset=70):
    icons = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "P"}
    for i, (name, up) in enumerate(states.items()):
        color  = (0, 255, 100) if up else (80, 80, 80)
        xp     = x_offset + i * 34
        cv2.rectangle(frame, (xp, y_offset), (xp + 28, y_offset + 44), color, -1)
        cv2.putText(frame, icons[name], (xp + 7, y_offset + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.rectangle(frame, (x_offset, y_offset + 50), (x_offset + 170, y_offset + 80), (40, 40, 40), -1)
    cv2.putText(frame, f"Count: {count}   {gesture}", (x_offset + 5, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def draw_pinch_meter(frame, lm, w, h):
    thumb_px = lm_to_px(lm[THUMB_TIP], w, h)
    index_px = lm_to_px(lm[INDEX_TIP], w, h)
    dist     = distance(thumb_px, index_px)
    cv2.line(frame, thumb_px, index_px, (255, 220, 0), 2)
    mid = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
    cv2.putText(frame, f"{int(dist)}px", (mid[0] + 5, mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)

def main():
    print("=" * 50)
    print("Hand Gesture Detection (MediaPipe Tasks API)")
    print("Controls: Q=quit  L=labels  S=skeleton")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
        
    # Set to HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Setup Full Screen Window
    window_name = "05 - Hand Gestures"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    show_labels, show_skeleton = False, True
    prev_time = time.time()
    last_timestamp_ms = 0

    # Initialize Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_lms, hand_info in zip(result.hand_landmarks, result.handedness):
                    lm         = hand_lms
                    hand_label = hand_info[0].category_name
                    confidence = hand_info[0].score

                    if show_skeleton: draw_skeleton(frame, lm, w, h)
                    if show_labels: draw_landmark_labels(frame, lm, w, h)

                    count, states = count_fingers(lm, hand_label)
                    gesture       = recognize_gesture(lm, hand_label, w, h)

                    draw_pinch_meter(frame, lm, w, h)
                    x_off = 10 if hand_label == "Right" else w - 190
                    draw_finger_status(frame, states, count, gesture, x_off, 70)

                    wrist_px = lm_to_px(lm[WRIST], w, h)
                    cv2.putText(frame, f"{hand_label} ({confidence:.0%})", (wrist_px[0] - 40, wrist_px[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                    gesture_color = (0, 255, 150) if count > 0 else (0, 100, 255)
                    gy = h - 50 if hand_label == "Right" else h - 20
                    cv2.putText(frame, gesture, (10, gy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, gesture_color, 3)
            else:
                cv2.putText(frame, "No hand detected", (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

            curr_time, fps = time.time(), 0.0
            fps = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time
            cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "L=labels  S=skeleton  Q=quit", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('l'): show_labels = not show_labels
            elif key == ord('s'): show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()