"""
05_hand_gestures.py — Hand Gesture Detection
=============================================
Topics covered:
  - MediaPipe Hands for landmark detection
  - 21 hand landmarks explained
  - Finger counting logic
  - Gesture recognition (thumbs up, peace, fist, open hand)
  - Drawing skeleton + landmarks
  - Distance between fingertips (pinch detection)

Run:
    python 05_hand_gestures.py

Controls:
    Q — quit
    L — toggle landmark labels
    S — toggle skeleton drawing
"""

import cv2
import mediapipe as mp
import numpy as np
import time


# ─────────────────────────────────────────────
# MediaPipe Hand Landmark IDs
# ─────────────────────────────────────────────
#
#   Landmark map (21 points per hand):
#
#       8   12  16  20       ← fingertips
#       7   11  15  19
#       6   10  14  18
#       5    9  13  17       ← MCP joints (knuckles)
#            |
#       4 (thumb tip)
#       3
#       2
#       1
#       0  (wrist)
#
# Each landmark has: .x .y .z  (normalized 0.0–1.0)
# Multiply by frame width/height to get pixel coordinates

WRIST        = 0
THUMB_CMC    = 1;  THUMB_MCP    = 2;  THUMB_IP    = 3;  THUMB_TIP    = 4
INDEX_MCP    = 5;  INDEX_PIP    = 6;  INDEX_DIP   = 7;  INDEX_TIP    = 8
MIDDLE_MCP   = 9;  MIDDLE_PIP   = 10; MIDDLE_DIP  = 11; MIDDLE_TIP   = 12
RING_MCP     = 13; RING_PIP     = 14; RING_DIP    = 15; RING_TIP     = 16
PINKY_MCP    = 17; PINKY_PIP    = 18; PINKY_DIP   = 19; PINKY_TIP    = 20

# Fingertip and PIP (second joint) IDs for each finger
FINGERS = {
    "index":  (INDEX_TIP,  INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring":   (RING_TIP,   RING_PIP),
    "pinky":  (PINKY_TIP,  PINKY_PIP),
}


# ─────────────────────────────────────────────
# Landmark utilities
# ─────────────────────────────────────────────
def lm_to_px(landmark, w, h):
    """Convert normalized landmark to pixel coords."""
    return int(landmark.x * w), int(landmark.y * h)


def distance(p1, p2):
    """Euclidean distance between two (x,y) tuples."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ─────────────────────────────────────────────
# Finger state detection
# ─────────────────────────────────────────────
def is_finger_up(lm, tip_id, pip_id):
    """
    A finger is 'up' when its tip is higher on screen than its PIP joint.
    (lower y value = higher on screen)
    """
    return lm[tip_id].y < lm[pip_id].y


def is_thumb_up(lm, hand_label):
    """
    Thumb is extended when tip is to the side of the IP joint.
    Direction depends on which hand (left vs right).
    """
    if hand_label == "Right":
        return lm[THUMB_TIP].x < lm[THUMB_IP].x
    else:
        return lm[THUMB_TIP].x > lm[THUMB_IP].x


def count_fingers(lm, hand_label):
    """Return count of extended fingers (0–5) and which ones are up."""
    states = {
        "thumb":  is_thumb_up(lm, hand_label),
        "index":  is_finger_up(lm, INDEX_TIP,  INDEX_PIP),
        "middle": is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        "ring":   is_finger_up(lm, RING_TIP,   RING_PIP),
        "pinky":  is_finger_up(lm, PINKY_TIP,  PINKY_PIP),
    }
    count = sum(states.values())
    return count, states


# ─────────────────────────────────────────────
# Gesture recognition
# ─────────────────────────────────────────────
def recognize_gesture(lm, hand_label, w, h):
    """
    Identify common gestures from finger states and positions.
    Returns a gesture name string.
    """
    count, states = count_fingers(lm, hand_label)

    # Thumbs up: only thumb is up
    if states["thumb"] and not any([
        states["index"], states["middle"], states["ring"], states["pinky"]
    ]):
        return "Thumbs Up"

    # Thumbs down: thumb down, all others down, wrist higher than thumb
    if not states["thumb"] and count == 0:
        if lm[WRIST].y < lm[THUMB_TIP].y:
            return "Thumbs Down"
        return "Fist"

    # Peace / Victory: index + middle up, rest down
    if (states["index"] and states["middle"]
            and not states["ring"] and not states["pinky"]):
        return "Peace"

    # Point: only index up
    if (states["index"] and not states["middle"]
            and not states["ring"] and not states["pinky"]):
        return "Pointing"

    # Rock on (index + pinky up)
    if (states["index"] and states["pinky"]
            and not states["middle"] and not states["ring"]):
        return "Rock On!"

    # OK sign: thumb-index pinch, other fingers up
    thumb_px = lm_to_px(lm[THUMB_TIP], w, h)
    index_px = lm_to_px(lm[INDEX_TIP], w, h)
    if distance(thumb_px, index_px) < 40 and states["middle"]:
        return "OK"

    # Open hand
    if count == 5:
        return "Open Hand"

    # Pinch (thumb + index close together)
    if distance(thumb_px, index_px) < 50:
        return "Pinch"

    return f"{count} fingers"


# ─────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────
def draw_skeleton(frame, lm, w, h):
    """Draw colored hand skeleton on the frame."""
    # Finger colors (BGR)
    FINGER_COLORS = {
        "thumb":  (255, 150, 0),
        "index":  (0, 255, 0),
        "middle": (0, 200, 255),
        "ring":   (255, 0, 200),
        "pinky":  (200, 100, 255),
    }
    PALM_COLOR = (200, 200, 200)

    def pt(id): return lm_to_px(lm[id], w, h)
    def line(a, b, color, thick=2):
        cv2.line(frame, pt(a), pt(b), color, thick)
    def dot(id, color, r=5):
        cv2.circle(frame, pt(id), r, color, -1)

    # Palm connections
    palm_ids = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP, WRIST]
    for i in range(len(palm_ids) - 1):
        line(palm_ids[i], palm_ids[i + 1], PALM_COLOR, 1)
    line(THUMB_CMC, INDEX_MCP, PALM_COLOR, 1)

    # Thumb
    for a, b in [(WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP),
                 (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP)]:
        line(a, b, FINGER_COLORS["thumb"], 3)

    # Four fingers
    finger_chains = {
        "index":  [INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP],
        "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
        "ring":   [RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP],
        "pinky":  [PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP],
    }
    for name, chain in finger_chains.items():
        color = FINGER_COLORS[name]
        for i in range(len(chain) - 1):
            line(chain[i], chain[i + 1], color, 3)

    # Landmark dots
    for i in range(21):
        dot(i, (255, 255, 255), r=4)
        dot(i, (80, 80, 80), r=2)   # dark center for contrast


def draw_landmark_labels(frame, lm, w, h):
    """Draw landmark index numbers (useful for learning)."""
    for i, point in enumerate(lm):
        x, y = lm_to_px(point, w, h)
        cv2.putText(frame, str(i), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 100), 1)


def draw_finger_status(frame, states, count, gesture, x_offset=10, y_offset=70):
    """Draw a finger status panel."""
    icons = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "P"}
    for i, (name, up) in enumerate(states.items()):
        color  = (0, 255, 100) if up else (80, 80, 80)
        label  = icons[name]
        xp     = x_offset + i * 34
        # Finger rectangle
        cv2.rectangle(frame, (xp, y_offset), (xp + 28, y_offset + 44), color, -1)
        cv2.putText(frame, label, (xp + 7, y_offset + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Count badge
    cv2.rectangle(frame, (x_offset, y_offset + 50), (x_offset + 170, y_offset + 80),
                  (40, 40, 40), -1)
    cv2.putText(frame, f"Count: {count}   {gesture}",
                (x_offset + 5, y_offset + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_pinch_meter(frame, lm, w, h):
    """Show a visual pinch distance meter."""
    thumb_px = lm_to_px(lm[THUMB_TIP], w, h)
    index_px = lm_to_px(lm[INDEX_TIP], w, h)
    dist     = distance(thumb_px, index_px)

    # Line between thumb and index tip
    cv2.line(frame, thumb_px, index_px, (255, 220, 0), 2)
    mid = ((thumb_px[0] + index_px[0]) // 2,
           (thumb_px[1] + index_px[1]) // 2)
    cv2.putText(frame, f"{int(dist)}px", (mid[0] + 5, mid[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("Hand Gesture Detection (MediaPipe)")
    print("Controls: Q=quit  L=labels  S=skeleton")
    print("=" * 50)

    # ── MediaPipe setup ────────────────────────
    mp_hands    = mp.solutions.hands
    mp_drawing  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,      # False = video stream (faster)
        max_num_hands=2,              # detect up to 2 hands
        min_detection_confidence=0.7, # how confident before reporting a hand
        min_tracking_confidence=0.5   # how confident to keep tracking
    )

    # ── Webcam setup ───────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_labels   = False
    show_skeleton = True
    prev_time     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Convert and process ────────────────
        # MediaPipe needs RGB, OpenCV gives BGR
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # ── Process each detected hand ─────────
        if result.multi_hand_landmarks:
            for hand_lms, hand_info in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                lm         = hand_lms.landmark
                hand_label = hand_info.classification[0].label   # "Left" or "Right"
                confidence = hand_info.classification[0].score

                # Draw skeleton
                if show_skeleton:
                    draw_skeleton(frame, lm, w, h)

                # Landmark labels (for learning)
                if show_labels:
                    draw_landmark_labels(frame, lm, w, h)

                # Gesture analysis
                count, states = count_fingers(lm, hand_label)
                gesture       = recognize_gesture(lm, hand_label, w, h)

                # Pinch meter
                draw_pinch_meter(frame, lm, w, h)

                # Finger status panel (offset second hand to the right)
                x_off = 10 if hand_label == "Right" else w - 190
                draw_finger_status(frame, states, count, gesture, x_off, 70)

                # Hand label + confidence above wrist
                wrist_px = lm_to_px(lm[WRIST], w, h)
                cv2.putText(frame, f"{hand_label} ({confidence:.0%})",
                            (wrist_px[0] - 40, wrist_px[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Large gesture text
                gesture_color = (0, 255, 150) if count > 0 else (0, 100, 255)
                gy = h - 50 if hand_label == "Right" else h - 20
                cv2.putText(frame, gesture, (10, gy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, gesture_color, 3)

        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

        # ── FPS ────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time = curr_time
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        # ── Top bar hints ──────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "L=labels  S=skeleton  Q=quit",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("05 - Hand Gestures", frame)

        # ── Key handling ───────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_labels = not show_labels
            print(f"Landmark labels: {'ON' if show_labels else 'OFF'}")
        elif key == ord('s'):
            show_skeleton = not show_skeleton
            print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
