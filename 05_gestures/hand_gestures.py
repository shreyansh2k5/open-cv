"""
Phase 5 — Hand Gesture Detection
==================================
Uses MediaPipe Hands to detect and interpret hand gestures in real time.

Detects:
  - Finger count (0–5)
  - Named gestures: thumbs up, peace sign, fist, open hand, OK sign, pointing

Install: pip install opencv-python mediapipe numpy

Run: python hand_gestures.py
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# MediaPipe landmark indices reference
#
#   WRIST = 0
#   THUMB:  CMC=1  MCP=2  IP=3   TIP=4
#   INDEX:  MCP=5  PIP=6  DIP=7  TIP=8
#   MIDDLE: MCP=9  PIP=10 DIP=11 TIP=12
#   RING:   MCP=13 PIP=14 DIP=15 TIP=16
#   PINKY:  MCP=17 PIP=18 DIP=19 TIP=20
# ─────────────────────────────────────────────

FINGER_TIPS = [4, 8, 12, 16, 20]   # Tip landmark IDs for each finger
FINGER_PIPS = [3, 6, 10, 14, 18]   # PIP (second joint) IDs — used for up/down check


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class HandInfo:
    landmarks:    list           # 21 landmark objects (normalized 0–1)
    fingers_up:   list[bool]     # [thumb, index, middle, ring, pinky]
    finger_count: int
    gesture:      str
    handedness:   str            # "Left" or "Right"
    bbox:         tuple          # (x, y, w, h) in pixels


# ─────────────────────────────────────────────
# 1. Finger state detection
# ─────────────────────────────────────────────
def get_fingers_up(landmarks, handedness: str) -> list[bool]:
    """
    Returns a list of 5 booleans — True if that finger is raised.
    Order: [thumb, index, middle, ring, pinky]
    """
    lm = landmarks
    fingers = []

    # ── Thumb ──────────────────────────────
    # Thumb is special: compare x positions (left/right) instead of y
    # For a right hand, thumb tip (4) should be to the LEFT of thumb IP (3) when up
    if handedness == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # ── Fingers (index → pinky) ───────────
    # A finger is up if its TIP y-coordinate is ABOVE its PIP y-coordinate
    # (in image coordinates, y increases downward, so tip.y < pip.y means raised)
    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(lm[tip_id].y < lm[pip_id].y)

    return fingers


# ─────────────────────────────────────────────
# 2. Distance between two landmarks (normalized)
# ─────────────────────────────────────────────
def landmark_distance(lm, id1: int, id2: int) -> float:
    """Euclidean distance between two landmarks (in normalized coordinates)."""
    dx = lm[id1].x - lm[id2].x
    dy = lm[id1].y - lm[id2].y
    return math.sqrt(dx * dx + dy * dy)


# ─────────────────────────────────────────────
# 3. Gesture classifier
# ─────────────────────────────────────────────
def classify_gesture(fingers_up: list[bool], landmarks) -> str:
    """
    Map finger states + landmark geometry to a named gesture.
    Extend this function to add your own gestures.
    """
    thumb, index, middle, ring, pinky = fingers_up
    count = sum(fingers_up)

    # ── 0 fingers: Fist ──
    if count == 0:
        return "Fist"

    # ── 5 fingers: Open hand ──
    if count == 5:
        return "Open hand"

    # ── Thumbs up: only thumb up ──
    if fingers_up == [True, False, False, False, False]:
        return "Thumbs up"

    # ── Thumbs down: only thumb up but pointing down (tip below wrist) ──
    if fingers_up == [True, False, False, False, False]:
        if landmarks[4].y > landmarks[0].y:
            return "Thumbs down"

    # ── Pointing: only index up ──
    if fingers_up == [False, True, False, False, False]:
        return "Pointing"

    # ── Peace / V sign: index + middle up ──
    if fingers_up == [False, True, True, False, False]:
        return "Peace"

    # ── OK sign: thumb and index form a circle, others raised ──
    if count == 3 and middle and ring and pinky:
        d = landmark_distance(landmarks, 4, 8)
        if d < 0.07:
            return "OK"

    # ── Rock: index + pinky up ──
    if fingers_up == [False, True, False, False, True]:
        return "Rock"

    # ── Call me: thumb + pinky up ──
    if fingers_up == [True, False, False, False, True]:
        return "Call me"

    # ── Three: index + middle + ring ──
    if fingers_up == [False, True, True, True, False]:
        return "Three"

    # ── Four: all except thumb ──
    if fingers_up == [False, True, True, True, True]:
        return "Four"

    return f"{count} fingers"


# ─────────────────────────────────────────────
# 4. Bounding box from landmarks
# ─────────────────────────────────────────────
def get_bounding_box(landmarks, img_w: int, img_h: int,
                     padding: int = 20) -> tuple:
    """Return pixel bounding box (x, y, w, h) around all landmarks."""
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x  = max(0, int(min(xs)) - padding)
    y  = max(0, int(min(ys)) - padding)
    w  = min(img_w - x, int(max(xs) - min(xs)) + 2 * padding)
    h  = min(img_h - y, int(max(ys) - min(ys)) + 2 * padding)
    return x, y, w, h


# ─────────────────────────────────────────────
# 5. Annotate frame
# ─────────────────────────────────────────────
def annotate_frame(frame: np.ndarray,
                   hand: HandInfo,
                   mp_drawing,
                   mp_hands,
                   hand_landmarks_obj) -> np.ndarray:
    """Draw landmarks, connections, bounding box, and gesture label."""
    h, w = frame.shape[:2]

    # ── Draw skeleton ──────────────────────
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks_obj,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(200, 200, 0), thickness=2),
    )

    # ── Bounding box ───────────────────────
    x, y, bw, bh = hand.bbox
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)

    # ── Gesture label ──────────────────────
    label_bg_y = max(0, y - 40)
    cv2.rectangle(frame, (x, label_bg_y), (x + bw, y), (0, 200, 255), -1)
    cv2.putText(frame, hand.gesture, (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # ── Finger count circles ───────────────
    for i, (is_up, tip_id) in enumerate(zip(hand.fingers_up, FINGER_TIPS)):
        lm  = hand.landmarks[tip_id]
        px  = int(lm.x * w)
        py  = int(lm.y * h)
        color = (0, 255, 100) if is_up else (0, 80, 200)
        cv2.circle(frame, (px, py), 10, color, -1)

    # ── Handedness label ───────────────────
    cv2.putText(frame, hand.handedness,
                (x + 5, y + bh + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return frame


# ─────────────────────────────────────────────
# 6. Main detection loop
# ─────────────────────────────────────────────
def run_gesture_detection(camera_index: int = 0,
                           max_hands: int = 2,
                           min_detection_conf: float = 0.7,
                           min_tracking_conf: float = 0.6) -> None:
    """
    Real-time hand gesture detection loop.

    Keys:
      q — quit
      s — save screenshot
      l — toggle landmark display
    """
    # ── MediaPipe setup ───────────────────
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands_detector = mp_hands.Hands(
        static_image_mode=False,     # False = video mode (tracks across frames)
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf,
    )

    # ── Camera setup ──────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_landmarks = True
    screenshot_n   = 0
    prev_time      = 0

    print("Hand gesture detection started.")
    print("Keys: q=quit | s=screenshot | l=toggle landmarks")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)   # mirror for natural selfie feel
        h, w   = frame.shape[:2]

        # ── Convert BGR → RGB for MediaPipe ─
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        detected_hands: list[HandInfo] = []

        # ── Process each detected hand ──────
        if result.multi_hand_landmarks:
            for hand_lms_obj, classification in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                lm_list    = hand_lms_obj.landmark
                handedness = classification.classification[0].label  # "Left"/"Right"

                fingers    = get_fingers_up(lm_list, handedness)
                count      = sum(fingers)
                gesture    = classify_gesture(fingers, lm_list)
                bbox       = get_bounding_box(lm_list, w, h)

                hand = HandInfo(
                    landmarks=lm_list,
                    fingers_up=fingers,
                    finger_count=count,
                    gesture=gesture,
                    handedness=handedness,
                    bbox=bbox,
                )
                detected_hands.append(hand)

                if show_landmarks:
                    annotate_frame(frame, hand, mp_drawing,
                                   mp_hands, hand_lms_obj)

        # ── HUD ─────────────────────────────
        curr_time = time.time() if 'time' in dir() else 0
        import time as _time
        curr_time = _time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands: {len(detected_hands)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if detected_hands:
            h0 = detected_hands[0]
            finger_str = "".join(["1" if f else "0" for f in h0.fingers_up])
            cv2.putText(frame, f"Fingers: {finger_str} ({h0.finger_count})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

        hint = "q=quit | s=screenshot | l=landmarks"
        cv2.putText(frame, hint, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Hand Gesture Detection", frame)

        # ── Keys ────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_n += 1
            fname = f"gesture_{screenshot_n:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks

    hands_detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ─────────────────────────────────────────────
# 7. Gesture-controlled counter demo
# ─────────────────────────────────────────────
def run_finger_counter() -> None:
    """
    Simple demo: show just the finger count as a large number.
    Great first project to verify everything works.
    """
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_det  = mp_hands.Hands(max_num_hands=1,
                                min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Finger Counter — hold up fingers! Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_det.process(rgb)

        count = 0
        if result.multi_hand_landmarks:
            for hand_lms, classification in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                handedness = classification.classification[0].label
                fingers    = get_fingers_up(hand_lms.landmark, handedness)
                count      = sum(fingers)
                mp_drawing.draw_landmarks(frame, hand_lms,
                                          mp_hands.HAND_CONNECTIONS)

        # Big number display
        cv2.putText(frame, str(count), (260, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 100), 20)
        cv2.putText(frame, "fingers up", (200, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands_det.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=== Phase 5: Hand Gesture Detection ===\n")
    print("Choose demo:")
    print("  1. Full gesture detection (names gestures, shows skeleton)")
    print("  2. Finger counter (simple count display)")

    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "2":
        run_finger_counter()
    else:
        run_gesture_detection(
            camera_index=0,
            max_hands=2,
            min_detection_conf=0.7,
            min_tracking_conf=0.6,
        )


if __name__ == "__main__":
    main()
