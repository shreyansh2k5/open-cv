"""
03_color_detection.py — Real-Time Color Detection
===================================================
Topics covered:
  - HSV color space for robust color detection
  - Creating color masks with inRange()
  - Finding and drawing contours
  - Bounding boxes around detected objects
  - Live webcam color tracker

Run:
    python 03_color_detection.py

Controls:
    Q — quit
    1 — track RED
    2 — track GREEN
    3 — track BLUE
    4 — track YELLOW
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# HSV Color Ranges
# ─────────────────────────────────────────────
# H: 0-179 (hue/color)   S: 0-255 (saturation)   V: 0-255 (brightness)
# Tip: Use a color picker to find HSV values for any color
# In OpenCV, Hue range is 0-179 (not 0-360 like most tools — divide by 2)

COLOR_RANGES = {
    "RED": [
        # Red wraps around in HSV, so we need two ranges
        (np.array([0,   120, 70]),  np.array([10,  255, 255])),
        (np.array([170, 120, 70]),  np.array([180, 255, 255])),
    ],
    "GREEN": [
        (np.array([35, 80, 50]),    np.array([85,  255, 255])),
    ],
    "BLUE": [
        (np.array([100, 80, 50]),   np.array([130, 255, 255])),
    ],
    "YELLOW": [
        (np.array([20, 100, 100]),  np.array([35,  255, 255])),
    ],
}

# Display color for each (BGR)
DISPLAY_COLORS = {
    "RED":    (0,   0,   255),
    "GREEN":  (0,   255, 0),
    "BLUE":   (255, 0,   0),
    "YELLOW": (0,   220, 255),
}

COLOR_KEYS = list(COLOR_RANGES.keys())
active_color = "GREEN"   # default


def create_mask(hsv_frame, color_name):
    """
    Create a binary mask for the given color.
    White pixels = color detected, Black = not detected.
    """
    ranges = COLOR_RANGES[color_name]
    mask = None
    for (lower, upper) in ranges:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def clean_mask(mask):
    """
    Remove noise from mask using morphological operations.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)    # remove tiny blobs
    mask = cv2.dilate(mask, kernel, iterations=2)   # fill gaps
    return mask


def find_objects(mask, min_area=1500):
    """
    Find contours in the mask and return bounding boxes for large-enough ones.
    Returns list of (x, y, w, h, area, center_x, center_y)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # skip tiny noise blobs

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2   # center x
        cy = y + h // 2   # center y
        objects.append((x, y, w, h, area, cx, cy))

    # Sort by area, largest first
    objects.sort(key=lambda o: o[4], reverse=True)
    return objects


def draw_detections(frame, objects, color, color_name):
    """Draw bounding boxes and labels on detected objects."""
    for i, (x, y, w, h, area, cx, cy) in enumerate(objects):
        # Bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Center dot
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Label
        label = f"{color_name} #{i+1}  area:{int(area)}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Crosshair lines to center
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), color, 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), color, 1)

    return frame


def draw_ui(frame, active_color, objects):
    """Draw the HUD overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Active color indicator
    dot_color = DISPLAY_COLORS[active_color]
    cv2.circle(frame, (20, 22), 10, dot_color, -1)
    cv2.putText(frame, f"Tracking: {active_color}  |  Objects found: {len(objects)}",
                (38, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Key hints
    hint = "Keys: 1=RED  2=GREEN  3=BLUE  4=YELLOW  Q=quit"
    cv2.putText(frame, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
print("=" * 50)
print("Color Detection")
print("Hold a colored object in front of your webcam")
print("Keys: 1=RED  2=GREEN  3=BLUE  4=YELLOW  Q=quit")
print("=" * 50)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    frame = cv2.flip(frame, 1)   # mirror for natural feel

    # ── Step 1: Convert to HSV ─────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── Step 2: Create color mask ──────────────
    mask = create_mask(hsv, active_color)

    # ── Step 3: Clean the mask ─────────────────
    mask_clean = clean_mask(mask)

    # ── Step 4: Find objects via contours ──────
    objects = find_objects(mask_clean, min_area=1500)

    # ── Step 5: Draw results on frame ──────────
    draw_color = DISPLAY_COLORS[active_color]
    frame = draw_detections(frame, objects, draw_color, active_color)
    frame = draw_ui(frame, active_color, objects)

    # ── Step 6: Show raw mask in corner ────────
    mask_small = cv2.resize(mask_clean, (160, 120))
    mask_bgr   = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    frame[50:170, 0:160] = mask_bgr
    cv2.putText(frame, "Mask", (5, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("03 - Color Detection", frame)

    # ── Key handling ───────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        active_color = "RED"
        print(f"Switched to: {active_color}")
    elif key == ord('2'):
        active_color = "GREEN"
        print(f"Switched to: {active_color}")
    elif key == ord('3'):
        active_color = "BLUE"
        print(f"Switched to: {active_color}")
    elif key == ord('4'):
        active_color = "YELLOW"
        print(f"Switched to: {active_color}")

cap.release()
cv2.destroyAllWindows()
print("Done!")
