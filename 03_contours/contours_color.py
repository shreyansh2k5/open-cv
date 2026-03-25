"""
Phase 3 — Contours & Color Detection
======================================
Learn how to:
  - Find and draw contours
  - Extract bounding boxes and area
  - Detect objects by color using HSV masking
  - Build a real-time colored object tracker

Run: python contours_color.py
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# HSV color ranges for common colors
# Tip: use the tune_hsv_ranges() function below to find your own
# ─────────────────────────────────────────────
COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]),   np.array([10, 255, 255])),   # lower red
        (np.array([170, 120, 70]), np.array([180, 255, 255])),  # upper red (wraps)
    ],
    "green": [
        (np.array([40, 60, 60]),   np.array([80, 255, 255])),
    ],
    "blue": [
        (np.array([100, 100, 50]), np.array([130, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
    "orange": [
        (np.array([10, 100, 100]), np.array([25, 255, 255])),
    ],
}


# ─────────────────────────────────────────────
# 1. Contour Detection
# ─────────────────────────────────────────────
def find_and_draw_contours(img: np.ndarray) -> np.ndarray:
    """
    Find all contours in an image and annotate them.
    Returns image with contours drawn.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # Find contours from the edge map
    # RETR_EXTERNAL: only outermost contours (ignores holes inside shapes)
    # CHAIN_APPROX_SIMPLE: compresses horizontal/vertical/diagonal segments
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    print(f"Found {len(contours)} contours")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 500:          # skip tiny noise contours
            continue

        # Draw contour outline
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        # Bounding rectangle (axis-aligned)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(output, (int(cx), int(cy)), int(radius), (0, 0, 255), 1)

        # Centroid using moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            cv2.circle(output, (centroid_x, centroid_y), 4, (0, 255, 255), -1)

        # Label with area
        cv2.putText(output, f"A={int(area)}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return output


# ─────────────────────────────────────────────
# 2. Shape Classification by Contour Approximation
# ─────────────────────────────────────────────
def classify_shape(contour) -> str:
    """Approximate a contour to a polygon and name the shape."""
    perimeter = cv2.arcLength(contour, True)
    # epsilon: how much the approximated shape can deviate from the real contour
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides  = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    else:
        return "Circle" if sides > 8 else f"Polygon ({sides})"


def detect_shapes(img: np.ndarray) -> np.ndarray:
    """Detect and label shapes in an image."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()

    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue

        shape = classify_shape(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.drawContours(output, [contour], -1, (0, 200, 0), 2)
        cv2.putText(output, shape, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return output


# ─────────────────────────────────────────────
# 3. Color Mask Creation
# ─────────────────────────────────────────────
def create_color_mask(frame: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask that is white where the chosen color exists.
    Works in HSV space for more robust color detection.
    """
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown color: {color_name}. Choose from {list(COLOR_RANGES.keys())}")

    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = np.zeros(frame.shape[:2], dtype=np.uint8)

    for lower, upper in COLOR_RANGES[color_name]:
        mask |= cv2.inRange(hsv, lower, upper)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # remove small noise
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes

    return mask


# ─────────────────────────────────────────────
# 4. Real-Time Colored Object Tracker (webcam)
# ─────────────────────────────────────────────
def track_colored_object(color_name: str = "red") -> None:
    """
    Open webcam and track the largest object of the given color.
    Draws bounding box and centroid trail.
    Press 'q' to quit, 1-5 to switch color.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    color_keys   = list(COLOR_RANGES.keys())
    active_color = color_name
    trail        = []          # store recent centroid positions
    MAX_TRAIL    = 30

    print(f"Tracking: {active_color}")
    print("Keys: q=quit | 1=red | 2=green | 3=blue | 4=yellow | 5=orange")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror view
        mask  = create_color_mask(frame, active_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame.copy()

        if contours:
            # Track the largest contour
            largest = max(contours, key=cv2.contourArea)
            area    = cv2.contourArea(largest)

            if area > 1000:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy     = x + w // 2, y + h // 2

                # Draw bounding box
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw centroid
                cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

                # Trail
                trail.append((cx, cy))
                if len(trail) > MAX_TRAIL:
                    trail.pop(0)

                for i in range(1, len(trail)):
                    thickness = int(i * 3 / MAX_TRAIL) + 1
                    cv2.line(output, trail[i - 1], trail[i], (0, 255, 255), thickness)

                # Info label
                cv2.putText(output, f"{active_color} | area={int(area)}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # Show mask as small inset (top-right corner)
        mask_small = cv2.resize(mask, (160, 120))
        mask_bgr   = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        output[10:130, output.shape[1] - 170:output.shape[1] - 10] = mask_bgr
        cv2.putText(output, "mask", (output.shape[1] - 160, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # HUD
        cv2.putText(output, f"Tracking: {active_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, "q=quit | 1-5=color", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Color Tracker", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            idx          = key - ord('1')
            active_color = color_keys[idx % len(color_keys)]
            trail.clear()
            print(f"Switched to: {active_color}")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 5. Interactive HSV Tuner (find your own color ranges)
# ─────────────────────────────────────────────
def tune_hsv_ranges() -> None:
    """
    Open webcam with trackbars to find the right HSV range
    for any color you want to detect.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H min", "HSV Tuner", 0,   179, lambda x: None)
    cv2.createTrackbar("H max", "HSV Tuner", 179, 179, lambda x: None)
    cv2.createTrackbar("S min", "HSV Tuner", 0,   255, lambda x: None)
    cv2.createTrackbar("S max", "HSV Tuner", 255, 255, lambda x: None)
    cv2.createTrackbar("V min", "HSV Tuner", 0,   255, lambda x: None)
    cv2.createTrackbar("V max", "HSV Tuner", 255, 255, lambda x: None)

    print("Adjust trackbars to isolate your target color.")
    print("Press 'p' to print current values. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask  = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("HSV Tuner",    frame)
        cv2.imshow("Mask",         mask)
        cv2.imshow("Masked result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\nLower: {lower}")
            print(f"Upper: {upper}")
            print(f"Add to COLOR_RANGES as:")
            print(f'  (np.array({list(lower)}), np.array({list(upper)})),')

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=== Phase 3: Contours & Color Detection ===\n")
    print("Choose demo:")
    print("  1. Contour detection (no webcam needed)")
    print("  2. Shape detection (no webcam needed)")
    print("  3. Real-time color tracker (webcam)")
    print("  4. HSV range tuner (webcam)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        # Build test image with shapes
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50),   (200, 200), (255, 100, 0), -1)
        cv2.circle(img,    (380, 150),  80,         (0, 200, 100), -1)
        cv2.circle(img,    (200, 320),  60,         (100, 0, 200), -1)
        cv2.rectangle(img, (350, 250), (550, 380),  (0, 150, 200), -1)

        result = find_and_draw_contours(img)
        cv2.imshow("Original", img)
        cv2.imshow("Contours", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == "2":
        img = np.zeros((500, 700, 3), dtype=np.uint8)
        # Triangle
        pts = np.array([[100, 400], [250, 100], [400, 400]])
        cv2.fillPoly(img, [pts], (200, 200, 200))
        # Square
        cv2.rectangle(img, (440, 100), (640, 300), (180, 180, 180), -1)
        # Circle
        cv2.circle(img, (160, 260), 80, (150, 150, 150), -1)

        result = detect_shapes(img)
        cv2.imshow("Shape Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == "3":
        color = input("Track which color? (red/green/blue/yellow/orange) [red]: ").strip() or "red"
        track_colored_object(color)

    elif choice == "4":
        tune_hsv_ranges()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
